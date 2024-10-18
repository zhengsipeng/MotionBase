import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, 
                dataset_name, 
                feat_bias = 5, 
                unit_length = 4, 
                codebook_size = 1024, 
                tokenizer_name=None, 
                split_name=None, 
                meta_dir='', 
                tokenizer=None,
                train_target='t2m'
                ):
        
        # self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.train_target = train_target
        print(f'train_target: {self.train_target}')

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1

        self.data_root = '/share/zsp/datasets/motion/text-motion-all'
        self.motion_dir = pjoin(self.data_root, 'vector_263_20')
        self.text_dir = pjoin(self.data_root, 'motion_seq_text_with_prefix_pos')
        self.joints_num = 22
        radius = 4
        fps = 20
        self.max_motion_length = 26 if unit_length == 8 else 51
        dim_pose = 263
        self.meta_dir = meta_dir

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, self.split_name)

        print('meta_dir:', self.meta_dir)
        print('split_file:', split_file)

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        cnt = 0
        error_dict = {}
        good_dict = {}
        for name in tqdm(id_list):
            subset_name = name.split('/')[0]
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, f'{name}.npy'))
                cnt += 1
                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens

                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                if len(m_token_list_new) == 0:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    print(len(m_token_list))
                                    print(int(f_tag*fps/unit_length), int(to_tag*fps/unit_length))
                                    continue

                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                    'text':[text_dict]}

                                new_name_list.append(new_name)
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, name)
                            # print(asd)
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text': text_data}

                    new_name_list.append(name)

                if subset_name not in good_dict:
                    good_dict[subset_name] = 0
                good_dict[subset_name] += 1
            except Exception as e:

                if subset_name not in error_dict:
                    error_dict[subset_name] = 0
                error_dict[subset_name] += 1
                pass

        print(cnt)
        print('new_name_list:', len(new_name_list))
        print('error_dict:', error_dict)
        print('good_dict:', good_dict)

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[self.name_list[idx]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption = text_data['caption']
        
        coin = np.random.choice([False, False, True])
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_str = ''.join([f'<motion_id_{token}>' for token in m_tokens])
        m_str = f"<Motion Token>{m_str}</Motion Token>"
        # m_tokens_len = m_tokens.shape[0]

        if self.train_target == "t2m+m2t":
            object_choice = random.choice(['t2m', 'm2t'])
        elif self.train_target == "t2m":
            object_choice = 't2m'
        elif self.train_target == "m2t":
            object_choice = 'm2t'

        if object_choice == 't2m':
            text = f'{caption}\n{m_str}'
        elif object_choice == 'm2t':
        elif object_choice == 'm2m':
            text = f'Motion completion: {m_str}'
        elif object_choice == 't2t':
            text = f'Text completion: {caption}'

        return text


'''For use of training text-2-motion generative model'''
class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, meta_dir="", split_name=""):
        
        self.split_name = split_name

        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        
        self.data_root = '/share/zsp/datasets/motion/text-motion-all'
        self.motion_dir = pjoin(self.data_root, 'vector_263_20')
        self.text_dir = pjoin(self.data_root, 'motion_seq_text_with_prefix_pos')
        self.joints_num = 22
        radius = 4
        fps = 20
        self.max_motion_length = 196
        dim_pose = 263
        self.meta_dir = meta_dir

        print('meta_dir:', self.meta_dir)

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        split_file = pjoin(self.data_root, self.split_name)
        print('split_file:', split_file)

        min_motion_len = 40

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        if line.strip() == '':
                            continue
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]

                        if line_split[1] == '':
                            continue
                        
                        tokens = line_split[1].split(' ')

                        skip_flag = False
                        for item in tokens:
                            # if '/' not in t:
                            #     skip_flag = True
                            try:
                                word, pos = item.split('/')
                            except:
                                skip_flag = True
                                print(item)
                                break
                                # print(asd)
                        if skip_flag:
                            print(name)
                            print(line)
                            print(line_split)
                            print(caption)
                            print(tokens)
                            print(skip_flag)
                            
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass
        print('eval_new_name_list:', len(new_name_list))
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        # data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        caption = f"{caption}\n"
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name

def ValDATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4,
                meta_dir = "", split_name = "") : 
    
    valSet = Text2MotionDatasetEval(dataset_name, is_test, w_vectorizer, unit_length=unit_length, meta_dir=meta_dir, split_name=split_name)
    val_loader = torch.utils.data.DataLoader(valSet,
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    return val_loader#, valSet.mean, valSet.std