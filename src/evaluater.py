import os
import re

import clip
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric

from transformers import GenerationConfig
from transformers import Trainer
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Sequence, List

def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

# def my_custom_eval_func(trainer: Trainer, eval_dataset: Dataset) -> Dict[str, float]:
#     """
#     自定义评估函数，使用模型在评估集上进行预测，并计算准确率。
#     """
#     model = trainer.model
#     tokenizer = trainer.tokenizer
#     dataloader = DataLoader(eval_dataset, batch_size=trainer.args.eval_batch_size)

#     total_correct = 0
#     total_samples = 0

#     model.eval()  # 将模型设置为评估模式
#     with torch.no_grad():  # 不计算梯度
#         for batch in dataloader:
#             inputs = tokenizer(batch["input_str"], padding=True, truncation=True, return_tensors="pt").to(trainer.args.device)
#             outputs = model(**inputs)  # 使用模型进行预测
#             logits = outputs.logits  # 获取模型输出的 logits

#             # 假设这是一个分类任务，计算预测的类别
#             predictions = torch.argmax(logits, dim=-1)

#             # 假设 batch["labels"] 包含 ground truth 标签
#             labels = batch["labels"].to(trainer.args.device)

#             # 比较预测和标签，计算准确率
#             correct = (predictions == labels).sum().item()
#             total_correct += correct
#             total_samples += labels.size(0)

#     accuracy = total_correct / total_samples  # 计算整体准确率

#     metrics = {"accuracy": accuracy}  # 将准确率添加到指标字典中
#     return metrics

@torch.no_grad()        
def evaluation_transformer(trainer: Trainer, eval_dataset: Dataset, val_loader=None, net=None, eval_wrapper=None) -> Dict[str, float]:
    """
    自定义评估函数，使用模型在评估集上进行预测。
    """
    model = trainer.model
    tokenizer = trainer.tokenizer
    train_dataset = trainer.train_dataset
    # print(eval_dataset)
    # print(len(eval_dataset))
    # val_loader = DataLoader(eval_dataset,
    #                         32,
    #                         shuffle = True,
    #                         num_workers=8,
    #                         collate_fn=collate_fn,
    #                         drop_last = True)
    net.to(trainer.args.device)
    model.eval()
    nb_sample = 0
    
    # draw_org = []
    # draw_pred = []
    # draw_text = []
    # draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # eos_token_id = tokenizer.eos_token_id
    # print(tokenizer.pad_token_id)
    # print(asd)

    # 创建 GenerationConfig 对象
    generation_config = GenerationConfig(
        max_new_tokens=100,
        pad_token_id = tokenizer.pad_token_id,
        # pad_token_id = 128004,
    )

    nb_sample = 0
    cntttttt = 0
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(trainer.args.device)
        pred_len = torch.ones(bs).long()

        # print(clip_text)
        # 使用Llama2生成运动序列
        inputs = tokenizer(clip_text, 
                        padding=True, 
                        truncation=True,
                        max_length=256,
                        add_special_tokens=True, 
                        return_tensors="pt").to(trainer.args.device)

        #         # 获取 BOS token 的 ID
        # bos_token_id = tokenizer.bos_token_id

        # # 在每个输入序列的开头插入 BOS token ID
        # inputs['input_ids'] = torch.cat([torch.tensor([[bos_token_id]]).repeat(inputs['input_ids'].shape[0], 1), inputs['input_ids']], dim=1).to(trainer.args.device)
        # inputs['attention_mask'] = torch.cat([torch.tensor([[1]]).repeat(inputs['attention_mask'].shape[0], 1), inputs['attention_mask']], dim=1).to(trainer.args.device)
        
        outputs = model.generate(**inputs, generation_config=generation_config)

        index_motions = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:])

        for k in range(bs):

            pred_str = index_motions[k]
            pred_numbers_str = re.findall(r'<motion_id_(\d+)>', pred_str.split("<Motion Token>")[-1].split("</Motion Token>")[0])
            if len(pred_numbers_str) == 0:
                index_motion = torch.ones(1,1).to(trainer.args.device).long()
            else:
                try:
                    pred_numbers = [int(n) for n in pred_numbers_str]
                    index_motion = torch.tensor([pred_numbers]).to(trainer.args.device)
                except:
                    index_motion = torch.ones(1,1).to(trainer.args.device).long()
            
            if (k == 0) and (cntttttt == 0):
                print('instruction:', clip_text[k])
                print('response:', index_motions[k])
                print('motion:', index_motion)
                cntttttt += 1

            pred_pose = net.forward_decoder(index_motion)
            cur_len = pred_pose.shape[1]

            pred_len[k] = min(cur_len, seq)

            ###
            pred_denorm = train_dataset.inv_transform(pred_pose.detach().cpu().numpy()) # to origin pose
            pred_pose_new = val_loader.dataset.forward_transform(pred_denorm)
            pred_pose = torch.from_numpy(pred_pose_new).float().cuda()
            ###

            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

            # if draw:
            #     # pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            #     pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

            #     if i == 0 and k < 4:
            #         draw_pred.append(pred_xyz)
            #         draw_text_pred.append(clip_text[k])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
        
        # if i == 0:
        pose = pose.cuda().float()
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        # if draw:
        #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
        #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


        #     for j in range(min(4, bs)):
        #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
        #         draw_text.append(clip_text[j])

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    # diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 50)
    # diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 50)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    metrics = {"FID": float(fid), 
                "Diversity Real": float(diversity_real), 
                "Diversity": float(diversity), 
                "R Top1 real": float(R_precision_real[0]), 
                "R Top2 real": float(R_precision_real[1]), 
                "R Top3 real": float(R_precision_real[2]), 
                "R Top1": float(R_precision[0]), 
                "R Top2": float(R_precision[1]), 
                "R Top3": float(R_precision[2]), 
                "match_score_real": float(matching_score_real), 
                "matching_score_pred": float(matching_score_pred)}  # 将准确率添加到指标字典中
    return metrics

@torch.no_grad()        
def evaluation_transformer_2(trainer: Trainer, eval_dataset: Dataset, val_loader=None, val_loader_2=None, net=None, eval_wrapper=None, eval_wrapper_2=None) -> Dict[str, float]:
    """
    自定义评估函数，使用模型在评估集上进行预测。
    """
    model = trainer.model
    tokenizer = trainer.tokenizer
    train_dataset = trainer.train_dataset

    net.to(trainer.args.device)
    model.eval()
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    print(tokenizer.pad_token_id)
    print(asd)

    # 创建 GenerationConfig 对象
    generation_config = GenerationConfig(
        max_new_tokens=100,
        pad_token_id = tokenizer.pad_token_id,
    )

    nb_sample = 0
    cntttttt = 0
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(trainer.args.device)
        pred_len = torch.ones(bs).long()

        # print(clip_text)
        # 使用Llama2生成运动序列
        inputs = tokenizer(clip_text, 
                        padding=True, 
                        truncation=True, 
                        add_special_tokens=True, 
                        return_tensors="pt").to(trainer.args.device)
        # print(inputs)
        # print(inputs["input_ids"])
        # print(asd)
        outputs = model.generate(**inputs, generation_config=generation_config)

        index_motions = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:])

        for k in range(bs):

            pred_str = index_motions[k]
            pred_numbers_str = re.findall(r'<motion_id_(\d+)>', pred_str.split("<Motion Token>")[-1].split("</Motion Token>")[0])
            if len(pred_numbers_str) == 0:
                index_motion = torch.ones(1,1).to(trainer.args.device).long()
            else:
                try:
                    pred_numbers = [int(n) for n in pred_numbers_str]
                    index_motion = torch.tensor([pred_numbers]).to(trainer.args.device)
                except:
                    index_motion = torch.ones(1,1).to(trainer.args.device).long()
            
            if (k == 0) and (cntttttt == 0):
                print('instruction:', clip_text[k], '\nresponse:', index_motions[k], '\nmotion:', index_motion)
                cntttttt += 1

            pred_pose = net.forward_decoder(index_motion)
            cur_len = pred_pose.shape[1]

            pred_len[k] = min(cur_len, seq)

            ###
            pred_denorm = train_dataset.inv_transform(pred_pose.detach().cpu().numpy()) # to origin pose

            pred_pose_new = val_loader.dataset.forward_transform(pred_denorm)
            pred_pose = torch.from_numpy(pred_pose_new).float().cuda()

            pred_pose_new_2 = val_loader_2.dataset.forward_transform(pred_denorm)
            pred_pose_2 = torch.from_numpy(pred_pose_new_2).float().cuda()
            ###

            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            pred_pose_eval_2[k:k+1, :cur_len] = pred_pose_2[:, :seq]

            # if draw:
            #     # pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            #     pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

            #     if i == 0 and k < 4:
            #         draw_pred.append(pred_xyz)
            #         draw_text_pred.append(clip_text[k])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
        et_pred_2, em_pred_2 = eval_wrapper_2.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval_2, pred_len)
        
        # if i == 0:
        pose = pose.cuda().float()
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        et_2, em_2 = eval_wrapper_2.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        # if draw:
        #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
        #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


        #     for j in range(min(4, bs)):
        #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
        #         draw_text.append(clip_text[j])

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    # diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 50)
    # diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 50)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    metrics = {"FID": float(fid), 
                "Diversity Real": float(diversity_real), 
                "Diversity": float(diversity), 
                "R Top1 real": float(R_precision_real[0]), 
                "R Top2 real": float(R_precision_real[1]), 
                "R Top3 real": float(R_precision_real[2]), 
                "R Top1": float(R_precision[0]), 
                "R Top2": float(R_precision[1]), 
                "R Top3": float(R_precision[2]), 
                "match_score_real": float(matching_score_real), 
                "matching_score_pred": float(matching_score_pred)}  # 将准确率添加到指标字典中
    return metrics

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist