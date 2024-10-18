import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
from functools import partial
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import models.vqvae as vqvae
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt

from src.dataset import Text2MotionDataset, Text2MotionDatasetEval, ValDATALoader
from src.trainer import TrainerText2motion
from src.evaluater import evaluation_transformer
from src.arguments import Text2MotionArguments, ModelArguments, DataArguments, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PAD_ID = 128004

@dataclass
class DataCollatorForMotionDataset(object):
    tokenizer: AutoTokenizer

    def __call__(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:

        eos_token_id = self.tokenizer.eos_token_id
        
        inputs = self.tokenizer(batch_texts, 
                    padding=True,
                    truncation=True,
                    max_length=256,
                    add_special_tokens=True,
                    return_tensors="pt")

        input_ids = inputs['input_ids']
        input_ids = torch.cat([input_ids, torch.tensor([[eos_token_id] * input_ids.shape[0]]).T.to(input_ids.device)], dim=1)

        attention_mask = inputs['attention_mask']
        attention_mask = torch.cat([attention_mask, torch.tensor([[1] * input_ids.shape[0]]).T.to(input_ids.device)], dim=1)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

def make_supervised_data_module(data_args, t2m_args, tokenizer=None) -> Dict:
    print("loading data...")

    train_dataset = Text2MotionDataset(t2m_args.dataname, 
                                                    codebook_size=t2m_args.nb_code, 
                                                    tokenizer_name=t2m_args.vq_name, 
                                                    unit_length=2**t2m_args.down_t, 
                                                    split_name=t2m_args.train_split_file, 
                                                    meta_dir=t2m_args.train_meta_dir, 
                                                    tokenizer=tokenizer,
                                                    train_target=t2m_args.train_target)
    data_collator = DataCollatorForMotionDataset(tokenizer=tokenizer)

    print("finish loading data")
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, Text2MotionArguments))
    model_args, data_args, training_args, t2m_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                        torch_dtype=torch.bfloat16, 
                                        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                trust_remote_code=True, use_fast=False)

    m_codebook_size = t2m_args.nb_code
    print('m_codebook_size', m_codebook_size)

    tokenizer.add_tokens([f'<motion_id_{i}>' for i in range(m_codebook_size)] + [f'<Motion Token>', f'</Motion Token>']) # 添加新token_id

    new_token_type = "insert"
    if new_token_type == "insert":
        model.resize_token_embeddings(len(tokenizer))
    elif new_token_type == "mlp":
        shared = NewTokenEmb(model.shared,
                                m_codebook_size)
        # lm_head = NewTokenEmb(model.lm_head,
        #   self.m_codebook_size + 3)
        model.resize_token_embeddings(len(tokenizer))
        model.shared = shared
        # self.language_model.lm_head = lm_head

    tokenizer.padding_side = 'left'
    print('padding_side', tokenizer.padding_side)
    print('truncation_side:', tokenizer.truncation_side)
    global PAD_ID
    PAD_ID = tokenizer.eos_token_id
    tokenizer.pad_token_id = PAD_ID

    data_module = make_supervised_data_module(data_args=data_args, t2m_args=t2m_args, tokenizer=tokenizer)

    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    wrapper_opt.text_mot_match_path = t2m_args.text_mot_match_path
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    net = vqvae.HumanVQVAE(t2m_args, ## use args to define different parameters in different quantizers
                       t2m_args.nb_code,
                       t2m_args.code_dim,
                       t2m_args.output_emb_width,
                       t2m_args.down_t,
                       t2m_args.stride_t,
                       t2m_args.width,
                       t2m_args.depth,
                       t2m_args.dilation_growth_rate)

    print ('loading checkpoint from {}'.format(t2m_args.resume_pth))
    ckpt = torch.load(t2m_args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    # net.cuda()
                    
    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = ValDATALoader(t2m_args.dataname, False, 32, w_vectorizer, meta_dir=t2m_args.val_meta_dir, split_name=t2m_args.val_split_file)

    evaluation_transformer_partial = partial(evaluation_transformer, val_loader=val_loader, net=net, eval_wrapper=eval_wrapper)

    trainer = TrainerText2motion(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module,
        custom_eval_func=evaluation_transformer_partial
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()

if __name__ == "__main__":
    train()
