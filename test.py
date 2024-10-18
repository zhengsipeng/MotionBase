import options.option_transformer as option_trans
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
import os

from utils.motion_process import recover_from_ric # 263 dim -> 22*3 xyz keypoints
import visualization.plot_3d_global as plot_3d  # vis
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re

import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

args = option_trans.get_args_parser()

args.dataname = 't2m'
args.down_t = 2
args.depth = 3
args.block_size = 51
args.nb_code = 1024

############# load model ##############
model = AutoModelForCausalLM.from_pretrained(args.resume_trans, 
                                    torch_dtype=torch.bfloat16, 
                                    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.resume_trans,
                                            trust_remote_code=True, use_fast=False)
tokenizer.padding_side = 'left'

model.eval()
model.cuda()

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()
############# load model down ##############

############# load mean/std ##############
mean = torch.from_numpy(np.load(os.path.join(args.meta_dir, 'mean.npy'))).cuda()
std = torch.from_numpy(np.load(os.path.join(args.meta_dir, 'std.npy'))).cuda()

if len(args.instructions) > 0:
    clip_texts = args.instructions
else:
    # change the text here
    clip_texts = ["The person performs a Knee Tap To Jackknife.",
                "A woman is simultaneously bathing pets and sitting.",
                "A person performs 32 Form Tai Chi Demonstration Master Form27 Step Back To Ride The Tiger.",
                "he is moving his left hand in round motion and then touched his head with the same hand.",
                "The person passes binoculars.",
                "A person is getting on his knees.",
                "Someone walks forward, then quickly turns around and walks back.",
                "transition",
                "step forward",
                "raise left hand over brow"]

eos_token_id = tokenizer.eos_token_id

generation_config = GenerationConfig(
    max_new_tokens=100,
    pad_token_id = tokenizer.eos_token_id,
)

for clip_text in tqdm(clip_texts):
    print(clip_text)
    inputs = tokenizer([clip_text], 
                            padding=True, 
                            truncation=True, 
                            add_special_tokens=True, 
                            return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs,
                            generation_config=generation_config)
    pred_str = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:])
    # print(pred_str)
    pred_numbers_str = re.findall(r'<motion_id_(\d+)>', pred_str.split("<Motion Token>")[-1].split("</Motion Token>")[0])
    if len(pred_numbers_str) == 0:
        index_motion = torch.ones(1,1).cuda().long()
    else:
        try:
            pred_numbers = [int(n) for n in pred_numbers_str]
            index_motion = torch.tensor([pred_numbers]).to(outputs)
        except:
            index_motion = torch.ones(1,1).cuda().long()
    pred_pose = net.forward_decoder(index_motion)
    print(pred_pose.shape)
    
    pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)
    xyz = pred_xyz.reshape(1, -1, 22, 3)

    xyz_name = clip_text.replace(" ", "_")
    os.makedirs(f'outputs/{args.exp_name}', exist_ok = True)
    np.save(f'outputs/{args.exp_name}/{xyz_name}.npy', xyz.detach().cpu().numpy())
    
    pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(), [f"{clip_text}"], [f'outputs/{args.exp_name}/{xyz_name}.gif'])