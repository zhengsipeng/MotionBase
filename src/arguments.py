from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import transformers

@dataclass
class Text2MotionArguments:

    # train target
    train_target: str = field(default="t2m", metadata={"help": "t2m/m2t/t2m+m2t"})

    ## Dataloader
    dataname: str = field(default="t2m", metadata={"help": "Dataset name"})
    fps: List[int] = field(default_factory=lambda: [20], metadata={"help": "Frames per second"})
    seq_len: int = field(default=64, metadata={"help": "Training motion length"})
    train_split_file: str = field(default="train.txt", metadata={"help": "Split name"})
    train_meta_dir: str = field(default="", metadata={"help": "Meta dir"})
    val_meta_dir: str = field(default="checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta", metadata={"help": "Val meta dir"})
    val_split_file: str = field(default="val/test.txt", metadata={"help": "Val split name"})
    min_motion_len: int = field(default=24, metadata={"help": "Min motion len"})
    max_motion_len: int = field(default=196, metadata={"help": "Max motion len"})
    text_mot_match_path: str = field(default="", metadata={"help": "text_mot_match_path"})

    ## VQVAE Arch
    code_dim: int = field(default=512, metadata={"help": "Embedding dimension"})
    nb_code: int = field(default=512, metadata={"help": "Nb of embedding"})
    mu: float = field(default=0.99, metadata={"help": "Exponential moving average to update the codebook"})
    down_t: int = field(default=3, metadata={"help": "Downsampling rate"})
    stride_t: int = field(default=2, metadata={"help": "Stride size"})
    width: int = field(default=512, metadata={"help": "Width of the network"})
    depth: int = field(default=3, metadata={"help": "Depth of the network"})
    dilation_growth_rate: int = field(default=3, metadata={"help": "Dilation growth rate"})
    output_emb_width: int = field(default=512, metadata={"help": "Output embedding width"})
    vq_act: str = field(default="relu", metadata={"help": "Activation function", "choices": ["relu", "silu", "gelu"]})

    # ## GPT Arch
    # block_size: int = field(default=25, metadata={"help": "Seq len"})
    # embed_dim_gpt: int = field(default=512, metadata={"help": "Embedding dimension"})
    # clip_dim: int = field(default=512, metadata={"help": "Latent dimension in the clip feature"})
    # num_layers: int = field(default=2, metadata={"help": "Nb of transformer layers"})
    # n_head_gpt: int = field(default=8, metadata={"help": "Nb of heads"})
    # ff_rate: int = field(default=4, metadata={"help": "Feedforward size"})
    # drop_out_rate: float = field(default=0.1, metadata={"help": "Dropout ratio in the pos encoding"})

    ## Quantizer
    quantizer: str = field(default="ema_reset", metadata={"help": "Eps for optimal transport", "choices": ["ema", "orig", "ema_reset", "reset"]})
    quantbeta: float = field(default=1.0, metadata={"help": "Quantization beta"})

    ## Resume
    resume_pth: Optional[str] = field(default=None, metadata={"help": "Resume vq pth"})
    # resume_trans: Optional[str] = field(default=None, metadata={"help": "Resume gpt pth"})

    ## Output Directory
    # out_dir: str = field(default="output_GPT_Final/", metadata={"help": "Output directory"})
    # exp_name: str = field(default="exp_debug", metadata={"help": "Name of the experiment, will create a file inside out-dir"})
    vq_name: str = field(default="exp_debug", metadata={"help": "Name of the generated dataset .npy, will create a file inside out-dir"})

    # ## Other
    # print_iter: int = field(default=200, metadata={"help": "Print frequency"})
    # eval_iter: int = field(default=5000, metadata={"help": "Evaluation frequency"})
    # seed: int = field(default=123, metadata={"help": "Seed for initializing training."})
    # if_maxtest: bool = field(default=False, metadata={"help": "Test in max"})
    # pkeep: float = field(default=1.0, metadata={"help": "Keep rate for gpt training"})
    # num_gpu: int = field(default=1, metadata={"help": "Gpu num"})


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/share/pretrain/llm/Meta-Llama-3.1-8B")

@dataclass
class DataArguments:    
    
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    batch_method: str = field(default="naive")

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # ['gate_proj', 'o_proj', 'k_proj', 'q_proj', 'up_proj', 'down_proj', 'v_proj']
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['o_proj', 'k_proj', 'q_proj', 'v_proj']
    )
    # lora_target_modules = None
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False