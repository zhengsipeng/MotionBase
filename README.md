## MotionBase for ICLR2025

**A Text-to-Motion Generation Model based on GPT and VQVAE**
this is the released code for ICLR2025-submission, including testing, visuliaze codes and the pre-trained model (gpt-2-700m).
Our training code, dataset and more pretrained models will be released after ICLR final decision.

### Checkpoints
Please download ckpts from [https://www.dropbox.com/scl/fo/mx3u3mrvl72lkp6orc7o6/AJDXOC1RBWbCwXGZKeivgUk?rlkey=weq4o5enb7p3eyqnooe526orj&st=tab84oo7&dl=0](ckpt_url)

### Project Structure

```
├── ckpt                     # Stores model parameters
│   ├── MotionX                 # Model trained on MotionX dataset
│   │   ├── GPT             # GPT model parameters
│   │   ├── VQVAE           # VQVAE model parameters
│   │   │   └── net_last.pth
│   │   └── meta            # Dataset information
├── options                  # Parameter configuration
├── visualization            # Pose visualization code
└── utils                    # Utility functions

├── test.py                  # Test script
└── test.sh                  # Test script launcher
```

or you can also build the conda environment from environment.yaml

### Environment Setup

```bash
conda create -n motiongpt python==3.10.14
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tensorboard
pip install scipy
pip install matplotlib==3.4.3
pip install imageio
```

### Testing

1. **Modify the test script `test.sh`:**

   ```bash
   # export EXP_NAME="MotionX"    # Choose to use the MotionX model (more diverse actions)
   
   export VQ_MODEL="./ckpt/$EXP_NAME/VQVAE/net_last.pth"
   export TRANS_MODEL="./ckpt/$EXP_NAME/GPT"
   export META_DIR="./ckpt/$EXP_NAME/meta"
   
   python test.py \
   --exp-name $EXP_NAME \
   --resume-pth $VQ_MODEL \
   --resume-trans $TRANS_MODEL \
   --meta-dir $META_DIR \
   --instructions "Someone lifts their legs to perform a hip opener." "A person lifts a doorknob." "a man walks forwards at high speed, while swinging his arms." 
   ```

2. **Run the test script:**

   ```bash
   bash test.sh
   ```
