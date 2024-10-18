
export EXP_NAME="MotionX"

export VQ_MODEL="./ckpt/$EXP_NAME/VQVAE/net_last.pth"
export TRANS_MODEL="./ckpt/$EXP_NAME/GPT"
export META_DIR="./ckpt/$EXP_NAME/meta"

python test.py \
--exp-name $EXP_NAME \
--resume-pth $VQ_MODEL \
--resume-trans $TRANS_MODEL \
--meta-dir $META_DIR \
--instructions "Someone lifts their legs to perform a hip opener." "A person lifts a doorknob." "a man walks forwards at high speed, while swinging his arms."
