#! /bin/bash
GPU_NUM=0
SEED_NUM=0
DATE='1111'
#APPROACH = ['finetuning', 'joint', 'bic', 'lucir']

# CON_ALPHA=0.1
# EXP_NAME="con_alpha_${CON_ALPHA}"

# CUDA_VISIBLE_DEVICES=2 python main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'bic' --datasets 'imagenet100' --network 'resnet18' --batch-size 64 --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random  --num-workers 8  --lr-scheduler 'multisteplr'


# CUDA_VISIBLE_DEVICES=2 python main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'bic' --datasets 'imagenet100' --network 'resnet18' --batch-size 64 --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --datasets-unsup 'MS_COCO_2017_unsup'


CON_ALPHA=0.1
CON_STRATEGY='SimCLR'
PROTOTYPES=0


BATCH=64
SPLIT=False
NUM_SPLITS=1
RATIO=3
FIX=False
SEPERATE=False
CN=8
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_CN_${CN}_1215_gpu0"
NOTE="load BN apply Split-BN, update only running mean"

CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'bic' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis --cn $CN

BATCH=64
SPLIT=False
NUM_SPLITS=1
RATIO=3
FIX=False
SEPERATE=False
CN=16
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_CN_${CN}_1215_gpu0"
NOTE="load BN apply Split-BN, update only running mean"

CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'bic' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis --cn $CN

BATCH=64
SPLIT=False
NUM_SPLITS=1
RATIO=3
FIX=False
SEPERATE=False
CN=32
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_CN_${CN}_1215_gpu0"
NOTE="load BN apply Split-BN, update only running mean"

CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'bic' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis --cn $CN

BATCH=64
SPLIT=False
NUM_SPLITS=1
RATIO=3
FIX=False
SEPERATE=False
CN=64
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_CN_${CN}_1215_gpu0"
NOTE="load BN apply Split-BN, update only running mean"

CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'bic' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis --cn $CN

BATCH=10
SPLIT=False
NUM_SPLITS=1
RATIO=3
FIX=False
SEPERATE=False
CN=0
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_CN_${CN}_1206_gpu0"
NOTE="load BN apply Split-BN, update only running mean"

#CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'finetuning' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis

BATCH=10
SPLIT=False
NUM_SPLITS=1
RATIO=3
FIX=False
SEPERATE=False
CN=4
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_CN_${CN}_1206_gpu0"
NOTE="load BN apply Split-BN, update only running mean"

#CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'finetuning' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis --cn $CN



BATCH=64
SPLIT=False
NUM_SPLITS=1
RATIO=0
FIX=False
SEPERATE=False
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_log_bn_change_only_mu_1119_gpu0"
NOTE="load BN apply FT-BN, update running mean"
#CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'finetuning' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100  --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis --change-mu True --model-freeze True

BIAS_ANALYSIS='plain'
FREEZE=False
CHANGE_MU=True
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_${BIAS_ANALYSIS}_freeze_${FREEZE}_change_mu_${CHANGE_MU}_${DATE}_gpu0"
NOTE="load BN, insert SplitBN(BN-)"

#CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'finetuning' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100 --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis --bias-analysis $BIAS_ANALYSIS --change-mu $CHANGE_MU

BIAS_ANALYSIS='plain'
FREEZE=False
CHANGE_MU=False
NOISE_SCALE=0.01
EXP_NAME="split_${SPLIT}_num_splits_${NUM_SPLITS}_batch_${BATCH}_Fix-batch_${FIX}_Ratio_${RATIO}_${BIAS_ANALYSIS}_freeze_${FREEZE}_change_mu_${CHANGE_MU}_${DATE}_Noise_${NOISE_SCALE}_split_gpu0"
NOTE="load BN, insert SplitBN(BN-)"

#CUDA_VISIBLE_DEVICES=0 python3 main_incremental.py --gpu $GPU_NUM --seed $SEED_NUM --approach 'finetuning' --datasets 'imagenet100' --network 'resnet18' --batch-size $BATCH --num-tasks 10 --nepochs 100 --num-exemplars 2000 --lr-factor 10 --weight-decay 0.0001 --exemplar-selection random --num-workers 8 --con-temp 0.1 --con-alpha $CON_ALPHA --exp-name $EXP_NAME --con-strategy $CON_STRATEGY --nmb_prototypes $PROTOTYPES --bn-splits $NUM_SPLITS --log-bn True --last-layer-analysis --bias-analysis $BIAS_ANALYSIS --noise $NOISE_SCALE
