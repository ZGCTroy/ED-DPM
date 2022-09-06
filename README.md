# ED-DPM (**E**ntropy-**D**riven - Diffusion Probabilistic Model)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/entropy-driven-sampling-and-training-scheme/conditional-image-generation-on-imagenet)](https://paperswithcode.com/sota/conditional-image-generation-on-imagenet?p=entropy-driven-sampling-and-training-scheme)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/entropy-driven-sampling-and-training-scheme/conditional-image-generation-on-imagenet-2)](https://paperswithcode.com/sota/conditional-image-generation-on-imagenet-2?p=entropy-driven-sampling-and-training-scheme)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/entropy-driven-sampling-and-training-scheme/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=entropy-driven-sampling-and-training-scheme)

Accepted by ECCV 2022


This is the official codebase for [Entropy-driven Sampling and Training Scheme for Conditional Diffusion Generation](https://arxiv.org/abs/2206.11474).
(not Camera Ready Version, Camera Ready Version is coming soon)

This repository is heavily based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), 
with modifications listed below:
1. add **EDS** (**E**ntropy-**D**riven conditional **S**ampling) in classifier-guidance sample process without retraining
2. add **ECT** (**E**ntropy **C**onstraint **T**raining) in classifier training process, the model trained with **ECT** will 
have more realistic generation results when combined with **EDS** in sampling process.
3. support Distributed Training of Pytorch

## Update
* 2022.09.06 !!! [Camera Ready version of paper](https://drive.google.com/file/d/1NjbH-fwAPK64o3pD3P7e2_9Dp9aKySkZ/view?usp=sharing) is now available !!!
* 2022.08.03 The paper in arxiv is not Camera Ready Version, Camera Ready Version is coming soon !!!
* 2022.08.03 fix the bug of mixed precision training to follow [openai/guided-diffusion commit](https://github.com/openai/guided-diffusion/commit/22e0df8183507e13a7813f8d38d51b072ca1e67c)
* 2022.08.02 upload pretrained model of [256x256_classifier+0.1ECT.pt](https://drive.google.com/drive/folders/1xldlyBYS7PSrC4tZxSPne9dpIW06QC51?usp=sharing)
* 2022.07.14 upload the code

## Based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), only a few lines of code to apply EDS
set args.use_entropy_scale = True to apply EDS 
```bash
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            cond_grad = th.autograd.grad(selected.sum(), x_in)[0]

        guidance = {
            'gradient': cond_grad,
            'scale': args.classifier_scale
        }

        # a few lines of code to apply EDS
        if args.use_entropy_scale:
            with th.no_grad():
                probs = F.softmax(logits, dim=-1)  # (B, C)
                entropy = (-log_probs * probs).sum(dim=-1)  # (B,)
                entropy_scale = 1.0 / (entropy / np.log(NUM_CLASSES))  # (B,)
                entropy_scale = entropy_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                guidance['scale'] = guidance['scale'] * entropy_scale

        return guidance

```
# Download pre-trained models

We have released checkpoints for the main models in the paper. Here are the download links for each model checkpoint:


 * 256x256 diffusion  (CADM): [256x256_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)
 * 256x256 diffusion (UADM): [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)
 * 256x256 classifier (-G): [256x256_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt)
 * 256x256 classifier (-G+ECT): [256x256_classifier+0.1ECT.pt](https://drive.google.com/drive/folders/1xldlyBYS7PSrC4tZxSPne9dpIW06QC51?usp=sharing)


# Sampling from pre-trained models

To sample from these models, you can use the `classifier_sample.py` scripts.
Here, we provide flags for sampling from all of these models.
We assume that you have downloaded the relevant model checkpoints into a folder called `./pretrain_model/`.

## 1. set up environment
```bash
  conda create -n ED-DPM
  conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
  
```

## 2. input some parameters
```bash
WORKSPACE_DIR=/workspace/mnt/storage/guangcongzheng/zju_zgc/ED-DPM
cd $WORKSPACE_DIR

rm -rf ./entropy_driven_guided_diffusion.egg-info

python setup.py build develop

MODEL_FLAGS="
  --attention_resolutions 32,16,8
  --image_size 256 --learn_sigma True --num_channels 256
  --num_head_channels 64 --num_res_blocks 2 --resblock_updown True
  --use_fp16 True --use_scale_shift_norm True
  "

CLASSIFIER_FLAGS="
  --image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2
  --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True
  --classifier_use_scale_shift_norm True  --classifier_use_fp16 True
  "
  
# For 1 node, 8 GPUs:
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM + 10000))
NUM_GPUS=2
  
```

## 3. select the model hyper-parameters 
* ddim25
  * UADM
    * [UADM(25)-G](./model_card.md#1)
    * [UADM(25)-G+EDS](./model_card.md#2)
    * [UADM(25)-G+EDS+ECT](./model_card.md#3)
  * CADM
    * [CADM(25)-G](./model_card.md#7)
    * [CADM(25)-G+EDS](./model_card.md#8)
    * [CADM(25)-G+EDS+ECT](./model_card.md#9)

* ddpm250
  * UADM
    * [UADM-G](./model_card.md#4)
    * [UADM-G+EDS](./model_card.md#5)
    * [UADM-G+EDS+ECT](./model_card.md#6)
  * CADM 
    * [CADM-G](./model_card.md#10)
    * [CADM-G+EDS](./model_card.md#11)
    * [CADM-G+EDS+ECT](./model_card.md#12)


## 4. sample
```bash
NUM_SAMPLES=16 # set 50000 to reproduct the results reported in paper, howerver may take few days 
FIX_SEED=True # set False when sampling 50000 samples
BATCH_SIZE=1

# CUDA_VISIBLE=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
               --nnodes $WORLD_SIZE --node_rank $RANK --nproc_per_node ${NUM_GPUS} \
               --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
               scripts/classifier_sample.py \
               $MODEL_FLAGS $CLASSIFIER_FLAGS \
              --class_cond ${CLASS_COND} --model_path ${MODEL_PATH} \
              --classifier_path ${CLASSIFIER_PATH} --classifier_scale ${CLASSIFIER_SCALE} \
              --log_dir ${LOG_DIR} --num_samples ${NUM_SAMPLES} --batch_size ${BATCH_SIZE} \
              --timestep_respacing ${STEPS} --use_ddim ${USE_DDIM} \
              --use_entropy_scale ${USE_ENTROPY_SCALE} --fix_seed ${FIX_SEED}

```

## 5. evaluate FID, IS, sFID, Precision, Recall 
need to install tensorflow
```bash
# CUDA_VISIBLE=0,1,2,3,4,5,6,7 \
python evaluations/evaluator.py \
              --ref_batch pretrain_model/VIRTUAL_imagenet256_labeled.npz \
              --sample_batch ${LOG_DIR}/npz_results/samples_${NUM_SAMPLES}x256x256x3.npz \
              --save_result_path ${LOG_DIR}/metric_results/samples_${NUM_SAMPLES}x256x256x3.yaml
               
```


# Results

This table summarizes our ImageNet results for pure guided diffusion models:


## Reproduction Imagenet 256x256, steps=ddim25
| Dataset            | classifier scale | FID   | sFid | IS     | Precision | Recall |
|--------------------|------------------|-------|------|--------|-----------|--------|
| UADM(25)-G         | 10.0             | 14.22 | 8.62 | 83.38  | 0.7       | 0.46   |
| UADM(25)-G+EDS     | 6.0              | 10.09 | 6.87 | 133.89 | 0.73      | 0.46   |
| UADM(25)-G+EDS+ECT | 6.0              | 8.28  | 6.38 | 163.17 | 0.76      | 0.45   |
| CADM(25)-G         | 2.5              | 5.47  | 5.4  | 196    | 0.81      | 0.49   |
| CADM(25)-G+EDS     | 1.5              | 4.76  | 5.15 | 221.38 | 0.8       | 0.51   |
| CADM(25)-G+EDS+ECT | 2.0              | 4.68  | 5.13 | 235.25 | 0.82   | 0.48      |

## Reproduction Imagenet 256x256, steps=ddpm250
| Dataset        | classifier scale | FID  | sFid  |  IS    | Precision | Recall |
|----------------|------------------|------|-------|--------|-----------|--------|
| UADM-G         | 10.0             | 12   | 10.4      | 95.41   | 0.76 | 0.44 |
| UADM-G+EDS     | 6.0              | 7.98 | 8.54      | 179.36   | 0.82 | 0.4 |
| UADM-G+EDS+ECT | 4.0              | 6.78 | 6.56 | 168.78   | 0.81 | 0.45 |
| CADM-G         | 1.0              | 4.59 | 5.25 | 186.7   | 0.82 | 0.52 |
| CADM-G+EDS     | 0.75             | 3.96 | 5.06 | 219.13   | 0.83 | 0.52 |
| CADM-G+EDS+ECT | 1.0              | 4.09 | 5.07 | 221.57   |  0.83 | 0.50 |


# Cite
```
@article{zheng2022entropy,
  title={Entropy-driven Sampling and Training Scheme for Conditional Diffusion Generation},
  author={Zheng, Guangcong and Li, Shengming and Wang, Hui and Yao, Taiping and Chen, Yang and Ding, Shoudong and Li, Xi},
  journal={arXiv preprint arXiv:2206.11474},
  year={2022}
}
```

