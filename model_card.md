
# 1
```bash
# UADM(25)-G
CLASS_COND=False
USE_DDIM=True
STEPS='ddim25'
CLASSIFIER_SCALE=10.0
MODEL_PATH='./pretrain_model/256x256_diffusion_uncond.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier.pt'
LOG_DIR=./log/imagenet256x256/UADM\(25\)-G/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=False
```

# 2
```bash
# UADM(25)-G+EDS
CLASS_COND=False
USE_DDIM=True
STEPS='ddim25'
USE_ENTROPY_SCALE=True
CLASSIFIER_SCALE=6.0
MODEL_PATH='./pretrain_model/256x256_diffusion_uncond.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier.pt'
LOG_DIR=./log/imagenet256x256/UADM\(25\)-G+EDS/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=True
```

# 3
```bash
# UADM(25)-G+EDS+ECT
CLASS_COND=False
USE_DDIM=True
STEPS='ddim25'
USE_ENTROPY_SCALE=True
CLASSIFIER_SCALE=6.0
MODEL_PATH='./pretrain_model/256x256_diffusion_uncond.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier+0.1ECT.pt'
LOG_DIR=./log/imagenet256x256/UADM\(25\)-G+EDS+0.1ECT/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=True
```

# 4
```bash
# UADM-G
CLASS_COND=False
USE_DDIM=False
STEPS='250'
USE_ENTROPY_SCALE=False
CLASSIFIER_SCALE=10.0
MODEL_PATH='./pretrain_model/256x256_diffusion_uncond.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier.pt'
LOG_DIR=./log/imagenet256x256/UADM-G/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
```

# 5
```bash
# UADM-G+EDS
CLASS_COND=False
USE_DDIM=False
STEPS='250'
USE_ENTROPY_SCALE=True
CLASSIFIER_SCALE=6.0
MODEL_PATH='./pretrain_model/256x256_diffusion_uncond.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier.pt'
LOG_DIR=./log/imagenet256x256/UADM-G+EDS/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=True
```

# 6
```bash
# UADM-G+EDS+ECT
CLASS_COND=False
USE_DDIM=False
STEPS='250'
USE_ENTROPY_SCALE=True
CLASSIFIER_SCALE=4.0
MODEL_PATH='./pretrain_model/256x256_diffusion_uncond.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier+0.1ECT.pt'
LOG_DIR=./log/imagenet256x256/UADM-G+EDS+0.1ECT/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=True

# 7
```bash
# CADM(25)-G
CLASS_COND=True
USE_DDIM=True
STEPS='ddim25'
CLASSIFIER_SCALE=2.5
MODEL_PATH='./pretrain_model/256x256_diffusion.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier.pt'
LOG_DIR=./log/imagenet256x256/CADM\(25\)-G/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=False
```

# 8
```bash
# CADM(25)-G+EDS
CLASS_COND=True
USE_DDIM=True
STEPS='ddim25'
USE_ENTROPY_SCALE=True
CLASSIFIER_SCALE=1.5
MODEL_PATH='./pretrain_model/256x256_diffusion.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier.pt'
LOG_DIR=./log/imagenet256x256/CADM\(25\)-G+EDS/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=True
```

# 9
```bash
# CADM(25)-G+EDS+ECT
CLASS_COND=True
USE_DDIM=True
STEPS='ddim25'
USE_ENTROPY_SCALE=True
CLASSIFIER_SCALE=2.0
MODEL_PATH='./pretrain_model/256x256_diffusion.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier+0.1ECT.pt'
LOG_DIR=./log/imagenet256x256/CADM\(25\)-G+EDS+0.1ECT/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=True
```


# 10
```bash
# CADM-G
CLASS_COND=True
USE_DDIM=False
STEPS='250'
USE_ENTROPY_SCALE=False
CLASSIFIER_SCALE=1.0
MODEL_PATH='./pretrain_model/256x256_diffusion.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier.pt'
LOG_DIR=./log/imagenet256x256/CADM-G/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
```


# 11
```bash
# CADM-G+EDS
CLASS_COND=True
USE_DDIM=False
STEPS='250'
USE_ENTROPY_SCALE=True
CLASSIFIER_SCALE=0.75
MODEL_PATH='./pretrain_model/256x256_diffusion.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier.pt'
LOG_DIR=./log/imagenet256x256/CADM-G+EDS/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=True
```

# 12
```bash
# CADM-G+EDS+ECT
CLASS_COND=True
USE_DDIM=False
STEPS='250'
USE_ENTROPY_SCALE=True
CLASSIFIER_SCALE=1.0
MODEL_PATH='./pretrain_model/256x256_diffusion.pt' 
CLASSIFIER_PATH='pretrain_model/256x256_classifier+0.1ECT.pt'
LOG_DIR=./log/imagenet256x256/CADM-G+EDS+0.1ECT/scale\=${CLASSIFIER_SCALE}_steps\=${STEPS}
USE_ENTROPY_SCALE=True
```

