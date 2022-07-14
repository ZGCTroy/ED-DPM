"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import random

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import utils

from entropy_driven_guided_diffusion import dist_util, logger
from entropy_driven_guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(local_rank=args.local_rank)

    if args.fix_seed:
        seed = 23333 + dist.get_rank()
        np.random.seed(seed)
        th.manual_seed(seed)  # CPU随机种子确定
        th.cuda.manual_seed(seed)  # GPU随机种子确定
        th.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

        th.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
        th.backends.cudnn.deterministic = True  # 确定为默认卷积算法

        random.seed(seed)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

    logger.configure(dir=args.log_dir)
    logger.log(args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.model_path:
        logger.log("loading model from {}".format(args.model_path))
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu"),
            strict=True
        )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))

    if args.classifier_path:
        logger.log("loading classifier from {}".format(args.classifier_path))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu"),
            strict=True
        )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

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

        with th.no_grad():
            if args.use_entropy_scale:
                probs = F.softmax(logits, dim=-1)  # (B, C)
                entropy = (-log_probs * probs).sum(dim=-1)  # (B,)
                entropy_scale = 1.0 / (entropy / np.log(NUM_CLASSES))  # (B,)
                entropy_scale = entropy_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                guidance['scale'] = guidance['scale'] * entropy_scale

            if args.detail:
                probs = F.softmax(logits, dim=-1)
                original_grad_norm = th.norm(guidance["gradient"] * guidance["scale"], p=2, dim=(1, 2, 3), dtype=th.float32).detach()
                selected_probability = probs[range(len(logits)), y.view(-1)]
                entropy = (-log_probs * probs).sum(dim=-1) / np.log(1000)  # (B, )
                entropy_scale = 1.0 / entropy  # (B, )
                model_variance = th.sqrt(kwargs['variance']).view(args.batch_size, -1).mean(-1)

                logger.log(
                    '\n',
                    't = ', t[0].detach(), '\n',
                    '\t\t mean std median', '\n',
                    '\t\t grad_norm =', original_grad_norm.mean(-1).detach(), original_grad_norm.std(-1).detach(), original_grad_norm.median(-1).values, '\n',
                    '\t\t probability = ', selected_probability.mean(-1).detach(), selected_probability.std(-1).detach(), selected_probability.median(-1).values, '\n',
                    '\t\t entropy = ', entropy.mean(-1).detach(), entropy.std(-1).detach(), entropy.median(-1).values, '\n',
                    '\t\t entropy_scale = ', entropy_scale.mean(-1).detach(), entropy_scale.std(-1).detach(), entropy_scale.median(-1).values, '\n',
                    '\t\t model_variance = ', model_variance.mean(-1).detach(), model_variance.std(-1).detach(), model_variance.median(-1).values, '\n',
                )

        return guidance

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    current_num_samples = 0
    arr, label_arr = np.ones([0, 512, 512, 3], dtype=np.uint8), np.ones([0], dtype=np.uint8)
    npz_id = 0
    while current_num_samples < args.num_samples:
        model_kwargs = {}
        if args.specified_class is not None:
            classes = th.randint(
                low=int(args.specified_class), high=int(args.specified_class) + 1, size=(args.batch_size,), device=dist_util.dev()
            )
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
        model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )  # (B, 3, H, W)

        if args.save_imgs_for_visualization and dist.get_rank() == 0 and current_num_samples < 100:
            save_img_dir = os.path.join(
                logger.get_dir(),
                "imgs/{}".format(args.sample_suffix)
            )
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
            utils.save_image(
                sample.clamp(-1, 1),
                os.path.join(save_img_dir, "{}_{}.png".format(args.sample_suffix, current_num_samples)),
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # (B, H, W, 3)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)

        images_arr_for_all_gpus = np.concatenate([sample.cpu().numpy() for sample in gathered_samples], axis=0)
        labels_arr_for_all_gpus = np.concatenate([labels.cpu().numpy() for labels in gathered_labels], axis=0)

        current_num_samples += labels_arr_for_all_gpus.shape[0]

        logger.log(f"created {current_num_samples} / {args.num_samples} samples")

        if dist.get_rank() == 0:
            arr = np.append(arr, images_arr_for_all_gpus, axis=0)
            label_arr = np.append(label_arr, labels_arr_for_all_gpus, axis=0)

            if (label_arr.shape[0] >= args.intermediate_num_samples) or (current_num_samples >= args.num_samples):
                total_npz_shape = [int(_) for _ in arr.shape]
                total_npz_shape[0] = args.num_samples
                shape_str = "x".join([str(x) for x in total_npz_shape])
                if args.sample_suffix:
                    out_path = os.path.join(logger.get_dir(), f"npz_results/samples_{shape_str}_{args.sample_suffix}/{npz_id}.npz")
                else:
                    out_path = os.path.join(logger.get_dir(), f"npz_results/samples_{shape_str}/{npz_id}.npz")
                logger.log(f"saving to {out_path}")
                dir_name = os.path.dirname(os.path.dirname(out_path))
                os.makedirs(dir_name, exist_ok=True)
                dir_name = os.path.dirname(out_path)
                os.makedirs(dir_name, exist_ok=True)
                np.savez(out_path, arr[:args.intermediate_num_samples], label_arr[:args.intermediate_num_samples])
                arr, label_arr = arr[args.intermediate_num_samples:], label_arr[args.intermediate_num_samples:]
                npz_id += 1

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,

        log_dir="",
        sample_suffix="",
        fix_seed=False,
        save_imgs_for_visualization=True,
        specified_class=None,
        detail=False,

        intermediate_num_samples=128
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
