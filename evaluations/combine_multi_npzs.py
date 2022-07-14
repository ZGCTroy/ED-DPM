import numpy as np

arr, label_arr = np.ones([0, 512, 512, 3], dtype=np.uint8), np.ones([0], dtype=np.uint8)

for id in [0, 1, 2, 3, 4, 5, 6]:
    print(id)
    npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/ddpm_origin/log/imagenet1000_classifier512x512_channel128_0.1ECT/conditional_ddim25/sample50000/npz_results/samples_50000x512x512x3_model499999_CADM-G(25)_0.1ECT_EDS_scale8.0/{}.npz'.format(
        id
    )
    # npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/ddpm_origin/log/imagenet1000_classifier512x512_channel128_0.1ECT/conditional_250/sample50000/npz_results/samples_50000x512x512x3_model499999_CADM-G+0.1ECT+EDS_scale=4.0/{}.npz'.format(
    #     id
    # )

    loaded_arr = np.load(npz_path)

    arr = np.append(arr, loaded_arr['arr_0'], axis=0)

    del loaded_arr

save_npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/ddpm_origin/log/imagenet1000_classifier512x512_channel128_0.1ECT/conditional_ddim25/sample50000/npz_results/samples_50000x512x512x3_model499999_CADM-G(25)+0.1ECT+EDS_scale=8.0.npz'
# save_npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/ddpm_origin/log/imagenet1000_classifier512x512_channel128_0.1ECT/conditional_250/sample50000/npz_results/samples_50000x512x512x3_model499999_CADM-G+0.1ECT+EDS_scale=4.0/samples_50000x512x512x3_model499999_CADM-G+0.1ECT+EDS_scale=4.0.npz'
np.savez(save_npz_path, arr_0=arr)
