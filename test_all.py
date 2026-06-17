import numpy as np
import os
import argparse
import torch
from torch.utils.data import DataLoader

from model.network.hf_SA import HierarchyFlow
from model.utils.dataset_real import get_dataset as get_dataset_real
from model.utils.dataset_synthetic import get_dataset as get_dataset_synthetic

import torch.nn as nn
import pandas as pd

import yaml
from easydict import EasyDict
import matplotlib.pyplot as plt
from generative.metrics import MultiScaleSSIMMetric, SSIMMetric
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize_per_image(img_tensor, min_vals, max_vals):
    img_denorm = torch.zeros_like(img_tensor)
    img_tensor_c = img_tensor.clone()
    img_denorm[:, 0] = img_tensor_c[:, 0] * (max_vals[0] - min_vals[0]) + min_vals[0]
    img_denorm[:, 1] = img_tensor_c[:, 1] * (max_vals[1] - min_vals[1]) + min_vals[1]
    img_denorm[:, 2] = img_tensor_c[:, 2] * (max_vals[2] - min_vals[2]) + min_vals[2]

    return img_denorm


class MRSignalModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, q_map, params_table_name):

        t1_t2 = q_map[:, 0:2, :, :]
        t1 = t1_t2[:, 0:1, :, :]
        t2 = t1_t2[:, 1:2, :, :]
        pd = q_map[:, 2:3, :, :]


        TE_t1 = torch.tensor(params_table_name['TE_T1'].values).unsqueeze(-1).unsqueeze(-1).to(device)
        TR_t1 = torch.tensor(params_table_name['TR_T1'].values).unsqueeze(-1).unsqueeze(-1).to(device)
        ESP_t1 = torch.tensor(params_table_name['ESP_T1'].values).unsqueeze(-1).unsqueeze(-1).to(device)

        TE_t2 = torch.tensor(params_table_name['TE_T2'].values).unsqueeze(-1).unsqueeze(-1).to(device)
        TR_t2 = torch.tensor(params_table_name['TR_T2'].values).unsqueeze(-1).unsqueeze(-1).to(device)
        ESP_t2 = torch.tensor(params_table_name['ESP_T2'].values).unsqueeze(-1).unsqueeze(-1).to(device)

        TE_pd = torch.tensor(params_table_name['TE_PD'].values).unsqueeze(-1).unsqueeze(-1).to(device)
        TR_pd = torch.tensor(params_table_name['TR_PD'].values).unsqueeze(-1).unsqueeze(-1).to(device)
        ESP_pd = torch.tensor(params_table_name['ESP_PD'].values).unsqueeze(-1).unsqueeze(-1).to(device)

        eps = 1e-8
        denom_t2 = t2 + eps
        denom_t1 = t1 + eps

        etl = 10.0


        t1_w = torch.exp(-TE_t1 / denom_t2) * (1 - torch.exp(-(TR_t1 - etl * ESP_t1) / denom_t1))
        t2_w = torch.exp(-TE_t2 / denom_t2) * (1 - torch.exp(-(TR_t2 - etl * ESP_t2) / denom_t1))
        pd_w = torch.exp(-TE_pd / denom_t2) * (1 - torch.exp(-(TR_pd - etl * ESP_pd) / denom_t1))


        t1_weigh = (pd * t1_w)

        t2_weigh = (pd * t2_w)

        pd_weigh = (pd * pd_w)

        return torch.cat((t1_weigh, t2_weigh, pd_weigh), 1)

def transform(image):
    max_val = np.max(image)
    min_val = np.min(image)
    scale = 2.0 / (max_val - min_val + 1e-8)
    offset = -1.0 - min_val * scale
    image_scaled = image * scale + offset

    return torch.from_numpy(image_scaled).unsqueeze(0).unsqueeze(0)


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['step']

def main():

    parser = argparse.ArgumentParser(
        description='HierarchyFlow Evaluation'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )

    parser.add_argument(
        '--load_path',
        type=str,
        required=True,
        help='Checkpoint path'
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = EasyDict(cfg)

    global pd
    df_params = pd.read_csv(cfg.df_params_path)

    model = HierarchyFlow(cfg.network.pad_size, cfg.network.in_channel, cfg.network.out_channels,
                          cfg.network.weight_type)

    model = model.to(device='cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model, optimizer, resumed_step = load_checkpoint(
        args.load_path,
        model,
        optimizer
    )

    t1_list_ms_ssim = []
    t2_list_ms_ssim = []
    pd_list_ms_ssim = []

    t1_list_psnr = []
    t2_list_psnr = []
    pd_list_psnr = []

    mr_model = MRSignalModel()

    ms_ssim_func = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)

    if cfg.dataset.synthetic:
        test_dataset = get_dataset_synthetic(cfg.dataset.test)
    else:
        test_dataset = get_dataset_real(cfg.dataset.test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False)
    model.eval()

    mr_model = MRSignalModel()

    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):
            content_images = batch[0].to(device='cuda')
            style_images = batch[1].to(device='cuda')

            name = batch[2][0]
            name_df = df_params[df_params['Name'] == name]

            arr_param = np.array([[name_df['TE_T1'].values[0], name_df['TR_T1'].values[0], name_df['ESP_T1'].values[0]],
                                  [name_df['TE_T2'].values[0], name_df['TR_T2'].values[0], name_df['ESP_T2'].values[0]],
                                  [name_df['TE_PD'].values[0], name_df['TR_PD'].values[0],
                                   name_df['ESP_PD'].values[0]]])

            tens_param = torch.tensor(arr_param).float().unsqueeze(0).to(device)

            outputs = model(content_images, style_images, tens_param)
            outputs = torch.clamp(outputs, 0, 1)

            denorm_outputs = denormalize_per_image(outputs, [0.0, 0.0, 0.0], [3.5, 0.45, 1.0])

            arr_maps = denorm_outputs[0].detach().cpu().numpy().transpose(1, 2, 0)

            reconstruct_images = mr_model(denorm_outputs, name_df)

            arr_imgs = reconstruct_images[0].detach().cpu().numpy().transpose(1, 2, 0)

            res_weighted = reconstruct_images  # * 4095
            denorm_images = content_images  # * 4095

            gt_t1 = denorm_images[0, 0, :, :].cpu().numpy()
            gt_t2 = denorm_images[0, 1, :, :].cpu().numpy()
            gt_pd = denorm_images[0, 2, :, :].cpu().numpy()

            gt_t1_norm = transform(gt_t1)
            gt_t2_norm = transform(gt_t2)
            gt_pd_norm = transform(gt_pd)

            reconstruction_t1_norm = transform(res_weighted[0, 0, :, :].cpu().numpy())
            reconstruction_t2_norm = transform(res_weighted[0, 1, :, :].cpu().numpy())
            reconstruction_pd_norm = transform(res_weighted[0, 2, :, :].cpu().numpy())

            t1_list_ms_ssim.append(ms_ssim_func(gt_t1_norm, reconstruction_t1_norm))
            t2_list_ms_ssim.append(ms_ssim_func(gt_t2_norm, reconstruction_t2_norm))
            pd_list_ms_ssim.append(ms_ssim_func(gt_pd_norm, reconstruction_pd_norm))

            t1_list_psnr.append(
                sk_psnr(reconstruction_t1_norm.numpy()[0, 0, :, :], gt_t1_norm.numpy()[0, 0, :, :], data_range=2.0))
            t2_list_psnr.append(
                sk_psnr(reconstruction_t2_norm.numpy()[0, 0, :, :], gt_t2_norm.numpy()[0, 0, :, :], data_range=2.0))
            pd_list_psnr.append(
                sk_psnr(reconstruction_pd_norm.numpy()[0, 0, :, :], gt_pd_norm.numpy()[0, 0, :, :], data_range=2.0))

    print('MS-SSIM T1: ', np.round(np.mean(t1_list_ms_ssim), 4), np.round(np.std(t1_list_ms_ssim), 3))

    print('MS-SSIM T2: ', np.round(np.mean(t2_list_ms_ssim), 4), np.round(np.std(t2_list_ms_ssim), 3))

    print('MS-SSIM PD: ', np.round(np.mean(pd_list_ms_ssim), 4), np.round(np.std(pd_list_ms_ssim), 3))

    print('PSNR T1: ', np.round(np.mean(t1_list_psnr), 4), np.round(np.std(t1_list_psnr), 3))

    print('PSNR T2: ', np.round(np.mean(t2_list_psnr), 4), np.round(np.std(t2_list_psnr), 3))

    print('PSNR PD: ', np.round(np.mean(pd_list_psnr), 4), np.round(np.std(pd_list_psnr), 3))


if __name__ == "__main__":
    main()
