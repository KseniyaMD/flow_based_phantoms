import shutil
import random
from easydict import EasyDict
import numpy as np

import os
import argparse
import yaml


import torch
from torch.utils.data import DataLoader

from model.network.hf_SA import HierarchyFlow
from model.utils.dataset_real import get_dataset as get_dataset_real
from model.utils.dataset_synthetic import get_dataset as get_dataset_synthetic
from tensorboardX import SummaryWriter

import logging
from model.utils.log_helper import init_log
import matplotlib.pyplot as plt

import torch.nn as nn
import pandas as pd

torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



init_log('pytorch hierarchy flow')
global_logger = logging.getLogger('pytorch hierarchy flow')

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


def save_checkpoint(state, filename):
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint_1 = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint_1['state_dict'])
    optimizer.load_state_dict(checkpoint_1['optimizer'])
    return model, optimizer, checkpoint_1['step']



def main():
    parser = argparse.ArgumentParser(description='PyTorch HierarchyFlow Training')
    parser.add_argument('--config', type=str, default='./configs/config_real_after_synth_form.yaml', help='config file')
    parser.add_argument('--eval_only', default=False, action='store_true', help='evaluation mode')
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=0, help='seed for initializing training')
    parser.add_argument('--load_path', type=str, help='path for ckpt')

    args = parser.parse_args()

    if args.seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = args.seed
    torch.manual_seed(seed)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(cfg)
    cfg.eval_mode = args.eval_only

    global pd
    df_params = pd.read_csv(cfg.df_params_path)


    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    if not os.path.exists(os.path.join(cfg.output, cfg.task_name)):
        os.makedirs(os.path.join(cfg.output, cfg.task_name))
    if not os.path.exists(os.path.join(cfg.output, cfg.task_name, 'img_save')):
        os.makedirs(os.path.join(cfg.output, cfg.task_name, 'img_save'))
    if not os.path.exists(os.path.join(cfg.output, cfg.task_name, 'w_img_save')):
        os.makedirs(os.path.join(cfg.output, cfg.task_name, 'w_img_save'))
    if not os.path.exists(os.path.join(cfg.output, cfg.task_name, 'model_save')):
        os.makedirs(os.path.join(cfg.output, cfg.task_name, 'model_save'))
    if not os.path.exists(os.path.join(cfg.output, cfg.task_name, 'eval_results')):
        os.makedirs(os.path.join(cfg.output, cfg.task_name, 'eval_results'))
        os.makedirs(os.path.join(cfg.output, cfg.task_name, 'eval_results', 'pred'))
        os.makedirs(os.path.join(cfg.output, cfg.task_name, 'eval_results', 'cat_img'))
    shutil.copy(os.path.join(args.config), os.path.join(cfg.output, cfg.task_name, 'cfg.yaml'))


    model = HierarchyFlow(cfg.network.pad_size, cfg.network.in_channel, cfg.network.out_channels,
                          cfg.network.weight_type)

    model = model.to(device='cuda')


    global_logger.info(cfg)
    global_logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)


    if cfg.resume:
        model, optimizer, resumed_step = load_checkpoint(cfg.load_path, model, optimizer)
        global_logger.info(
            "=> loaded checkpoint '{}' with current step {}".format(cfg.load_path, resumed_step))
    else:
        model = model
        optimizer = optimizer
        resumed_step = -1

    if cfg.lr_scheduler.type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.max_iter,
                                                                       cfg.lr_scheduler.eta_min)
    else:
        raise RuntimeError('lr_scheduler {} is not implemented'.format(cfg.lr_scheduler))



    logger = SummaryWriter(os.path.join(cfg.output, cfg.task_name, 'runs'))

    if cfg.dataset.synthetic:
        train_dataset = get_dataset_synthetic(cfg.dataset.train)
    else:
        train_dataset = get_dataset_real(cfg.dataset.train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True)

    mr_model = MRSignalModel()


    for ep in range(cfg.dataset.train.epochs):
        for batch_id, batch in enumerate(train_loader):

            b_id = batch_id + (len(train_loader) * ep)
            content_images = batch[0].float().to(device='cuda')
            style_images = batch[1].float().to(device='cuda')

            name = batch[2][0]

            name_df = df_params[df_params['Name'] == name]

            arr_param = np.array([[name_df['TE_T1'].values[0], name_df['TR_T1'].values[0], name_df['ESP_T1'].values[0]],
                            [name_df['TE_T2'].values[0], name_df['TR_T2'].values[0], name_df['ESP_T2'].values[0]],
                            [name_df['TE_PD'].values[0], name_df['TR_PD'].values[0], name_df['ESP_PD'].values[0]]])

            tens_param = torch.tensor(arr_param).float().unsqueeze(0).to(device)

            outputs = model(content_images, style_images, tens_param)
            outputs = torch.clamp(outputs, 0, 1)

            denorm_outputs = denormalize_per_image(outputs, [0.0, 0.0, 0.0], [3.5, 0.45, 1.0])
            reconstruct_images = mr_model(denorm_outputs, name_df)

            res_weighted = reconstruct_images * 4095.0
            denorm_images = content_images * 4095.0

            loss_MSE = nn.MSELoss()

            loss = loss_MSE(res_weighted.float(), denorm_images.float())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]

            logger.add_scalar("current_lr", current_lr, b_id + 1)
            logger.add_scalar("loss", loss.item(), b_id + 1)

            if batch_id % cfg.print_freq == 0:
                global_logger.info(
                    'batch: {}, loss: {}'.format(b_id, loss.item()))

                t1 = denorm_outputs[0, 0, :, :].detach().cpu().numpy()
                t2 = denorm_outputs[0, 1, :, :].detach().cpu().numpy()
                pd = denorm_outputs[0, 2, :, :].detach().cpu().numpy()

                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 15))  # 3 строки

                for ax, mat, title in zip(axes, [t1, t2, pd], ['t1', 't2', 'pd']):
                    im = ax.imshow(mat, cmap='viridis', interpolation='none')
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax)

                output_name = os.path.join(cfg.output, cfg.task_name, 'img_save', str(b_id) + '.png')

                plt.tight_layout()
                fig.savefig(output_name, dpi=300, bbox_inches='tight')
                plt.close(fig)

                train_reconstruction = res_weighted.detach().cpu().numpy()

                t1 = train_reconstruction[0, 0, :, :]
                t2 = train_reconstruction[0, 1, :, :]
                pd = train_reconstruction[0, 2, :, :]

                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 15))  # 3 строки

                for ax, mat, title in zip(axes, [t1, t2, pd], ['t1', 't2', 'pd']):
                    im = ax.imshow(mat, cmap='gray', interpolation='none')
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax)

                output_name = os.path.join(cfg.output, cfg.task_name, 'W_img_save', str(b_id) + '.png')

                plt.tight_layout()
                fig.savefig(output_name, dpi=300, bbox_inches='tight')
                plt.close(fig)

                t1 = denorm_images[0, 0, :, :].detach().cpu().numpy()
                t2 = denorm_images[0, 1, :, :].detach().cpu().numpy()
                pd = denorm_images[0, 2, :, :].detach().cpu().numpy()

                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 15))  # 3 строки

                for ax, mat, title in zip(axes, [t1, t2, pd], ['t1', 't2', 'pd']):
                    im = ax.imshow(mat, cmap='gray', interpolation='none')
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax)

                output_name = os.path.join(cfg.output, cfg.task_name, 'W_img_save', str(b_id) + 'target' + '.png')

                plt.tight_layout()
                fig.savefig(output_name, dpi=300, bbox_inches='tight')
                plt.close(fig)

            if batch_id % cfg.save_freq == 0:
                save_checkpoint({
                    'step': b_id,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(cfg.output, cfg.task_name, 'model_save', str(b_id) + '.ckpt'))


            torch.cuda.empty_cache()




if __name__ == "__main__":
    main()