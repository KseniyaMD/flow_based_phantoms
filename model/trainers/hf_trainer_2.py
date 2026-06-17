import numpy as np
import os
# from np.lib.histograms import histogram


import torch
import torch.distributed as dist
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from model.losses import VGGLoss
from model.network.hf_baseline import HierarchyFlow
from model.utils.dataset_2 import get_dataset
from model.utils.sampler import DistributedGivenIterationSampler, DistributedTestSampler
from tensorboardX import SummaryWriter

import logging
from model.utils.log_helper import init_log
from model.trainers.MR_signal_model_2 import *
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True

init_log('pytorch hierarchy flow')
global_logger = logging.getLogger('pytorch hierarchy flow')


def mr_forward(q_map, mr_signal_model, seq, k_sort):
    return mr_signal_model(q_map, seq, k_sort)

def model_exp(model, content_images, style_images):
    outputs = model(content_images, style_images)
    outputs = torch.clamp(outputs, 0, 1)
    denorm_outputs = denormalize_per_image(outputs, [0.0, 0.0, 0.0], [3.5, 0.45, 1.0])

    denorm_outputs = denorm_outputs[:, :, 3:253, 131:381]
    return denorm_outputs





def save_checkpoint(state, filename):
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['step']


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class Trainer():
    def __init__(self, cfg, local_rank, world_size):
        self.cfg = cfg
        self.rank = local_rank
        self.world_size = world_size

        model = HierarchyFlow(self.cfg.network.pad_size, self.cfg.network.in_channel, self.cfg.network.out_channels,
                              self.cfg.network.weight_type)
        # model.cuda(self.rank)
        model = model.to(device='cuda')

        if self.rank == 0:
            global_logger.info(self.cfg)
            global_logger.info(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

        # if self.cfg.eval_mode or (self.cfg.resume and os.path.isfile(self.cfg.load_path)):
        if self.cfg.resume:
            self.model, self.optimizer, self.resumed_step = load_checkpoint('checkpoint_for_resume/0.ckpt.pth.tar',
                                                                            model, optimizer)
            global_logger.info(
                "=> loaded checkpoint '{}' with current step {}".format(self.cfg.load_path, self.resumed_step))
        else:
            self.model = model
            self.optimizer = optimizer
            self.resumed_step = -1

        if self.cfg.lr_scheduler.type == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.max_iter,
                                                                           self.cfg.lr_scheduler.eta_min)
        else:
            raise RuntimeError('lr_scheduler {} is not implemented'.format(self.cfg.lr_scheduler))

        self.criterion = VGGLoss(self.cfg.loss.vgg_encoder).cuda(self.rank)

        if self.rank == 0:
            self.logger = SummaryWriter(os.path.join(self.cfg.output, self.cfg.task_name, 'runs'))

    def train(self):
        train_dataset = get_dataset(self.cfg.dataset.train)
        # train_sampler = DistributedGivenIterationSampler(train_dataset,
        # self.cfg.max_iter, self.cfg.dataset.train.batch_size, world_size=self.world_size, rank=self.rank, last_iter=-1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.dataset.train.batch_size,
            shuffle=True)  # sampler=train_sampler num_workers=4 pin_memory=False

        seq_T1 = mr0.Sequence.import_file('C:/Users/kseniia.belousova/Documents/HierarchyFlow/seq/TSE_Ksenia_128_T1w.seq')
        seq_T2 = mr0.Sequence.import_file('C:/Users/kseniia.belousova/Documents/HierarchyFlow/seq/TSE_210525_1535_6_T2w.seq')
        seq_PD = mr0.Sequence.import_file('C:/Users/kseniia.belousova/Documents/HierarchyFlow/seq/TSE_Ksenia_128_PDw.seq')

        for ep in range(self.cfg.epochs):
            for batch_id, batch in enumerate(train_loader):
                b_id = batch_id + (len(train_loader) * ep)
                self.train_iter(b_id, batch, seq_T1, seq_T2, seq_PD)

        self.eval()

    def eval(self):
        test_dataset = get_dataset(self.cfg.dataset.test)
        test_sampler = DistributedTestSampler(test_dataset, world_size=self.world_size, rank=self.rank)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.dataset.test.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            sampler=test_sampler)
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                content_images = batch[0].cuda(self.rank)
                style_images = batch[1].cuda(self.rank)
                names = batch[2]
                outputs = self.model(content_images, style_images)
                outputs = torch.clamp(outputs, 0, 1)
                outputs = outputs.cpu()

                for idx in range(len(outputs)):
                    output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results', 'pred', names[idx])
                    save_image(outputs[idx].unsqueeze(0), output_name)
                    if idx == 0:
                        output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results', 'cat_img',
                                                   names[idx])
                        output_images = torch.stack((content_images[idx].cpu(), style_images[idx].cpu(), outputs[idx]),
                                                    0)
                        save_image(output_images, output_name, nrow=1)
                if self.rank == 0 and batch_id % 10 == 1:
                    global_logger.info('predicting {}th batch...'.format(batch_id))
        if self.rank == 0:
            global_logger.info('Save predictions to {}\nDone.'.format(
                os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results')))

    def train_iter(self, batch_id, batch, seq_t1, seq_t2, seq_pd):
        content_images = batch[0].cuda(self.rank)
        style_images = batch[1].cuda(self.rank)
        denorm_images = batch[2].cuda(self.rank)
        #max_vals = batch[3].cuda(self.rank)

        #outputs = self.model(content_images, style_images)
        denorm_outputs = checkpoint(model_exp, self.model, content_images.requires_grad_(), style_images)


        mr_model = MRSignalModel()
        #reconstruct_images = mr_model(denorm_outputs, max_vals)
        # reconstruct_images = mr_model(outputs, max_vals)

        train_reconstruction_t1 = checkpoint(mr_forward, denorm_outputs.requires_grad_(), mr_model, seq_t1,
                                             False)
        train_reconstruction_t2 = checkpoint(mr_forward, denorm_outputs.requires_grad_(), mr_model, seq_t2,
                                             True)
        train_reconstruction_pd = checkpoint(mr_forward, denorm_outputs.requires_grad_(), mr_model, seq_pd,
                                             False)

        new_output_weighted = []
        sim_image_list = [train_reconstruction_t1, train_reconstruction_t2, train_reconstruction_pd]
        res_weighted_tensor = torch.stack(sim_image_list, dim=0)

        new_output_weighted.append(res_weighted_tensor)

        reconstruct_images = torch.stack(new_output_weighted, dim=0)

        loss_MSE = nn.MSELoss()

        loss = loss_MSE(reconstruct_images.float(), denorm_images.float())

        # torch.distributed.barrier()

        # loss = reduce_mean(loss, self.world_size)
        # loss_c = reduce_mean(loss_c, self.world_size)
        # loss_s = reduce_mean(loss_s, self.world_size)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        plt.imshow(denorm_images[0, 0, :, :], cmap='gray')
        plt.show()
        plt.imshow(reconstruct_images[0, 0, :, :], cmap='gray')
        plt.show()

        if self.rank == 0:
            current_lr = self.lr_scheduler.get_lr()[0]
            self.logger.add_scalar("current_lr", current_lr, batch_id + 1)
            self.logger.add_scalar("loss_c", loss.item(), batch_id + 1)
            self.logger.add_scalar("loss_s", loss.item(), batch_id + 1)
            self.logger.add_scalar("loss", loss.item(), batch_id + 1)

        if batch_id % self.cfg.print_freq == 0:
            global_logger.info(
                'batch: {}, style_loss: {}, content_loss: {}, loss: {}'.format(batch_id, loss.item(), loss.item(),
                                                                               loss.item()))
            output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'img_save', str(batch_id) + '.jpg')
            output_images = torch.cat((content_images.cpu(), reconstruct_images[0, 0, :, :].cpu(), outputs.cpu()), 0)
            save_image(output_images, output_name, nrow=1)

        if batch_id % self.cfg.save_freq == 0:
            save_checkpoint({
                'step': batch_id,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, os.path.join(self.cfg.output, self.cfg.task_name, 'model_save', str(batch_id) + '.ckpt'))