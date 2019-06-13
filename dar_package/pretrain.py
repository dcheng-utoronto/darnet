from __future__ import division, print_function

import torch
import os
import datetime
import heapq
import shutil
import numpy as np

import matplotlib
matplotlib.use('agg')

from tqdm import tqdm
import matplotlib.pyplot as plt

from dar_package.utils.train_utils import AverageMeter, unpack_sample, save_config
torch.manual_seed(1234)
np.random.seed(1234)


def run(cfg_exp, config_object, Dataset, Network, split_num):
    # Environment setup
    time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    exp_id = "{}_{}".format(cfg_exp['name'], time)
    save_folder = os.path.join(cfg_exp['save_path'], exp_id)
    keep_best = int(cfg_exp['keep_best'])
    checkpoint_heap = []
    val_losses = AverageMeter()
    global_iter = 0
    np.seterr(divide='ignore', invalid='ignore')

    # Sanity checks
    assert keep_best > 0
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        print("Created {}".format(save_folder))
        save_config(config_object, save_folder)
    else:
        raise ValueError("Save folder already exists")

    batch_size = int(cfg_exp['batch_size'])
    num_workers = int(cfg_exp['num_workers'])
    num_epochs = int(cfg_exp['num_epochs'])
    learning_rate = float(cfg_exp['learning_rate'])
    weight_decay = float(cfg_exp['weight_decay'])
    patience = int(cfg_exp['patience'])
    save_delay = int(cfg_exp['save_delay'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get dataloaders
    train_dataset = Dataset(split='train_{}'.format(split_num))
    val_dataset = Dataset(split='val_{}'.format(split_num))
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # Get model and loss
    class ModelAndLoss(torch.nn.Module):
        def __init__(self):
            super(ModelAndLoss, self).__init__()
            self.net = Network()
            self.l1_loss_func = torch.nn.SmoothL1Loss()
        def forward(self, sample):
            output = self.net(sample['image'])
            assert len(output) == 3 or len(output) == 6
            beta, data, kappa = output[-3:]
            loss = self.l1_loss_func(beta, sample['distance_transform_outside'])
            loss += self.l1_loss_func(data, sample['distance_transform'])
            loss += self.l1_loss_func(kappa, sample['distance_transform_inside'])

            if len(output) == 6:
                beta0, data0, kappa0 = output[:3]
                loss += self.l1_loss_func(beta0, sample['distance_transform_outside'])
                loss += self.l1_loss_func(data0, sample['distance_transform'])
                loss += self.l1_loss_func(kappa0, sample['distance_transform_inside'])
            return loss, output
            
    model_and_loss = ModelAndLoss().to(device)

    # Get optimizer and scheduler
    optimizer = torch.optim.Adam(model_and_loss.net.parameters(), 
        lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
    if 'gamma' in cfg_exp:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=float(cfg_exp['gamma']), step_size=patience)
        print("Scheduler: StepLR with gamma {}".format(cfg_exp['gamma']))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=patience, verbose=True
        )
        print("Scheduler: ReduceLROnPlateau with factor 0.5")

    if len(train_dataset) // batch_size > 1000:
        epoch_iterator = range(num_epochs)
        train_iterator = tqdm(train_loader)
        val_iterator = tqdm(val_loader)
    else:
        epoch_iterator = tqdm(range(num_epochs))
        train_iterator = train_loader
        val_iterator = val_loader

    for epoch_num in epoch_iterator:
        if 'gamma' in cfg_exp:
            scheduler.step()

        # Train
        model_and_loss.train()
        for sample in train_iterator:
            unpack_sample(sample)
            optimizer.zero_grad()
            loss, output = model_and_loss(sample)
            loss.backward()
            optimizer.step()
            global_iter += 1

        # Val
        model_and_loss.eval()
        for sample in val_iterator:
            with torch.no_grad():
                unpack_sample(sample)
                loss, output = model_and_loss(sample)
                sample_batch_size = sample['image'].size(0)
                val_losses.update(loss, n=sample_batch_size)

        if 'gamma' not in cfg_exp:
            scheduler.step(val_losses.avg)
        
        # Save model; keep record of the validation loss in a heap
        # so the largest values can be popped off for checkpoint deletion
        if epoch_num + 1 >= save_delay:
            checkpoint_path = os.path.join(save_folder,
                "chk-{:04}.pth.tar".format(epoch_num))
            checkpoint = {
                'epoch': epoch_num + 1,
                'state_dict': model_and_loss.net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_losses.val
            }
            heapq.heappush(checkpoint_heap, (-val_losses.val, checkpoint_path))
            
            # Avoid saving checkpoints if not necessary to reduce bandwidth consumption
            if len(checkpoint_heap) > keep_best:
                _, checkpoint_to_delete = heapq.heappop(checkpoint_heap)
                if checkpoint_path == checkpoint_to_delete:
                    pass
                else:
                    torch.save(checkpoint, checkpoint_path)
                    os.remove(checkpoint_to_delete)
            else:
                torch.save(checkpoint, checkpoint_path)
        else:
            pass

        # On the very last epoch, save the checkpoint
        if epoch_num + 1 == num_epochs:
            checkpoint_path = os.path.join(save_folder,
                "chk-{:04}.pth.tar".format(epoch_num))
            torch.save(checkpoint, checkpoint_path)

        # Reset val losses for next epoch
        val_losses.reset()
        global_iter += 1

    # Set aside very best checkpoint
    _, best_checkpoint_path = checkpoint_heap[0]
    shutil.copyfile(best_checkpoint_path, os.path.join(save_folder, "best_chk.pth.tar"))

    return os.path.abspath(checkpoint_path)