"""
Train code for the pix2pixCC model
"""

from pix2pixCC_Options import TrainOption
opt = TrainOption().parse()

import os
import datetime
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from pix2pixCC_Networks import Discriminator, Generator, Loss
from pix2pixCC_Pipeline import CustomDataset
from pix2pixCC_Utils import Manager, weights_init, update_lr



#==============================================================================

if __name__ == '__main__':
    # Check the number of GPUs for training
    USE_CUDA = torch.cuda.is_available()
    # 분산 학습 사용 여부: CUDA_VISIBLE_DEVICES에 설정된 GPU 수를 기반으로 결정하거나,
    # torchrun을 사용해 DDP 환경에서 실행할 경우 WORLD_SIZE가 1 이상으로 자동 설정됩니다.
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) if os.environ.get('CUDA_VISIBLE_DEVICES') else 1
    use_ddp = True if num_gpus > 1 else False

    if use_ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=9000))
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        if rank == 0:
            print(f'Rank: {rank}, World size: {world_size}')
    else:
        rank = 0
        world_size = 1

    #--------------------------------------------------------------------------
    start_time = datetime.datetime.now()

    #--------------------------------------------------------------------------
    # [1] Initial Conditions Setup
    
    torch.backends.cudnn.benchmark = False

    cuda_num = torch.cuda.current_device() if USE_CUDA else torch.device('cpu')
    device = f'cuda:{cuda_num}'
    
    dtype = torch.float16 if opt.data_type == 16 else torch.float32
    print(device)
    
    if opt.val_during_train:
        from pix2pixCC_Options import TestOption
        test_opt = TestOption().parse()
        save_freq = opt.save_freq

    init_lr = opt.lr
    lr = opt.lr
    batch_sz = opt.batch_size
    
    
    # --- Dataset upload ---
    dataset = CustomDataset(opt)
    test_dataset = CustomDataset(test_opt)
    if use_ddp:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=not opt.no_shuffle)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_sz,
                                 sampler=train_sampler,
                                 num_workers=opt.n_workers)

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=not opt.no_shuffle)
        test_data_loader = DataLoader(dataset=test_dataset,
                                    batch_size=1, # test_opt.batch_size,
                                    num_workers=test_opt.n_workers,
                                    sampler=test_sampler,
                                    shuffle=not test_opt.no_shuffle)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_sz,
                                 num_workers=opt.n_workers,
                                 shuffle=not opt.no_shuffle)
        test_data_loader = DataLoader(dataset=test_dataset,
                                    batch_size=1, #test_opt.batch_size,
                                    num_workers=test_opt.n_workers,
                                    shuffle=not test_opt.no_shuffle)
    
    # --- Network and Optimizer update ---
    G_model = Generator(opt).to(device=device, dtype=dtype)
    D_model = Discriminator(opt).to(device=device, dtype=dtype)
    G_model.apply(weights_init)
    D_model.apply(weights_init)

    if use_ddp:
        G = torch.nn.parallel.DistributedDataParallel(G_model, device_ids=[rank])
        D = torch.nn.parallel.DistributedDataParallel(D_model, device_ids=[rank])
    else:
        G = torch.nn.DataParallel(G_model)
        D = torch.nn.DataParallel(D_model)

    G_optim = torch.optim.AdamW(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    D_optim = torch.optim.AdamW(D.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    criterion = Loss(opt, device)
    
    
    # --- Resume check --- 중지된 시점부터 
    G_init_path = opt.model_dir + '/' + str(opt.latest_iter) + '_G.pt'
    D_init_path = opt.model_dir + '/' + str(opt.latest_iter) + '_D.pt'
    
    if os.path.isfile(G_init_path) and os.path.isfile(D_init_path) :
        init_iter = opt.latest_iter
        print("Resume at iteration: ", init_iter)
        
        G.module.load_state_dict(torch.load(G_init_path))
        D.module.load_state_dict(torch.load(D_init_path))

        init_epoch = int(float(init_iter)/(batch_sz*len(data_loader)))
        current_step = int(init_iter)

    else:
        init_epoch = 1
        current_step = 0
   
    manager = Manager(opt)
    
    
    #--------------------------------------------------------------------------
    # [2] Model training
    
    total_step = opt.n_epochs * len(data_loader) * batch_sz
    best_RMSE = 1000000
    best_epoch = 0

    for epoch in range(init_epoch, opt.n_epochs + 1):
        for input, target, _, _ in tqdm(data_loader, desc=f"Training at epoch : {epoch}"):
            G.train()
            D.train()

            current_step += batch_sz
            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
            
            #import sys
            #sys.stdout.write(f"{target.shape}\n") # torch.Size([B, 1, 256, 256])
            #sys.stdout.flush()


            D_loss, G_loss, target_tensor, generated_tensor = criterion(D, G, input, target)
            # 34, 17, 7, 3
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'D_loss': D_loss.detach().item(),
                       'G_loss': G_loss.detach().item(),
                       'D_state_dict': D.module.state_dict(),
                       'G_state_dict': G.module.state_dict(),
                       'D_optim_state_dict': D_optim.state_dict(),
                       'G_optim_state_dict': G_optim.state_dict(),
                       'target_tensor': target_tensor,
                       'generated_tensor': generated_tensor.detach()
                       }

            manager(package)
        if use_ddp and dist.is_initialized():
            dist.barrier()
    #--------------------------------------------------------------------------
    # [3] Model Checking 
        if rank == 0: #and epoch % 5 == 0:
            if opt.val_during_train:
                with torch.no_grad():
                    G.eval()
                    test_image_dir = os.path.join(test_opt.image_dir, str(epoch))
                    os.makedirs(test_image_dir, exist_ok=True)
                    test_model_dir = test_opt.model_dir

                    for p in G.parameters():
                        p.requires_grad_(False)

                    total_RMSE = 0

                    for input, target, in_name, tar_name in tqdm(test_data_loader):
                        input = input.to(device=device, dtype=dtype)
                        target = target.to(device=device, dtype=dtype)
                        fake, mu, logvar = G(input)
                        
                        total_RMSE += torch.sum(torch.abs(fake - target)).item() #torch.mean((fake - target)**2).item()

                        np_fake = fake.cpu().numpy().squeeze(axis=0) # (1, 720, 576)
                        np_real = target.cpu().numpy().squeeze(axis=0) # (3, 720, 576)

                        #import sys
                        #sys.stdout.write(f"{np_fake.shape}, {np_real.shape}\n")
                        #sys.stdout.flush()


                        #print("shape : ", np_fake.shape, np_real.shape, input.shape)
                        
                        if opt.display_scale != 1: # display_scale = 1
                            sav_fake = np.clip(np_fake*np.float(opt.display_scale), -1, 1)
                            sav_real = np.clip(np_real*np.float(opt.display_scale), -1, 1)
                        else:
                            sav_fake = np_fake
                            sav_real = np_real
                        if epoch == 1:
                            manager.save_image(torch.from_numpy(sav_real), path=os.path.join(test_image_dir, 'Check_{:d}_'.format(epoch)+ tar_name[0] + '_real.png'))    
                        manager.save_image(torch.from_numpy(sav_fake), path=os.path.join(test_image_dir, 'Check_{:d}_'.format(epoch)+ in_name[0] + '_fake.png'))

                    total_RMSE = total_RMSE / len(test_data_loader)    

                    if best_RMSE >= total_RMSE:
                        best_RMSE = total_RMSE
                        best_epoch = epoch
                    
                    print(f"Diff at {epoch}: {total_RMSE}")
                    print(f"Best Diff = {best_RMSE}", f"Best Epoch = {best_epoch}")

                    for p in G.parameters():
                        p.requires_grad_(True)
        
        
    #--------------------------------------------------------------------------    


        if epoch > opt.epoch_decay :
            lr = update_lr(lr, init_lr, opt.n_epochs - opt.epoch_decay, D_optim, G_optim)

    end_time = datetime.datetime.now()
    
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()
    
    print("Total time taken: ", end_time - start_time)
    
    


#==============================================================================
