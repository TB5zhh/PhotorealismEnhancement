import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from .dataset import ImageBatch, ImageDataset, SyntheticNpz
from .dataset.utils import read_filelist
from .network import VGG16


def seed_worker(id):
	np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
	pass


if __name__ == '__main__':

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--img_list', type=Path, help="Path to csv file with path to images in first column.")
    parser.add_argument('-D', '--dataset_root', type=str, help='Root directory of the dataset', default=None)
    parser.add_argument('-O', '--out_dir', type=Path, help="Where to store the crop info.", default='.')
    parser.add_argument('-C', '--num_crops', type=int, help="Number of crops to sample per image. Default = 15.", default=30)
    parser.add_argument('-S', '--crop_size', nargs='+', type=int, help="Max, min of crop size. Default = 196.", default=[196, 196])
    parser.add_argument('--dataset_type', type=str, default='Default', help='Dataset class to load data')
    parser.add_argument('--num_loaders', type=int, default=1)
    args = parser.parse_args()

    network   = VGG16(False, padding='none').to(device)
    extract   = lambda img: network.fw_relu(img, 13)[-1]
    crop_size = args.crop_size # VGG-16 receptive field at relu 5-3
    dim       = 512 # channel width of VGG-16 at relu 5-3
    num_crops = args.num_crops

    if args.dataset_type == 'Default':
        dataset = ImageDataset('TempDataset', read_filelist(args.img_list, 1, False, args.dataset_root))
    elif args.dataset_type == 'Npz':
        dataset = SyntheticNpz('TempDataset', args.img_list, args.dataset_root)

    
    # compute mean/std

    loader  = torch.utils.data.DataLoader(dataset, \
        batch_size=1, shuffle=True, \
        num_workers=args.num_loaders, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=ImageBatch.collate_fn)

    print('Computing mean/std...')

    m, s = [], []
    for i, batch in enumerate(tqdm(loader)):
        m.append(batch.img.mean(dim=(2,3)))
        s.append(batch.img.std(dim=(2,3)))
        pass

    m = torch.cat(m, 0).mean(dim=0)
    s = torch.cat(s, 0).mean(dim=0)

    network.set_mean_std(m[0], m[1], m[2], s[0], s[1], s[2])
    
    loader  = torch.utils.data.DataLoader(dataset, \
        batch_size=1, shuffle=False, \
        num_workers=args.num_loaders, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=ImageBatch.collate_fn)

    features = np.zeros((len(dataset) * num_crops, dim), np.float16)

    print('Sampling crops...')

    ip = 0
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out_dir / f'crop.csv', 'w') as log:
        log.write('id,path,r0,r1,c0,c1\n')
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                
                n,_,h,w = batch.img.shape
                assert n == 1

                if i == 0:
                    print(f'Image size is {h}x{w} - sampling {num_crops} crops per image.')
                    pass
                
                crop_size_taken = random.randint(crop_size[0], crop_size[1])
                c0s = torch.randint(w-crop_size_taken+1, (num_crops,1))
                r0s = torch.randint(h-crop_size_taken+1, (num_crops,1))

                samples = []
                for j in range(num_crops):
                    r0 = r0s[j].item()
                    c0 = c0s[j].item()
                    r1 = r0 + crop_size_taken
                    c1 = c0 + crop_size_taken
                    samples.append(torch.nn.functional.interpolate(batch.img[0:1,:,r0:r1,c0:c1], 196, mode='bilinear'))
                    log.write(f'{ip},{batch.path[0]},{r0},{r1},{c0},{c1}\n')
                    ip += 1
                    pass

                samples = torch.cat(samples, 0)
                samples = samples.to(device, non_blocking=True)
                f = extract(samples)
                
                features[ip-num_crops:ip,:] = f.cpu().numpy().astype(np.float16).reshape(num_crops, dim)
                pass
            pass
        pass

    print('Saving features.')
    np.savez_compressed(args.out_dir / f'crop', crops=features)
