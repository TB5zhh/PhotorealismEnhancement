from argparse import ArgumentParser
import logging
import os
from pathlib import Path

from tqdm import tqdm
import faiss
import numpy as np
from .matching.filter import load_and_filter_matching_crops, save_matching_crops

logger = logging.getLogger('epe.match_filter_calc')

def match_feature(src_dir, dst_dir):
    features_ref = np.load(Path(src_dir) / 'crop.npz')['crops'].astype(np.float32)
    features_ref = features_ref / np.sqrt(np.sum(np.square(features_ref), axis=1, keepdims=True))
    logger.info(f'Found {features_ref.shape[0]} crops for source dataset.')
    features_nn = np.load(Path(dst_dir) / 'crop.npz')['crops'].astype(np.float32)
    features_nn = features_nn / np.sqrt(np.sum(np.square(features_nn), axis=1, keepdims=True))
    logger.info(f'Found {features_nn.shape[0]} crops for target dataset.')
    assert features_nn.shape[1] == features_ref.shape[1]

    nn_index = faiss.IndexFlatL2(features_nn.shape[1])
    nn_index.add(features_nn)
    return nn_index.search(features_ref, args.k)


if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('file_src', type=Path, help="Path to feature file for source dataset.")
    p.add_argument('file_dst', type=Path, help="Path to feature file for target dataset.")
    p.add_argument('out_dir', type=Path, help="Path to output file with matches.")
    p.add_argument('--max_dist', type=float, default=1.0)
    p.add_argument('--height', type=int, help="Height of images in dataset.")
    p.add_argument('--width', type=int, help="Width of images in dataset.")
    p.add_argument('-k', type=int, help="Number of neighbours to sample. Default = 5.", default=5)
    args = p.parse_args()

    if not os.path.isfile(Path(args.file_src) / 'crop.npz'):
        raise RuntimeError('TODO')
    if not os.path.isfile(Path(args.file_dst) / 'crop.npz'):
        raise RuntimeError('TODO')
    if not os.path.isfile(Path(args.file_src) / 'crop.csv'):
        raise RuntimeError('TODO')
    if not os.path.isfile(Path(args.file_dst) / 'crop.csv'):
        raise RuntimeError('TODO')

    D, I = match_feature(args.file_src, args.file_dst)

    sc, dc = load_and_filter_matching_crops(D, I, Path(args.file_src) / 'crop.csv', Path(args.file_dst) / 'crop.csv', 1.0)
    os.makedirs(Path(args.out_dir), exist_ok=True)
    save_matching_crops(sc, dc, Path(args.out_dir) / 'filtered_matches.csv')

    #### Compute Weights
    d = np.zeros((args.height, args.width))
    print('Computing density...')
    for s in tqdm(sc): 
        d[s[1]:s[2],s[3]:s[4]] += 196 / (s[2] - s[1])

    print('Computing individual weights...')
    w = np.zeros((len(sc), 1)) 
    for i, s in enumerate(tqdm(sc)):
        w[i,0] = np.mean(d[s[1]:s[2],s[3]:s[4]])
        pass

    N = np.max(d)
    p = N / w
    np.savez_compressed(Path(args.out_dir) / "weight", w=p)
