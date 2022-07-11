# %%
import argparse
from collections import namedtuple
from enum import Enum, IntEnum, auto, unique
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

GBUFFER_RAW = [
    '-SceneDepth.png',
    '-SceneColor.png',
    '-SceneGBufferA.png',
    '-SceneGBufferB.png',
    '-SceneGBufferC.png',
    '-SceneGBufferD.png',
]


def save_one_channel(arr, out):
    Image.fromarray(
        np.stack([(arr / arr.max() * 255).astype(np.uint8) for _ in range(3)],
                 axis=2)).save(out)


def save_three_channel(arr, out):
    Image.fromarray((arr / arr.max() * 255).astype(np.uint8)).save(out)


# %%
@unique
class CHANNEL(IntEnum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    DEPTH = auto()
    SCENE_RED = auto()
    SCENE_GREEN = auto()
    SCENE_BLUE = auto()
    NORMAL_X = auto()
    NORMAL_Y = auto()
    NORMAL_Z = auto()
    TEXTURE_RED = auto()
    TEXTURE_GREEN = auto()
    TEXTURE_BLUE = auto()
    BUFFER_B_0 = auto()
    BUFFER_B_1 = auto()
    BUFFER_B_2 = auto()
    BUFFER_D_0 = auto()
    BUFFER_D_1 = auto()
    BUFFER_D_2 = auto()
    MASK_CLS_0 = auto()
    MASK_CLS_1 = auto()
    MASK_CLS_2 = auto()
    MASK_CLS_3 = auto()
    MASK_CLS_4 = auto()
    MASK_CLS_5 = auto()
    MASK_CLS_6 = auto()
    MASK_CLS_7 = auto()
    MASK_CLS_8 = auto()
    MASK_CLS_9 = auto()
    MASK_CLS_10 = auto()
    MASK_CLS_11 = auto()
DATASET_ROOT = 'data/anomaly_dataset/v0.1.1'
CUT_HEAD = 10


# %%
class CarlaPalette:

    def __init__(self) -> None:
        TypeCls = namedtuple('Category', ['name', 'color', 'train_id'])
        self.num_train_id = 13
        self.categories = [
            TypeCls('sky', (70, 130, 180), 0),
            TypeCls('road', (128, 64, 128), 1),
            TypeCls('sidewalk', (244, 35, 232), 1),
            TypeCls('ground', (81, 0, 81), 1),
            TypeCls('static', (110, 190, 160), 1),
            TypeCls('road_line', (157, 234, 50), 1),
            TypeCls('car', (0, 0, 142), 2),
            TypeCls('terrain', (145, 170, 100), 3),
            TypeCls('vegetation', (107, 142, 35), 4),
            TypeCls('person', (220, 20, 60), 5),
            TypeCls('pole', (153, 153, 153), 6),
            TypeCls('traffic_light', (250, 170, 30), 7),
            TypeCls('traffic_sign', (220, 220, 0), 8),
            TypeCls('wall', (102, 102, 156), 10),
            TypeCls('rail_track', (230, 150, 140), 10),
            TypeCls('guard_rail', (180, 165, 180), 10),
            TypeCls('building', (70, 70, 70), 10),
            TypeCls('fence', (100, 40, 40), 10),
            TypeCls('bridge', (150, 100, 100), 10),
            TypeCls('other', (55, 90, 80), 11),
            TypeCls('dynamic', (170, 120, 50), 11),
            TypeCls('water', (45, 60, 150), 11),
            TypeCls('unlabeled', (0, 0, 0), 12),
        ]
        self.train_cates = [[
            cate for cate in self.categories if cate.train_id == train_id
        ] for train_id in range(self.num_train_id)]


# %%
def test(result):
    save_three_channel(result[:, :, :3], '0.png')
    save_one_channel(result[:, :, 3], '1.png')
    save_three_channel(result[:, :, 4:7], '2.png')
    save_three_channel(result[:, :, 7:10], '3.png')
    save_three_channel(result[:, :, 10:13], '4.png')
    save_one_channel(result[:, :, 13], '5.png')
    save_one_channel(result[:, :, 14], '6.png')
    save_one_channel(result[:, :, 15], '7.png')
    save_three_channel(result[:, :, 16:19], '8.png')
    for i in range(20, 32):
        save_one_channel(result[:, :, i], f'{i}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_dir', type=Path)
    parser.add_argument('--cut_head', type=int,default=10)
    args = parser.parse_args()
    path_list = []
    for root, dirs, files in os.walk(args.dataset_dir):
        if 'rgb_v' not in dirs or 'depth_v' not in dirs:
            continue
        if len(os.listdir(Path(root) / 'rgb_v')) < 100:
            continue
        int_sorted = lambda x: sorted(
            x, key=lambda y: int(y.split('.')[0].split('-')[0]))
        (
            rgb_list,
            depth_list,
            gbuffer_list,
            mask_list,
        ) = (
            int_sorted(os.listdir(Path(root) / 'rgb_v')),
            int_sorted(os.listdir(Path(root) / 'depth_v')),
            int_sorted(os.listdir(Path(root) / 'gbuffer_v')),
            int_sorted(os.listdir(Path(root) / 'mask_v')),
        )
        load = lambda x: np.asarray(Image.open(x)).astype(np.float64) / 255.0
        load_depth = lambda x: np.einsum(
            'xya,a->xy', np.asarray(Image.open(x)),
            np.array((1, 256, 65536)) / (256 * 256 * 256 - 1) * 1000)
        for idx, (rgb_f, depth_f,
                mask_f) in enumerate(zip(tqdm(rgb_list), depth_list, mask_list)):
            if idx < args.cut_head:
                continue
            rgbs = load(Path(root) / 'rgb_v' / rgb_f).astype(np.float32)
            try:
                channels = [
                    load_depth(Path(root) / 'depth_v' / depth_f)[:, :, np.newaxis],
                    load(Path(root) / 'gbuffer_v' / f'{idx}-SceneColor.png'),
                    load(Path(root) / 'gbuffer_v' / f'{idx}-SceneGBufferA.png'),
                    load(Path(root) / 'gbuffer_v' / f'{idx}-SceneGBufferC.png'),
                    load(Path(root) / 'gbuffer_v' / f'{idx}-SceneGBufferB.png'),
                    load(Path(root) / 'gbuffer_v' / f'{idx}-SceneGBufferD.png'),
                ]
            except FileNotFoundError:
                break

            masks = []
            mask = np.asarray(Image.open(Path(root) / 'mask_v' / mask_f))
            p = CarlaPalette()
            for i in range(p.num_train_id):
                if i == 9:
                    continue
                masks.append(
                    np.stack(
                        [(mask == c.color).all(axis=2) for c in p.train_cates[i]],
                        axis=2,
                    ).any(axis=2, keepdims=True).astype(np.uint8) * i)
            gbuffers = np.concatenate(channels, axis=2).astype(np.float32)
            masks = np.sum(np.stack(masks, axis=2), axis=2).astype(np.uint8) 
            os.makedirs(Path(root) / 'data', exist_ok=True)
            np.savez(Path(root) / 'data' / f'{idx}.npz',
                    rgbs=rgbs,
                    gbuffers=gbuffers,
                    masks=masks)
            path_list.append(
                str(Path(root) / 'rgb_v' / f'{idx}.png') + ',' +
                str(Path(root) / 'data' / f'{idx}.npz'))

    with open(Path(args.dataset_dir) / 'paths.txt', 'w') as f:
        for line in path_list:
            print(line, file=f)
