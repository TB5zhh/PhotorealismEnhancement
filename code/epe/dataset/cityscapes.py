from collections import namedtuple
import IPython
import imageio
import numpy as np
import torch
from .robust_labels import RobustlyLabeledDataset
from .batch_types import EPEBatch
from .utils import mat2tensor

TypeCls = namedtuple('Category', ['name', 'csId', 'train_id'])
CITYSCAPES_CATES = (
    TypeCls(  'sky'                  , 23 , 0),
    TypeCls(  'road'                 ,  7 , 1),
    TypeCls(  'sidewalk'             ,  8 , 1),
    TypeCls(  'parking'              ,  9 , 1),
    TypeCls(  'rail track'           , 10 , 1),
    TypeCls(  'car'                  , 26 , 2),
    TypeCls(  'truck'                , 27 , 2),
    TypeCls(  'bus'                  , 28 , 2),
    TypeCls(  'caravan'              , 29 , 2),
    TypeCls(  'trailer'              , 30 , 2),
    TypeCls(  'train'                , 31 , 2),
    TypeCls(  'motorcycle'           , 32 , 2),
    TypeCls(  'bicycle'              , 33 , 2),
    TypeCls(  'terrain'              , 22 , 3),
    TypeCls(  'vegetation'           , 21 , 4),
    TypeCls(  'person'               , 24 , 5),
    TypeCls(  'rider'                , 25 , 5),
    TypeCls(  'pole'                 , 17 , 6),
    TypeCls(  'polegroup'            , 18 , 6),
    TypeCls(  'traffic light'        , 19 , 7),
    TypeCls(  'traffic sign'         , 20 , 8),
    TypeCls(  'building'             , 11 , 9),
    TypeCls(  'wall'                 , 12 , 9),
    TypeCls(  'fence'                , 13 , 9),
    TypeCls(  'guard rail'           , 14 , 9),
    TypeCls(  'bridge'               , 15 , 9),
    TypeCls(  'tunnel'               , 16 , 9),
    TypeCls(  'static'               ,  4 , 10),
    TypeCls(  'dynamic'              ,  5 , 10),
    TypeCls(  'ground'               ,  6 , 10),
    TypeCls(  'unlabeled'            ,  0 , 10),
    TypeCls(  'ego vehicle'          ,  1 , 11),
    TypeCls(  'rectification border' ,  2 , 11),
    TypeCls(  'out of roi'           ,  3 , 11),
    TypeCls(  'license plate'        , -1 , 11),
)

def transform_labels(original_label_map):
    label_maps = [(original_label_map == csId).astype(np.long) * train_id for _, csId, train_id in CITYSCAPES_CATES]
    label_map = np.sum(np.stack(label_maps, axis=0), axis=0, keepdims=True)
    return label_map


class Cityscapes(RobustlyLabeledDataset):
    def __init__(self, name, img_and_robust_label_paths, img_transform=None, label_transform=None):
        super().__init__(name, img_and_robust_label_paths, img_transform, label_transform)

    def __getitem__(self, index):

        idx = index % self.__len__()
        img_path = self.paths[idx]
        img = self._load_img(img_path)

        if self.transform is not None:
            img = self.transform(img)
            pass

        img = mat2tensor(img)

        label_path = self._img2label[img_path]
        robust_labels = imageio.imread(label_path)

        robust_labels = torch.LongTensor(transform_labels(robust_labels))

        return EPEBatch(img, path=img_path, robust_labels=robust_labels)