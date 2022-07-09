import logging
from pathlib import Path

import torch.utils.data
import numpy as np

from .batch_types import ImageBatch
from .utils import mat2tensor


class SyntheticDataset(torch.utils.data.Dataset):
    """ Synthetic datasets provide additional information about a scene.

	They may provide image-sized G-buffers, containing geometry, material, 
	or lighting informations, or semantic segmentation maps.

	"""

    def __init__(self, name):
        super(SyntheticDataset, self).__init__()
        self._name = name
        self._log = logging.getLogger(f'epe.dataset.{self._name}')
        pass

    @property
    def name(self):
        return self._name

    @property
    def num_gbuffer_channels(self):
        """ Number of image channels the provided G-buffers contain."""
        raise NotimplementedError

    @property
    def num_classes(self):
        """ Number of classes in the semantic segmentation maps."""
        raise NotimplementedError

    @property
    def cls2gbuf(self):
        return NotimplementedError


class SyntheticNpz(SyntheticDataset):

    def __init__(self, name, npz_list_file, dataset_root='.'):
        super().__init__(name)
        if dataset_root is None:
            dataset_root = '.'
        with open(Path(dataset_root) / npz_list_file) as f:
            self.npz_files = [Path(dataset_root) / i.strip() for i in f.readlines()]
        # self.num_classes = -1
        # self.num_classes = -1
        # self.cls2gbuf = -1

    def __getitem__(self, index):
        arr = np.load(self.npz_files[index])['arr_0']
        return ImageBatch(mat2tensor(arr[:, :, :3]))

    def __len__(self):
        return len(self.npz_files)
