import os.path as osp
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class OpenEarthMapDataset(BaseSegDataset):
    """OpenEarthMap dataset.

    In segmentation map annotation for OpenEarthMap, 0 is the ignore index.
    ``reduce_zero_label`` should be set to False. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.tif'.
    """
    METAINFO = dict(
        classes=('background', 'bareland', 'grass', 'pavement', 'road', 'tree',
                 'water', 'cropland', 'building'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148],
                 [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73],
                 [222, 31, 7]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class UDD5Dataset(BaseSegDataset):
    """UDD5 dataset.
    
    """
    METAINFO = dict(
    classes=('vegetation', 'building', 'road', 'vehicle',
             'other'),
    palette=[[107, 142, 35], [102,102,156], [128,64,128],
             [0, 0, 142], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)









