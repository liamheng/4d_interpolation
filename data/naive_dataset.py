import os
import random

from PIL import Image

from data import BaseDataset


def random_switch(a, b):
    if random.random() < 0.5:
        return a, b
    else:
        return b, a


class NaiveDataset(BaseDataset):
    """ A dataset class for labeled image dataset.

        The file structure should be:
        - data_root
            - 0
                - moving.png (or moving.nii.gz for 3D)
                - fixed.png (or fixed.nii.gz for 3D)
            - 1
            - 2
            ...
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(preprocess=['linear3d_0.00392_0'])  # 缩放到1/255，即从[0,255]缩到[0,1]
        parser.add_argument('--random_mode', type=str, default='none',
                            choices=['none', 'random_direction', 'random_but_keep_direction', 'fully_random'],
                            help='shuffle mode')
        return parser

    def __init__(self, opt):
        """ Initialize this dataset class.

        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.opt = opt
        self.len = len(os.listdir(self.opt.data_dirname))

    def __getitem__(self, index):
        """ Return a data dict and its metadata information.

        :param index: an integer for data indexing
        :return a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        if self.opt.random_mode == 'none':
            moving_path_prefix, fixed_path_prefix = self.get_prefix(index, index)
        elif self.opt.random_mode == 'random_direction':
            moving_path_prefix, fixed_path_prefix = random_switch(*self.get_prefix(index, index))
        elif self.opt.random_mode == 'random_but_keep_direction':
            moving_path_prefix, fixed_path_prefix = self.get_prefix(random.randint(0, self.len - 1),
                                                                    random.randint(0, self.len - 1))
        elif self.opt.random_mode == 'fully_random':
            moving_path_prefix, fixed_path_prefix = random_switch(
                *self.get_prefix(random.randint(0, self.len - 1), random.randint(0, self.len - 1)))
        else:
            raise NotImplementedError('random_mode %s is not implemented.' % self.opt.random_mode)

        if self.opt.is_3d:
            moving_path = moving_path_prefix + '.nii.gz'
            moving, *_ = self.load_nifti(moving_path)
            fixed_path = fixed_path_prefix + '.nii.gz'
            fixed, *_ = self.load_nifti(fixed_path)
            if self.opt.assess_segmentation_offline:
                moving_segmentation_path = moving_path_prefix + '_segmentation.nii.gz'
                moving_segmentation, *_ = self.load_nifti(moving_segmentation_path)
                fixed_segmentation_path = fixed_path_prefix + '_segmentation.nii.gz'
                fixed_segmentation, *_ = self.load_nifti(fixed_segmentation_path)
        else:  # 2d
            moving_path = moving_path_prefix + '.png'
            moving = Image.open(moving_path).convert('RGB')
            fixed_path = fixed_path_prefix + '.png'
            fixed = Image.open(fixed_path).convert('RGB')
            if self.opt.assess_segmentation_offline:
                moving_segmentation_path = moving_path_prefix + '_segmentation.png'
                moving_segmentation = Image.open(moving_segmentation_path).convert('L')
                fixed_segmentation_path = fixed_path_prefix + '_segmentation.png'
                fixed_segmentation = Image.open(fixed_segmentation_path).convert('L')

        pair = BaseDataset.new_transform_pair(self.opt)  # 每个样本新建
        moving = pair.apply_image(moving)
        fixed = pair.apply_image(fixed)
        if self.opt.assess_segmentation_offline:
            moving_segmentation = pair.apply_segmentation(moving_segmentation)
            fixed_segmentation = pair.apply_segmentation(fixed_segmentation)

            return {'moving': moving, 'fixed': fixed,
                    'moving_segmentation': moving_segmentation,
                    'fixed_segmentation': fixed_segmentation,
                    'moving_path': moving_path,
                    'fixed_path': fixed_path}

        return {'moving': moving, 'fixed': fixed, 'moving_path': moving_path, 'fixed_path': fixed_path}

    def get_prefix(self, moving_index, fixed_index):
        return (os.path.join(self.opt.data_dirname, str(moving_index), 'moving'),
                os.path.join(self.opt.data_dirname, str(fixed_index), 'fixed'))

    def __len__(self):
        """ Return the total number of images in the dataset."""
        return self.len
