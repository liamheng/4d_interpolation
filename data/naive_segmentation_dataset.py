import os

from PIL import Image

from data import BaseDataset


class NaiveSegmentationDataset(BaseDataset):
    """ A dataset class for labeled image dataset.

        The file structure should be:
        - data_root
            - 0
                - image.png (original image)
                - label.png (ground truth)
                - mask.png (used to ignore unwanted pixels)
            - 1
            - 2
            ...
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_mask', action='store_true', help='whether the dataset has mask')
        parser.add_argument('--image_name', type=str, default='image', help='name of the image file')
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
        if self.opt.is_3d:
            image_path = os.path.join(self.opt.data_dirname, str(index), self.opt.image_name + '.nii.gz')
            image, *_ = self.load_nifti(image_path)
            label_path = os.path.join(self.opt.data_dirname, str(index), 'label.nii.gz')
            label, *_ = self.load_nifti(label_path)
        else:  # 2d
            image_path = os.path.join(self.opt.data_dirname, str(index), self.opt.image_name + '.png')
            image = Image.open(image_path).convert('RGB')
            label_path = os.path.join(self.opt.data_dirname, str(index), 'label.png')
            label = Image.open(label_path).convert('L')

        pair_transform = BaseDataset.new_transform_pair(self.opt)  # 每个样本新建
        image = pair_transform.apply_image(image)
        label = pair_transform.apply_label(label)

        if not self.opt.no_mask:
            if self.opt.is_3d:
                mask = self.load_nifti(os.path.join(self.opt.data_dirname, str(index), 'mask.nii.gz'))
            else:
                mask = Image.open(os.path.join(self.opt.data_dirname, str(index), 'mask.png')).convert('L')
            mask = pair_transform.apply_label(mask)
            return {'image_original': image, 'label': label, 'source_path': image_path, 'mask': mask}

        return {'image_original': image, 'label': label, 'source_path': image_path}

    def __len__(self):
        """ Return the total number of images in the dataset."""
        return self.len
