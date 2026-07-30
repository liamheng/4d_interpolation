import json
import os

import torch

from data import BaseDataset


class VideoDataset(BaseDataset):
    """A dataset class for 4D video without segmentation.

    The file structure should be:
    - data_root
        - 0
            - video.nii.gz
            - info.json
        - 1
        - 2
        ...
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.set_defaults(preprocess=['linear3d_0.00392_0'])  # Normalization option
        parser.set_defaults(preprocess=[])  # Normalization option
        return parser

    def __init__(self, opt):
        """ Initialize this dataset class.

        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(self)
        self.opt = opt
        self.len = len(os.listdir(self.opt.data_dirname))

    def __getitem__(self, index):
        """ Return a data dict and its metadata information.

        :param index: an integer for data indexing
        :return: a dictionary of data with their names, including video, n_first_frame, n_last_frame.
        """
        # Load info.json to get n_first_frame and n_last_frame
        info_path = os.path.join(self.opt.data_dirname, str(index), 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
            if 'n_first_frame' not in info or 'n_last_frame' not in info:
                raise ValueError(f"Missing 'n_first_frame' or 'n_last_frame' in {info_path}")
            n_first_frame = info['n_first_frame']
            n_last_frame = info['n_last_frame']

        # Load video (4D)
        video_path = os.path.join(self.opt.data_dirname, str(index), 'video.nii.gz')
        video, *_ = self.load_nifti(video_path)  # Should return T×D×H×W
        video = video.unsqueeze(1)  # Should return T×1×D×H×W

        processed_frames = []
        pair = BaseDataset.new_transform_pair(self.opt)
        for t in range(video.shape[0]):
            frame = video[t, ...]
            frame_transformed = pair.apply_image(frame)
            processed_frames.append(frame_transformed)
        video_transformed = torch.stack(processed_frames, dim=0)  # Stack back to T×1×D×H×W

        first_frame = video_transformed[n_first_frame]
        last_frame = video_transformed[n_last_frame]

        # Return the data in a dictionary
        return {
            'video': video_transformed,  # n×t×1×d×h×w
            'n_first_frame': n_first_frame,
            'n_last_frame': n_last_frame,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'video_path': video_path
        }

    def get_prefix(self, index):
        """ Get the prefix path for the video file based on the index """
        return os.path.join(self.opt.data_dirname, str(index))

    def __len__(self):
        """ Return the total number of video samples in the dataset. """
        return self.len
