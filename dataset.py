import json
import cv2
import os
import numpy as np
from constants import DATA_DIR, AGNOSTIC, EMPTY, DATA_DIR, STAGED, CAPTION_STAGED, CAPTION_EMPTY
from torch.utils.data import Dataset
import sys
sys.path.append('./ControlNet')


class PairedDataset(Dataset):
    """Super basic dataset with no augmentations."""

    def __init__(self, source_key=AGNOSTIC, target_key=STAGED, caption_key=CAPTION_STAGED):
        # These keys define the task we want to train.
        self.source_key = source_key
        self.target_key = target_key
        self.caption_key = caption_key
        self.data = []
        with open(os.path.join(DATA_DIR, "paired_train_datalist.json"), 'rb') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item[self.source_key]
        target_filename = item[self.target_key]
        prompt = item[self.caption_key]

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        target = cv2.resize(
            target, (512, 512), interpolation=cv2.INTER_LINEAR)
        source = cv2.resize(
            source, (512, 512), interpolation=cv2.INTER_LINEAR)

        assert source.shape == target.shape, str(idx)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
