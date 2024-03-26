# Script to prepare dataset
import os
import numpy as np
import cv2
import json
import random

from tqdm import tqdm

from image_captioning_endpoint import ImageCaptioner
from constants import AGNOSTIC, MASK, DATA_DIR, PAIRED_DATA_DIR, STAGED, EMPTY, PNG, CAPTION_EMPTY, CAPTION_STAGED


def _calculate_diff_mask(staged, empty, threshold=0.05):
    # Simple algorithm for getting the difference mask.
    # We take the difference of the grayscale staged/empty images,
    # and then apply a threshold (default 0.05 abs diff in grayscale space).
    # The returned diff mask is of shape HxW with ones indicating change.
    gray_staged = cv2.cvtColor(
        staged, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gray_empty = cv2.cvtColor(
        empty, cv2.COLOR_BGR2GRAY).astype(np.float32)/255

    # Find contours to fill holes
    diff_mask = (np.abs(gray_empty-gray_staged) > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(
        diff_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(diff_mask, [cnt], 0, 255, -1)
    return diff_mask


def prepare_paired_data():
    """Prepare paired training data and datalist."""
    # Could be done more gracefully with checks but just want to get the data.
    # So we just assume that there is nothing funky with the data we unzipped and just assume all pairs 0,1,...,501 exist
    num_pairs = 501

    # Initialize image captioner
    captioner = ImageCaptioner()

    # Prepare the datalist
    datalist = []

    # Here we go over all the pairs and calculate the difference masks.
    # Using the diff masks we create the agnostic scene representation, which
    # corresponds to the scene where all fournishing is masked with grey.
    # The masks and agnostic images are dumped to disk.
    for pair_ix in tqdm(range(num_pairs)):
        # Retrieve data for this pair ix
        staged_image_path = os.path.join(
            PAIRED_DATA_DIR, f"{pair_ix}_{STAGED}.{PNG}")
        empty_image_path = os.path.join(
            PAIRED_DATA_DIR, f"{pair_ix}_{EMPTY}.{PNG}")

        staged = cv2.imread(staged_image_path)
        empty = cv2.imread(empty_image_path)

        # Just quick fix.
        # Noticed some pairs have 1 px height diff.
        # Fixing it here and dumping to disk.
        if empty.shape != staged.shape:
            h, w, _ = empty.shape
            staged = cv2.resize(
                staged, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(staged_image_path, staged)
            staged = cv2.imread(staged_image_path)

        # and then fill holes in the resulting mask. Filling
        diff_mask = _calculate_diff_mask(staged, empty, threshold=0.05)

        # We also dump the "agnostic" scene representation to disk.
        # In this rep, the diff-mask region we want to manipulate with the
        # ControlNet is filled with black.
        # We specifically choose to copy the empty room room here such that
        # the trained model can learn to paint shadows even outside the
        # masked area.
        agnostic = np.copy(empty)
        agnostic[diff_mask == 255] = [0, 0, 0]

        agnostic_image_path = os.path.join(
            PAIRED_DATA_DIR, f"{pair_ix}_{AGNOSTIC}.{PNG}")
        mask_image_path = os.path.join(
            PAIRED_DATA_DIR, f"{pair_ix}_{MASK}.{PNG}")

        # Write data to disk
        cv2.imwrite(agnostic_image_path, agnostic)
        cv2.imwrite(mask_image_path, diff_mask)

        # Get image captions
        caption_staged = captioner.caption_image(staged)
        caption_empty = captioner.caption_image(empty)

        # Add this element to the datalist
        datalist.append(
            {EMPTY: empty_image_path, STAGED: staged_image_path, AGNOSTIC: agnostic_image_path, MASK: mask_image_path, CAPTION_EMPTY: caption_empty, CAPTION_STAGED: caption_staged})

    # In a desparate attempt to get something to actually work I only hold out
    # 10 samples for eval.
    random.shuffle(datalist)
    eval_datalist, train_datalist = datalist[0:10], datalist[10:]

    # Dump datalists to disk
    with open(os.path.join(DATA_DIR, 'paired_eval_datalist.json'), 'w') as fp:
        json.dump(eval_datalist, fp)
    with open(os.path.join(DATA_DIR, 'paired_train_datalist.json'), 'w') as fp:
        json.dump(train_datalist, fp)


if __name__ == "__main__":
    prepare_paired_data()
