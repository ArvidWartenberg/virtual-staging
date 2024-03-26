# OwlSam endpoint
from transformers import pipeline, SamModel, SamProcessor
import torch
import numpy as np
from PIL import Image
import cv2

from constants import OWLSAM_OBJECT_CLASSES_STR


class OwlSamSegmentor:
    """Class that uses OwlSam for segmentation using open vocabulary class prompts."""
    # Code is borrowed from: https://huggingface.co/spaces/merve/OWLSAM

    def __init__(self):
        checkpoint = "google/owlvit-base-patch16"
        self.detector = pipeline(
            model=checkpoint, task="zero-shot-object-detection")
        self.sam_model = SamModel.from_pretrained(
            "facebook/sam-vit-base").to("cuda")
        self.sam_processor = SamProcessor.from_pretrained(
            "facebook/sam-vit-base")

    def query(self, image, texts, threshold=0.05):
        """Use OwlSam to retrieve object instances from image based on prompted classes."""
        # Convert input image array to PILLOW image
        image = Image.fromarray(image)
        texts = texts.split(",")
        predictions = self.detector(
            image,
            candidate_labels=texts,
            threshold=threshold
        )

        result_labels = []
        for pred in predictions:

            box = pred["box"]
            score = pred["score"]
            label = pred["label"]
            box = [round(pred["box"]["xmin"], 2), round(pred["box"]["ymin"], 2),
                   round(pred["box"]["xmax"], 2), round(pred["box"]["ymax"], 2)]

            inputs = self.sam_processor(
                image,
                input_boxes=[[[box]]],
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            mask = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0][0][0].numpy()
            mask = mask[np.newaxis, ...]
            result_labels.append((mask, label))

        return result_labels

    def get_staged_objects_mask(self, image):
        """Simple endpoint that makes a compound binary mask from all different instances."""
        instances = self.query(image, texts=OWLSAM_OBJECT_CLASSES_STR)
        compound_staged_objects_mask = np.zeros(image.shape[0:2])
        for instance in instances:
            compound_staged_objects_mask += instance[0][0].astype(np.uint8)

        compound_staged_objects_mask = (
            compound_staged_objects_mask > 0.5).astype(np.uint8)

        # We dilate the mask a little bit as there will be some artifacts on boundaries between instances otherwise
        # TODO: 3 iterations is arbitrary here. Should investigate choices.
        dilated_mask = cv2.dilate(
            compound_staged_objects_mask.astype(np.uint8), None, iterations=3)

        return dilated_mask*255


if __name__ == "__main__":
    # This code is not optimized for the use-case and runs very slow.
    # As a PoC solution we don't consider individual semantic classes but just use an overall
    # binary mask to indicate which regions of the image correspond to fournishing.
    # The classes here were chosen ad-hoc based on a few samples I inspected and are not production ready.
    # This is a big area for improvment.
    owlsam = OwlSamSegmentor()
    image = Image.open("/workspace/data/500_empty_staged_384px/3_staged.png")
    staged_objects_mask = owlsam.get_staged_objects_mask(image)
    print()
