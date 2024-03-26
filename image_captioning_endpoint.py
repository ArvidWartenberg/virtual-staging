# Image captioning endpoint
import torch

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


class ImageCaptioner:
    """Class that captions images using off-the-shelf model."""
    # Model source https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def caption_image(self, image):
        """Caption image."""
        pixel_values = self.feature_extractor(
            images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values)

        preds = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]
