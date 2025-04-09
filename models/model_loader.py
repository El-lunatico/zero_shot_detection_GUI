from transformers import OwlViTProcessor, OwlViTForObjectDetection

def load_owlvit_model():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    return processor, model
