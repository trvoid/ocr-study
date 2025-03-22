################################################################################
# TrOCR Test
################################################################################

import os, sys, traceback
import fire
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from image_processor import ImageProcessor
import cv2

def main(filename, debug=False):
    image = ImageProcessor.load(filename)
    image = ImageProcessor.gray(image)

    model_id = 'microsoft/trocr-base-handwritten'
    #model_id = 'microsoft/trocr-large-handwritten'

    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    image = Image.fromarray(image).convert('RGB')

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f'Result: *{text.strip()}*')

    if debug:
        cv2.imshow('Preprocessed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        fire.Fire(main)
    except:
        traceback.print_exc(sys.stdout)
