################################################################################
# Tesseract Test
################################################################################

import os, sys, traceback
import fire
import pytesseract
from image_processor import ImageProcessor
import cv2

def main(filename, debug=False):
    image = ImageProcessor.load(filename)
    image = ImageProcessor.gray(image)

    text = pytesseract.image_to_string(image, lang='eng')
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
