import cv2

class ImageProcessor:
    @staticmethod
    def load(filename):
        return cv2.imread(filename)
    
    @staticmethod
    def gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
