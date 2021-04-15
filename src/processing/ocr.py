"""
OCR Class
@author r.kippers, 2021
"""

import numpy as np
import pytesseract
import cv2

class TesseractOCR():
    
    def __init__(self):
        """
        Class for extracting labels from floor plan 
        images using OpenCV
        """
        self.confidence_th = 80 
        self.kernel_size = (15,15)
    
    def extract_labels(self, image):
        """
        Extract labels from image
        
        Parameters
        ----------
        image : ndarray
            Numpy array with image 
        
        Output
        -----
        tuple
            ("label tekst", (x_min, y_min, x_max, y_max))
        """
        
        result = []
        
        # Change format 
        image = image.copy()
        image = np.uint8(image)
        
        # create mask and convert to grayscale / binary
        mask = np.zeros(image.shape, dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray,140,80,cv2.THRESH_BINARY_INV) 
        
        # Get contours and draw on mask
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w < 120 and h < 100:
                cv2.drawContours(mask, [cnt], 0, (255,255,255), 1)

        # Morphological operations (dilation) and find contours
        kernel = np.ones(self.kernel_size,np.uint8)
        dilation = cv2.dilate(mask,kernel,iterations = 1)
        gray_d = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
        _, threshold_d = cv2.threshold(gray_d,150,255,cv2.THRESH_BINARY)
        contours_d, hierarchy = cv2.findContours(threshold_d,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        for cnt in contours_d:
            x,y,w,h = cv2.boundingRect(cnt)
            if w < image.shape[1]/2 and h < image.shape[0]/2: #True: #if w > 35:
                roi_c = image[y:y+h, x:x+w]
                ocr_data = pytesseract.image_to_data(roi_c, output_type=pytesseract.Output.DICT,  config=f'--psm 11') #psm = page segmentation mode

                text = ""
                for i in range(len(ocr_data['text'])):
                    if int(ocr_data['conf'][i]) > self.confidence_th:
                        text += ocr_data['text'][i]
                
                if len(text) > 0:
                    result.append((text, (x,y,x+w, y+h)))
        
        return result