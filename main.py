import os
import glob
import cv2
import pytesseract
from pytesseract import Output
import easyocr
import cv2
import PIL
import numpy
import torch
from PIL import ImageDraw
from PIL import Image
from pathlib import Path

path = "C:/Users/HP/OCR/OCR/Input/*.*"

def Tess(file):

    img = cv2.imread(file, 0)
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    for i, word in enumerate(data['text']):
        if word!= "":
            x,y,w,h = data['left'][i],data['top'][i],data['width'][i],data['height'][i]
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    image_name = Path(file).name
    f = os.path.join("C:/Users/HP/OCR/OCR/Tesseract", image_name)
    print(f)
    cv2.imwrite(f, img)


def EasyOCR(file):

    im = PIL.Image.open(file)
    reader = easyocr.Reader(['en'], gpu=False)
    bound = reader.readtext(file)

    def draw_box(Image, bound, color = 'red', width = 2):
        draw = ImageDraw.Draw(Image)
        for bound in bound:
            p0, p1, p2, p3 = bound[0]
            draw.line([*p0, *p1, *p2, *p3, *p0], fill = color, width= width)
        return Image
        
    draw_box(im, bound)
    image = numpy.array(im)
    image_name = Path(file).name
    print(image_name)
    f = os.path.join("C:/Users/HP/OCR/OCR/EasyOCR", image_name)
    print(f)
    cv2.imwrite(f, image)

def main():
    for file in glob.glob(path):
        Tess(file)
        EasyOCR(file)

if __name__ == '__main__':
    main()  
