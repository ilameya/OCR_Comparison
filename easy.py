import os
import cv2
import easyocr
import PIL
import numpy
import torch
from PIL import ImageDraw
from PIL import Image
im = PIL.Image.open("images.png")

reader = easyocr.Reader(['en'])
#im = cv2.imread("images.png")
bound = reader.readtext("images.png")
bound

def draw_box(Image, bound, color = 'red', width = 2):
    draw = ImageDraw.Draw(Image)
    for bound in bound:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill = color, width= width)
    return Image    
draw_box(im, bound)

#print(type(im))
image = numpy.array(im)
cv2.imwrite("C:/Users/HP/OCR/OCR/EasyOCR/img4.png", image)
#print(type(im))
#cv2.imwrite("img2.jpeg", im)
#image = Image.open(im)
#image.save("img2.jpeg", "JPEG", quality=80, optimize=True, progressive=True)
