import os
import glob
import cv2
import pytesseract
from pytesseract import Output
import easyocr
import PIL
import numpy
import torch
from PIL import ImageDraw
from PIL import Image
from pathlib import Path
import time
from tqdm import tqdm
import pandas as pd

# user filepaths settings. 
# current users, nijhum and lameya. 
user = "nijhum"

if(user == "lameya"):
    path = "C:/Users/HP/OCR/OCR/Tesseract/*.jpg"
    tesseract_files_path = "C:/Users/HP/OCR/OCR/Tesseract"
    easyocr_files_path = "C:/Users/HP/OCR/OCR/EasyOCR"
elif(user == "nijhum"): 
    path = "C:/Users/Asus/JP/OCRcomparison/Input/*.*"
    tesseract_files_path = "C:/Users/Asus/JP/OCRcomparison/Tesseract"
    easyocr_files_path = "C:/Users/Asus/JP/OCRcomparison/EasyOCR"
    # Tesseract executable path    
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# apply tesseract OCR and return time duration for each image
def Tess(file):

    start = time.time()
    img = cv2.imread(file, 0)
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    for i, word in enumerate(data['text']):
        if word!= "":
            x,y,w,h = data['left'][i],data['top'][i],data['width'][i],data['height'][i]
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    image_name = Path(file).name
    f = os.path.join(tesseract_files_path, image_name)
    cv2.imwrite(f, img)
    
    end = time.time()
    duration = end - start
    return duration

# apply easyocr OCR and return time duration for each image
def EasyOCR(file):
    
    start = time.time()
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
    f = os.path.join(easyocr_files_path, image_name)
    cv2.imwrite(f, image)

    end = time.time()
    duration = end - start
    return duration

# main function
def main():
    # lists to store time duration of performing ocr on each image
    tess_durations = []
    easy_durations = []

    # iterate and perform ocr on all the files in the input directory. 
    # append the time duration to respective list
    for file in tqdm(glob.glob(path), desc="Processing"):
        tess_duration = Tess(file)
        easyocr_duration = EasyOCR(file)
        tess_durations.append(tess_duration)
        easy_durations.append(easyocr_duration)

    # save results to dataframe
    results = [["Total Duration: ", sum(tess_durations), sum(easy_durations)],
                ["Average Duration: ", sum(tess_durations)/len(tess_durations), sum(easy_durations)/len(easy_durations)], 
                ["Max Duration: ", max(tess_durations), max(easy_durations)], 
                ["Min Duration: ", min(tess_durations), min(easy_durations)]]

    df = pd.DataFrame(results,columns=['Calculation','Tesseract','EasyOCR'])
    print(df)

    # save dataframe to csv file
    df.to_csv('results.csv', index=False, header=True)

if __name__ == '__main__':
    main()  
