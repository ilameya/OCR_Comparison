import pytesseract
from pytesseract import Output
import cv2

img = cv2.imread("images.png")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
data = pytesseract.image_to_data(img, output_type=Output.DICT)

print(data)
print(data['left'])
for i, word in enumerate(data['text']):

    if word!= "":

        x,y,w,h = data['left'][i],data['top'][i],data['width'][i],data['height'][i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
        #cv2.putText(img, word, (x,y-16), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

  
print(type(img))
cv2.imwrite("C:/Users/HP/OCR/OCR/Tesseract/img1.png", img)