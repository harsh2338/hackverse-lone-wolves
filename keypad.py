import cv2
import numpy as np

def letter(text,x,y):
    width = height = 200
    th = 3
    cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (0, 0, 0), th)
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (0, 0, 0), font_th)


keyboard = np.zeros((1000, 1500, 3), np.uint8)
keyboard.fill(255)

letter('A',0,0)
letter('B',200,0)
letter('C',400,0)
letter('D',600,0)
letter('E',800,0)


letter('F',0,200)
letter('G',200,200)
letter('H',400,200)
letter('I',600,200)
letter('J',800,200)

letter('K',200,400)
letter('L',400,400)
letter('M',600,400)




cv2.imshow('Keyboard',keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()