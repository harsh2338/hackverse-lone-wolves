import cv2
import numpy as np
import dlib
import math
import constants
from point import Point

class Eye():
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        _, self.frame = self.capture.read()
        self.gray_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.detector(self.gray_img)
        self.keyboard = np.zeros((600, 1000, 3), np.uint8)
        self.keyboard.fill(255)
        self.counter=0
        self.algo()

    def get_eye_dimensions(self,extremes,top,bottom,landmarks):

        left_point = Point(landmarks.part(extremes[0]).x,
                           landmarks.part(extremes[0]).y)
        right_point = Point(landmarks.part(extremes[1]).x,
                            landmarks.part(extremes[1]).y)

        center_top = self.get_mid_point(landmarks.part(top[0]), landmarks.part(top[1]))
        center_bottom = self.get_mid_point(landmarks.part(bottom[0]),
                                      landmarks.part(bottom[1]))

        hor_line = cv2.line(self.frame, (left_point.x, left_point.y), (right_point.x, right_point.y), (0, 255, 0), 2)
        ver_line = cv2.line(self.frame, (center_top.x, center_top.y), (center_bottom.x, center_bottom.y), (0, 255, 0), 2)

        eye_height = self.get_distance(center_bottom, center_top)
        eye_length = self.get_distance(left_point, right_point)

        return eye_length,eye_height
    def is_blinking(self,landmarks):
        left_len,left_ht=self.get_eye_dimensions(constants.LEFT_EYE_HORIZONTAL_EXTREMES,constants.LEFT_EYE_TOP,constants.LEFT_EYE_BOTTOM,landmarks)
        right_len,right_ht=self.get_eye_dimensions(constants.RIGHT_EYE_HORIZONTAL_EXTREMES,constants.RIGHT_EYE_TOP,constants.RIGHT_EYE_BOTTOM,landmarks)
        return (left_len/left_ht>4) and (right_len/right_ht>4)
    def get_mid_point(self,p1,p2):
        return Point(int((p1.x+p2.x)/2),int((p1.y+p2.y)/2))
    def get_distance(self,p1,p2):
        return math.sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2))
    def extract_eye(self,left_eye_region):
        height, width, _ = self.frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(self.gray_img, self.gray_img, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY)

        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        # cv2.imshow("Eye", eye)
        cv2.imshow("Threshold", threshold_eye)
        # cv2.imshow("Left eye", left_eye)

        threshold_eye_row_num, threshold_eye_col_num = threshold_eye.shape
        left_half = threshold_eye[0:threshold_eye_row_num, 0:int(threshold_eye_col_num / 2)]
        right_half = threshold_eye[0:threshold_eye_row_num, int(threshold_eye_col_num / 2):threshold_eye_col_num]

        left_white = max(1, cv2.countNonZero(left_half))
        right_white = max(1, cv2.countNonZero(right_half))

        wb_ratio=left_white / right_white
        return  wb_ratio
    def get_gaze_direction(self, landmarks):
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                     (landmarks.part(43).x, landmarks.part(43).y),
                                     (landmarks.part(44).x, landmarks.part(44).y),
                                     (landmarks.part(45).x, landmarks.part(45).y),
                                     (landmarks.part(46).x, landmarks.part(46).y),
                                     (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
        left_ratio= self.extract_eye(left_eye_region)
        right_ratio= self.extract_eye(right_eye_region)

        if(left_ratio<1 and right_ratio<1):
            cv2.putText(self.frame, 'Right', (50, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=3,
                        fontScale=3)
        elif(left_ratio>2.4 and right_ratio>2.4):
            cv2.putText(self.frame, 'Left', (50, 150), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0), thickness=3,
                        fontScale=3)
        else:
            cv2.putText(self.frame, 'Centre', (50, 150), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0), thickness=3,
                        fontScale=3)

    def letter(self,text, x, y,is_highlighted):
        width = height = 200
        th = 3
        if(is_highlighted):
            cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
        else:
            cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (0, 0, 0), th)
        font_letter = cv2.FONT_HERSHEY_PLAIN
        font_scale = 10
        font_th = 4
        text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
        width_text, height_text = text_size[0], text_size[1]
        text_x = int((width - width_text) / 2) + x
        text_y = int((height + height_text) / 2) + y
        cv2.putText(self.keyboard, text, (text_x, text_y), font_letter, font_scale, (0, 0, 0), font_th)

    def show_left_keys(self,highlight_index):

        index=0
        ran=None
        for i in range(0,600,200):
            if(i==400):
                ran=range(200,800,200)
            else :
                ran=range(0,1000,200)
            for j in ran:
                self.letter(constants.LEFT_LETTERS[index],j,i,highlight_index==index)
                index+=1

    def algo(self):

        highlight_index=0
        blinking_counter=0
        while True:
            _, self.frame = self.capture.read()
            self.counter+=1

            if(self.counter%constants.FPS==0):
                highlight_index+=1
                highlight_index=highlight_index%13
                self.counter=0
            self.gray_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            self.faces = self.detector(self.gray_img)

            for face in self.faces:
                landmarks = self.predictor(self.gray_img, face)
                self.get_gaze_direction(landmarks)

                if (self.is_blinking(landmarks)):
                    cv2.putText(self.frame, 'Closed', (50, 150), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0), thickness=3,
                                fontScale=3)
                    if(blinking_counter==constants.FPS):
                        print(constants.LEFT_LETTERS[highlight_index])
                    blinking_counter+=1
                    self.counter-=1

                else:
                    blinking_counter=0


            cv2.imshow("Frame", self.frame)
            cv2.imshow("Keyboard", self.keyboard)
            self.show_left_keys(highlight_index)
            key = cv2.waitKey(1)
            if key == 27:
                break
        self.capture.release()
        cv2.destroyAllWindows()

def main():
    Eye()


main()

