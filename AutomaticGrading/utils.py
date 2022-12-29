import cv2
import numpy as np
import utils

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def rectContour(contours): 
    rect_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 50:
            perimeter = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.02*perimeter,True)
            if len(approx) == 4:
                rect_contours.append(contour)
    
    rect_contours = sorted(rect_contours, key=cv2.contourArea, reverse=True)
    return rect_contours

def getCornerPoints(contour):
    perimeter = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.02*perimeter,True)

    return approx

def reorder(points):
    points = points.reshape((4,2))
    points_new = np.zeros((4,1,2),np.int32)
    add = points.sum(1)

    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    difference = np.diff(points,axis=1)
    points_new[1] = points[np.argmin(difference)]
    points_new[2] = points[np.argmax(difference)]

    return points_new

def splitBoxes(img,choices,questions):
    rows = np.vsplit(img,questions)
    boxes = []

    for row in rows:
        cols = np.hsplit(row,choices)

        for box in cols:
            boxes.append(box)
    
    return boxes

def showAnswers(img,marked_indicies,grading,answers,questions,choices):
    section_width = int(img.shape[1]/choices)
    section_height = int(img.shape[0]/questions)

    for q in range(0,questions):
        marked_answer = marked_indicies[q]
        center_x = int((marked_answer*section_width)+section_width/2)
        center_y = int((q*section_height)+section_height/2)
        colour = (0,0,0)

        if grading[q] == "correct":
            colour = (0,255,0)
        else:
            colour = (0,0,255)
            correct_answer = answers[q]
            answer_x = int((correct_answer*section_width)+section_width/2)
            answer_y = int((q*section_height)+section_height/2)
            cv2.circle(img,(answer_x,answer_y),40,(0,255,0),15)
        
        cv2.circle(img,((center_x),center_y),50,colour,cv2.FILLED)
    
    return img
