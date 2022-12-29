import cv2
import numpy as np
import utils
import json

with open("saves.json","r") as f:
    save_data = json.load(f)
    f.close()

#Values
path = "Images/paper1.jpg"
widthImg = 800
heightImg = 500
grade_box_width = 800
questions = 5
choices = 4
answers = [1,2,1,3,1]
webcamFeed = True
cameraNo = 0

cap = cv2.VideoCapture(cameraNo)

#Load and process image
while True:
    if webcamFeed:
        success,img = cap.read()
    else:
        img = cv2.imread(path)
    
    img = cv2.resize(img,(widthImg,heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrey,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)

    try:
        #Finding all contours
        contours,hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours,-1,(0,255,0),5)

        #Find rectangles
        rect_contours = utils.rectContour(contours)
        biggest_contour = utils.getCornerPoints(rect_contours[0])
        grade_box_points = utils.getCornerPoints(rect_contours[1])

        if len(biggest_contour) != 0 and len(grade_box_points) != 0:
            cv2.drawContours(imgBiggestContours,biggest_contour,-1,(0,255,0),15)
            cv2.drawContours(imgBiggestContours,grade_box_points,-1,(0,0,255),15)

            biggest_contour = utils.reorder(biggest_contour)
            grade_box_points = utils.reorder(grade_box_points)

            pt1 = np.float32(biggest_contour)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(src=pt1,dst=pt2)
            imgWarpColoured = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

            gradept1 = np.float32(grade_box_points)
            gradept2 = np.float32([[0,0],[grade_box_width,0],[0,heightImg],[grade_box_width,heightImg]])
            grade_matrix = cv2.getPerspectiveTransform(src=gradept1,dst=gradept2)
            gradeWarpColoured = cv2.warpPerspective(img,grade_matrix,(grade_box_width,heightImg))

            #Apply threshold
            imgWarpGrey = cv2.cvtColor(imgWarpColoured,cv2.COLOR_BGR2GRAY)
            img_threshold = cv2.threshold(imgWarpGrey,175,255,cv2.THRESH_BINARY_INV)[1]

            boxes = utils.splitBoxes(img_threshold,choices,questions)

            #Get pixel values
            pixel_values = np.zeros((questions,choices))
            column_count = 0
            row_count = 0

            for box in boxes:
                total_pixels = cv2.countNonZero(box)
                pixel_values[row_count][column_count] = total_pixels
                column_count += 1
                if column_count == choices:
                    row_count += 1
                    column_count = 0
            
            #Get the indicies of all the marked options
            marked_options = []
            for row in pixel_values:
                index = np.where(row==max(row))[0][0]
                marked_options.append(index)

            #Grading the paper
            grading = []
            for q in range(0,questions):
                if answers[q] == marked_options[q]:
                    grading.append("correct")
                else:
                    grading.append("incorrect")
            
            correct_answers = grading.count("correct")
            percentage = correct_answers/questions * 100

            #Displaying answers
            imgResults = imgWarpColoured.copy()
            imgResults = utils.showAnswers(imgResults,marked_options,grading,answers,questions,choices)
            imgBlankResults = np.zeros_like(imgWarpColoured)
            imgBlankResults = utils.showAnswers(imgBlankResults,marked_options,grading,answers,questions,choices)
            
            inverse_matrix = cv2.getPerspectiveTransform(src=pt2,dst=pt1)
            imgInverseWarp = cv2.warpPerspective(imgBlankResults,inverse_matrix,(widthImg,heightImg))
            
            #Display grade percentage
            imgRawGrade = np.zeros_like(gradeWarpColoured)
            cv2.putText(imgRawGrade,str(int(percentage))+"%",(50,275),cv2.FONT_HERSHEY_COMPLEX,10,(0,255,255),10)
            inverse_matrix_grade = cv2.getPerspectiveTransform(src=gradept2,dst=gradept1)
            imgInverseWarpGrade = cv2.warpPerspective(imgRawGrade,inverse_matrix_grade,(widthImg,heightImg))

            imgFinal = img.copy()
            imgFinal = cv2.addWeighted(imgFinal,1,imgInverseWarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal,1,imgInverseWarpGrade,1,0)

        #Display images
        imgBlank = np.zeros_like(img)
        stack_row1 = [img,imgGrey,imgBlur,imgContours]
        stack_row2 = [imgCanny,imgBiggestContours,imgWarpColoured,gradeWarpColoured]
        stack_row3 = [img_threshold,imgResults,imgBlankResults,imgInverseWarp]
        imgStacked = utils.stackImages(0.3,(stack_row1,stack_row2,stack_row3))
        cv2.imshow("Images", imgStacked)
        cv2.imshow("Final Result",imgFinal)
    except:
        print("No rectangles found")
    
    if cv2.waitKey(0) & 0xFF == ord("s"):
        save_num = save_data["number_saved"]
        cv2.imwrite(f"Results/final{save_num}.jpg",imgFinal)
        save_data["number_saved"] = int(save_data["number_saved"]) + 1
    if cv2.waitKey(0) & 0xFF == ord("q"):       
        break

with open("saves.json","w") as f:
    json.dump(save_data,f)
    f.close()

cap.release()
cv2.destroyAllWindows()