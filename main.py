from imutils import perspective
import numpy as np
import imutils
import cv2
import math
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)
path = 'many.png'

#for test blob detection
#reduce the iterations of dilation

webcam = False
minarea = 2000
maxarea = 20000

#Euclidean calculation function of 2 points by pixel
def distance(x1,y1 , x2 , y2):
    return (((x1 - x2) **2 + (y1 - y2) **2))**0.5
#Function to calculate the rotation angle of the object relative to the image
def getAngle(points):
    b,c,a = points[-3:]
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = round(ang + 360 if ang < 0 else ang)
    
    rotation = ang - 360 
    print("angle :" ,ang )
    print("rotation :" ,rotation )
    cv2.circle(img, (c[0], c[1]), 10, (255,0,0), -1)
    cv2.circle(img, (a[0], a[1]), 10, (0,255,0), -1)
    cv2.circle(img, (b[0], b[1]), 10, (0,0,255), -1)
    
    pts = np.array( [c, b, a], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], False, (0,0,255), 3)

    cv2.putText(img, 'angle :{}'.format(ang), (b[0]-60,b[1]-60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 1)
    cv2.putText(img, 'rotation :{}'.format(rotation), (b[0]-10,b[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,
                            (0, 0, 255), 1)

#The function to sort the points in the following order:
#top left
#top right
#lower left
#right down
def reorder(myPoints): 
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

ret, img = cap.read()

# blob_image = None
while True:
    if webcam:
        
        ret, img = cap.read()
        orig = img
        blob_image = img.copy()

    else:
        img = cv2.imread(path)
        orig = img
        blob_image = img.copy()
        

    #Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #GaussianBlur
    blur = cv2.GaussianBlur(gray, (15, 15),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    imgCanny = cv2.Canny(thresh,50, 50)
    kernel = np.ones((3,3),np.uint8)
    
    dilation = cv2.morphologyEx(imgCanny,cv2.MORPH_CLOSE,kernel,iterations=3)
    # dilation = cv2.erode(imgCanny, kernel, iterations = 3)
    dilation = cv2.dilate(imgCanny, kernel, iterations = 3)

    # distanceTransform
    
    # dilation = cv2.distanceTransform(dilation, cv2.DIST_L2, 0)
    # dilation = cv2.convertScaleAbs(dilation)

 
    result_img = dilation.copy()
    contours,hierachy = cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    NumCon = 0

    print(img.shape)

    for cnt in contours:

        area = cv2.contourArea(cnt)
    
        if minarea > area and area < maxarea:
            continue

        box = cv2.minAreaRect(cnt)
        cv2.minEnclosingTriangle
        box_point = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box_point, dtype="int")
        

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

        
        nPoints = reorder (box_point) # all four corner point
        #Four corner points of the object        
        x1, y1 = nPoints[0]
        x2, y2 = nPoints[1]
        x3, y3 = nPoints[2]
        x4, y4 = nPoints[3]
        
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        x3 = int(x3)
        y3 = int(y3)
        x4 = int(x4)
        y4 = int(y4)

        cv2.circle(orig, (x1 , y1), 5, (255, 0, 0), -1)
        cv2.circle(orig, (x2 , y2), 5, (255, 0, 0), -1)
        cv2.circle(orig, (x3 , y3), 5, (255, 0, 0), -1)
        cv2.circle(orig, (x4 , y4), 5, (255, 0, 0), -1)

        points = []
        M = cv2.moments(box_point)
        #The center point of each object
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(orig, (cx, cy), 1, (0, 255, 64), -1)

        #Mathematical operations to calculate the angle of rotation
        a = (y4 - y3) / (x4 - x3)
        b = y3 - a * x3

        if abs(a) < 0.01:
            new_x = cx
            new_y = cy + 50
        else:
            new_x = round((cx * (1 - a**2) + 2 * a * (cy - b)) / (1 + a**2))
            new_y = round((2 * (a * cx + b) - cy * (1 - a**2)) / (1 + a**2))

        # Add points to list
        points.append([cx, cy])
        points.append([cx, cy + 100])
        points.append([new_x, new_y])
        print("corner point : (" ,x1 , y1, ")" ,"(" ,x2 , y2, ")" ,"(" ,x3 , y3, ")" ,"(" ,x4 , y4, ")"  )
        # rotation and angle
        getAngle(points)

        rect = cv2.minAreaRect(cnt)
        center = (int(rect[0][0]), int(rect[0][1]))
        print("center :",center)

        cv2.arrowedLine(orig, (x1,y1), (x2,y2),
                        (255, 0, 255), 3, 8, 0, 0.05)
        cv2.arrowedLine(orig, (x1,y1), (x3,y3),
                        (255, 0, 255), 3, 8, 0, 0.05)
        
        #Euclidean distance of the length and width of the object in pixels
        distance_pixel_l = distance(x1 , y1 , x2 , y2)
        distance_pixel_p = distance(x1 , y1 , x3 , y3)
        cv2.putText(orig, "L: {:.1f}pixel".format(distance_pixel_l),(int(x1+200), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,255), 2)
        cv2.putText(orig, "P: {:.1f}pixel".format(distance_pixel_p),(int(x2 ), int(y2+50)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,255), 2)
        NumCon+=1
        
    
    #blob detection
    params = cv2.SimpleBlobDetector_Params()
    # set filter by area
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 25000
    
    detector = cv2.SimpleBlobDetector_create(params)

    # keypoint of blob 
    keypoints = detector.detect(dilation)
    if len(keypoints) != 0 : 
        print(len(keypoints) , "blob detected !")
        for i , k in enumerate (keypoints):
            print( "####### blob: ",i+1,"/" , len(keypoints) , "#######")
            x = round(k.pt[0])
            y = round(k.pt[1])
            diameter = round(k.size)
            radius = round(diameter/2)
            print("Center:", "(" ,x , "," , y , ")")
            print("Radius:",radius)
            cv2.circle(blob_image , (x , y) ,3 , (255 , 0,255 ) , -1)
            cv2.circle(blob_image , (round(x+ radius)  ,y) ,3 , (255 , 0,255 ) , -1)
            
            cv2.arrowedLine(blob_image, ((x,y)), ((x+ radius , y)),
                                (255, 0, 255), 3, 8, 0, 0.05)
            cv2.putText(blob_image, 'Rad: {}pixel'.format(radius), (x, y + 30), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.4, (0, 0, 255), 1)
            cv2.putText(blob_image, 'center : ({},{})'.format(x,y), (x, y-30), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.4, (0, 0, 255), 1)
            blob_image = cv2.drawKeypoints(blob_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(orig, "Detected: {}".format(NumCon),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2, cv2.LINE_AA)  
    cv2.imshow('camera(square detected)',orig)
    cv2.imwrite("out_put.png",orig)
    cv2.imshow('camera(blob detected)',blob_image)
    cv2.imshow("Canny" , imgCanny)
    cv2.imshow("binary" , dilation)
    
    key = cv2.waitKey(1)
    if key == 27:
        
        break