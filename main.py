import cv2
import numpy as np
           
cap = cv2.VideoCapture(0)
pts1 = np.float32([[90,254],[550,254],[0,412],[640,412]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])   
M = cv2.getPerspectiveTransform(pts1, pts2)

while True:
    ret, img = cap.read()
    #print(img.shape[0],img.shape[1])
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out_main = cv2.VideoWriter('camera.avi', fourcc, 20.0, (640,480))
    #out_warp = cv2.VideoWriter('warp.avi', fourcc, 20.0, (300,300))
    #hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, img2 = cv2.threshold(img2, 100, 255, 0)
    #img2 = cv2.bitwise_not(img2)
    img_rec = np.copy(img)
    cv2.rectangle(img_rec,(0,0),(img.shape[1],int(img.shape[0]/2.5)),(255,255,255),cv2.FILLED)
    road = cv2.inRange(img_rec, (0,0,0),(120,120,120))
    #gaussian2 = cv2.medianBlur(img, 11)
    #img_canny = cv2.Canny(gaussian2, 10, 100)
    contours, hierarchy = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #
    cv2.drawContours(img, contours, -1, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
    cv2.rectangle(road,(0,0),(img.shape[1],int(4*img.shape[0]/5)),(0,0,0),cv2.FILLED)
    #road_canny = cv2.Canny(road, 10,100)
    
    warp = cv2.warpPerspective(img,M,(300,300))   
    cv2.line(img,(img.shape[1]-90,int(img.shape[0]*0.53)),(90,int(img.shape[0]*0.53)),(0,255,255),1)     
    cv2.line(img,(img.shape[1],int(img.shape[0]*0.86)),(0,int(img.shape[0]*0.86)),(0,255,255),1)
    cv2.line(img,(img.shape[1]-90,int(img.shape[0]*0.53)),(img.shape[1],int(img.shape[0]*0.86)),(0,255,255),1)     
    cv2.line(img,(90,int(img.shape[0]*0.53)),(0,int(img.shape[0]*0.86)),(0,255,255),1) 
    warp_line = cv2.Canny(warp, 10, 100)
    
    edges = cv2.Canny(warp,50,150, apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,50)
    if lines is not None:
        for r, theta in lines[0]:
            a = np.cos(theta)
            #print(r)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*a)
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*a)
            cv2.line(warp,(x1,y1),(x2,y2), (0,0,255), 2)
            if (theta < 0.05) or (theta > 3.09):
                text = 'OK'
                color = (0,255,0)
            elif (theta > np.pi/2): 
                text = '<- ' +  str(int((np.pi-theta)/np.pi*180)) + ' degrees' 
                color = (0,0,255)
            else:
                color = (0,0,255)
                text = str(int(theta/np.pi*180)) + ' degrees'+ ' ->' 
            cv2.putText(img, text, (250,50),cv2.FONT_HERSHEY_SIMPLEX, 1,color,2)
    else:
         cv2.putText(img, 'Not found', (250,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2) 
    
    cv2.line(warp,(150,0),(150,300),(255,0,0),2)  
    #out_main.write(img)
    #out_warp.write(warp)          
    cv2.imshow("Camera", img) 
    cv2.imshow("Warp",warp)  
 
    if cv2.waitKey(10) == 27:
        break
        
cap.release()
#out_main.release()
#out_warp.release()
cv2.destroyAllWindows()
