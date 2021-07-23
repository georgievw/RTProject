import cv2
from numpy import cos, sin, pi, float32, copy

def DrawLines(img): #возвращает изображение с выделенной разметкой дороги на img
    img_rec = copy(img)
    cv2.rectangle(img_rec,(0,0),(img.shape[1],int(img.shape[0]/2.5)),(255,255,255),cv2.FILLED)
    road = cv2.inRange(img_rec, (0,0,0),(120,120,120))
    contours, hierarchy = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(img, contours, -1, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
    return img
               
def CheckDirection(img, warp): #определяет угол между текущим и необходимым векторами движения, возвращает изображения с необходимыми пометками
    edges = cv2.Canny(warp,50,150, apertureSize = 3) #выделение границ на изображении 
    lines = cv2.HoughLines(edges,1,pi/180,50) #поиск прямых на изображении
    if lines is not None:
        for r, theta in lines[0]: #построение на изображении одной прямой, соответствующей необходимому направлению движения (одной - из расчета, что линии разметки  - параллельные прямые)
            a = cos(theta) #определение координат точек для построения прямой
            b = sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*a)
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*a)
            cv2.line(warp,(x1,y1),(x2,y2), (0,0,255), 2) #построение прямой
            if (theta < 0.05) or (theta > 3.09): #анализ отклонения от необходимого вектора движения
                text = 'OK'
                color = (0,255,0)
            elif (theta > pi/2): 
                text = '<- ' +  str(int((pi-theta)/pi*180)) + ' degrees' 
                color = (0,0,255)
            else:
                color = (0,0,255)
                text = str(int(theta/pi*180)) + ' degrees'+ ' ->' 
            cv2.putText(img, text, (250,50),cv2.FONT_HERSHEY_SIMPLEX, 1,color,2)
    else:
         cv2.putText(img, 'Not found', (250,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2) 
    cv2.line(warp,(150,0),(150,300),(255,0,0),2)
    return img, warp
                   
cap = cv2.VideoCapture(0)
pts1 = float32([[90,254],[550,254],[0,412],[640,412]])
pts2 = float32([[0,0],[300,0],[0,300],[300,300]])   
M = cv2.getPerspectiveTransform(pts1, pts2)  #определение матрицы преобразования для квадратной области перед машинкой
while True:
    ret, img = cap.read()
    img = DrawLines(img)   
 
    warp = cv2.warpPerspective(img,M,(300,300))
       
    cv2.line(img,(img.shape[1]-90,int(img.shape[0]*0.53)),(90,int(img.shape[0]*0.53)),(0,255,255),1) #выделение области перед машинкой на главном изображении
    cv2.line(img,(img.shape[1],int(img.shape[0]*0.86)),(0,int(img.shape[0]*0.86)),(0,255,255),1)
    cv2.line(img,(img.shape[1]-90,int(img.shape[0]*0.53)),(img.shape[1],int(img.shape[0]*0.86)),(0,255,255),1)     
    cv2.line(img,(90,int(img.shape[0]*0.53)),(0,int(img.shape[0]*0.86)),(0,255,255),1) 
  
    img, warp = CheckDirection(img, warp)
              
    cv2.imshow("Camera", img) 
    cv2.imshow("Warp",warp)  
 
    if cv2.waitKey(10) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
