import cv2
import numpy as np
from time import sleep

largeur_min=80 #Largeur minimale du retangule
altitude_min=80 #Altitude minimale du retangule

offset=6 #Erreur autorisée entre les pixels  

pos_ligne=550 #Position de la ligne de comptage

delay= 60 #FPS du video

detec = []
cars= 0

	
def captur_centre(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
substract = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    temps = float(1/delay)
    sleep(temps) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = substract.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatation = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatation = cv2.morphologyEx (dilatation, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(dilatation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos_ligne), (1200, pos_ligne), (255,127,0), 3) 
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        valider_contour = (w >= largeur_min) and (h >= altitude_min)
        if not valider_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centre = captur_centre(x, y, w, h)
        detec.append(centre)
        cv2.circle(frame1, centre, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(pos_ligne+offset) and y>(pos_ligne-offset):
                cars+=1
                cv2.line(frame1, (25, pos_ligne), (1200, pos_ligne), (0,127,255), 3)  
                detec.remove((x,y))
                print("car is detected : "+str(cars))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detecter",dilatation)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
