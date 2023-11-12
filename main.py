import cv2
import pandas as pd
import time
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('veh2.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

cy1=323
cy2=367
offset=6
vh_down = {}  # For counting cars going down side
vh_up = {}    # For counting cars going up side

vh_down_time = {}  # For counting time when cars going down side
vh_up_time = {}    # For counting time when cars going up side

down_counter = []
up_counter = []

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        
        # Now what we need is when car touches the 1st line then circle will draw on it and id will be noted
        # we are taking here of some better result
        
        # We have concept like if car going down so it will first touches the line 01 (cy1)
        # if car going up so it will first touches the line 02 (cy2)
        
        # --- This is for down going cars --- #
        if cy1 < (cy+offset) and cy1 > (cy-offset):
            vh_down[id] = time.time()  # Noting the time when vehicle touches the line
            
            # Now we have one more condition like when car touches first line as well as second line then we draw circle on it
        if id in vh_down:
            if cy2 < (cy+offset) and cy2 > (cy-offset):
                down_elapsed_time = time.time() - vh_down[id]  # calculating the time elpased after touches the line 1 to line 2
                if down_counter.count(id) == 0:
                    down_counter.append(id)
                    
                    # -- Taking the distance between our lines are 10 meters. It can be calculated by point distance formula
                    distance = 10
                    down_speed_ms = distance / down_elapsed_time
                    down_speed_kh = down_speed_ms * 3.6
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    cv2.putText(frame,str(int(down_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        
        # --- This is for up going cars --- #
        if cy2 < (cy+offset) and cy2 > (cy-offset):
            vh_up[id] = time.time()
            
            # Now we have one more condition like when car touches first line as well as second line then we draw circle on it
        if id in vh_up:
            if cy1 < (cy+offset) and cy1 > (cy-offset):
                up_elapsed_time = time.time() - vh_up[id]  # calculating the time elpased after touches the line 1 to line 2
                if up_counter.count(id) == 0:
                    up_counter.append(id)
                    # -- Taking the distance between our lines are 10 meters. It can be calculated by point distance formula
                    distance = 10
                    up_speed_ms = distance / up_elapsed_time
                    up_speed_kh = up_speed_ms * 3.6
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    cv2.putText(frame,str(int(up_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
           

        # Drawing the line to track the vehicles passing through it
        cv2.line(frame,(267,cy1),(829,cy1),(255,255,255),1)
        cv2.putText(frame,('1st Line'),(274,318),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        
        cv2.line(frame,(167,cy2),(932,cy2),(255,255,255),1)
        cv2.putText(frame,('2nd Line'),(181,363),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        
        cv2.putText(frame,('Going Down: ') + str(len(down_counter)),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        cv2.putText(frame,('Going Up: ') + str(len(up_counter)),(60,100),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

