from ultralytics import YOLO
import pandas as pd
from tracker import *
import cv2
import numpy as np
class Yolo_Loading():
    def __init__(self,path):
        self.path=path
        self.traker=Tracker()
    def load_model(self):
        try:
            self.model = YOLO(self.path)
            self.name_of_class = self.model.names
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")

    def procees_frame(self,frame,people_entering,entering,people_exiting,exiting,area1,area2):
        results=self.model.predict(frame)
        list=[]
        result=results[0].boxes.data
        info_of_result=pd.DataFrame(result).astype("float")
        for index,row in info_of_result.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            name_of_object=self.name_of_class[0]
            if name_of_object == "person":  # Correct string comparison
                 list.append([x1, y1, x2, y2])
        bbox_of_tracker = self.traker.update(list)
        for bbox in bbox_of_tracker:
            x3,y3,x4,y4,id = bbox
        
            results = cv2.pointPolygonTest(np.array(area2,np.int32) ,((x4,y4)) , False )
            if results >=0:
                people_entering[id] = (x4,y4)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                
            if id in people_entering:
                results1 = cv2.pointPolygonTest(np.array(area1,np.int32) ,((x4,y4)) , False )
                if results1 >=0:
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                    cv2.circle(frame , (x4,y4) , 4 , (255,0,255),-1)
                    cv2.putText(frame,str(name_of_object),(x3,y3-10),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                    cv2.putText(frame,str(id),(x3+65,y3-10),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,0,255),1)
                    entering.add(id)
            
            # people exiting
            results2 = cv2.pointPolygonTest(np.array(area1,np.int32) ,((x4,y4)) , False )
            if results2 >=0:
                people_exiting[id] = (x4,y4)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
            if id in people_exiting:
                results3 = cv2.pointPolygonTest(np.array(area2,np.int32) ,((x4,y4)) , False )
                if results3 >=0:
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                    cv2.circle(frame , (x4,y4) , 4 , (255,0,255),-1)
                    cv2.putText(frame,str(name_of_object),(x3,y3-10),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                    cv2.putText(frame,str(id),(x3+55,y3-10),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,0,255),1)
                    exiting.add(id)