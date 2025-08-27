import cv2
import numpy as np
import subprocess
import time
from playsound import playsound
from PIL import Image
from gtts import gTTS
import os


#traffic1 = "stop"
#traffic2 = " school nearby"
#traffic3 = "zig zag road ahead"
#language = 'en'

#output1 = gTTS(text=traffic1, lang=language, slow=False)
#output2 = gTTS(text=traffic2, lang=language, slow=False)
#output3 = gTTS(text=traffic3, lang=language, slow=False)

#output1.save("output1.mp3")
#output2.save("output2.mp3")
#output3.save("output3.mp3")

net = cv2.dnn.readNet('C:/Users/91999/OneDrive/Desktop/New folder (4)/yolov3_training_last .weights', 'C:/Users/91999/OneDrive/Desktop/New folder (4)/yolov3_testing.cfg')

classes = []
with open("C:/Users/91999/OneDrive/Desktop/New folder (4)/classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=False, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                print(class_id)
                if class_id == 0:
                    playsound("C:/Users/91999/OneDrive/Desktop/New folder (4)/Bump Ahead.mp3")
                    #os.system("C:/Users/91999/OneDrive/Desktop/New folder (4)/Bump Ahead.mp3")
                    #image = subprocess.Popen(["feh",  "-x", "-q" "-B", "black", "-g", "1280*800", "/home/pi/Downloads/traffic/stop.jpg"])
                    #time.sleep(2)
                    #image.kill()
                elif class_id == 1:
                    playsound("C:/Users/91999/OneDrive/Desktop/New folder (4)/Stop.mp3")
                    #os.system("omxplayer --no-keys C:/Users/91999/OneDrive/Desktop/New folder (4)/Stop.mp3")
                    #image = subprocess.Popen(["feh",  "-x", "-q" "-B", "black", "-g", "1280*800", "/home/pi/Downloads/traffic/double_bend.jpg"])
                    #time.sleep(2)
                    #image.kill()
                elif class_id == 2:
                    playsound("C:/Users/91999/OneDrive/Desktop/New folder (4)/40 kmhr.mp3")
                    #os.system("omxplayer --no-keys C:/Users/91999/OneDrive/Desktop/New folder (4)/40 kmhr.mp3")
                    #image = subprocess.Popen(["feh", "-x", "-q" "-B", "black", "-g", "1280*800", "/home/pi/Downloads/traffic/school_ahead.jpg"])
                    #time.sleep(2)
                    #image.kill()
                elif class_id == 3:
                    playsound("C:/Users/91999/OneDrive/Desktop/New folder (4)/Double bend Ahead.mp3")
                    #os.system("omxplayer --no-keys C:/Users/91999/OneDrive/Desktop/New folder (4)/Double bend Ahead.mp3")
                    #image = subprocess.Popen(["feh", "-x", "-q" "-B", "black", "-g", "1280*800", "/home/pi/Downloads/traffic/school_ahead.jpg"])
                    #time.sleep(2)
                    #image.kill()
                elif class_id == 4:
                    playsound("C:/Users/91999/OneDrive/Desktop/New folder (4)/School ahead.mp3")
                    #os.system("omxplayer --no-keys C:/Users/91999/OneDrive/Desktop/New folder (4)/School ahead.mp3")
                    #image = subprocess.Popen(["feh", "-x", "-q" "-B", "black", "-g", "1280*800", "/home/pi/Downloads/traffic/school_ahead.jpg"])
                    #time.sleep(2)
                    #image.kill()
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
