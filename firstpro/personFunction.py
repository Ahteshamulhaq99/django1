import torch
from deepface import DeepFace
import numpy as np

import detect_face 
import cv2
from tensorflow.keras.preprocessing import image

from math import atan2, degrees, radians
from PIL import Image
model_rec = DeepFace.build_model("Facenet512")
weightPath = "yolov5s-face.pt"
device = "cpu"
model = detect_face.load_model(weightPath, device)

def get_angle(point_1, point_2): #These can also be four parameters instead of two arrays
    angle = atan2(point_1[1] - point_2[1], point_1[0] - point_2[0])
    angle = degrees(angle)
    return angle   



def process(img):
    boxes, landmarks, confs = detect_face.detect_one(model, img, device)
    
    if len(boxes) > 0:
        box = boxes[0]

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        q1, r1, q2, r2 = landmarks[0][:4]
        angle=get_angle((q2, r2),(q1, r1))
        
        img2 = Image.fromarray(img[int(y1):int(y2),int(x1):int(x2)])
        img2=np.array(img2.rotate(angle)) 
        
        width = height = 160
        # img=cv2.resize(img,(width,height))
        # img=img.reshape(1,width,height,3)
        img = preprocess_face(img2, target_size=(width, height), grayscale=False, enforce_detection=True,
                              detector_backend='opencv', return_region=False, align=True)

        img_representation = model_rec.predict(img)[0, :]
        return (img_representation)
    else:
        return (" Face not found ")


def preprocess_face(img, target_size=(224, 224), grayscale=False, enforce_detection=True, detector_backend='opencv',
                    return_region=False, align=False):
    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    # img = load_image(img)
    base_img = img.copy()

    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection == True:
            raise ValueError("Detected face shape is ", img.shape,
                             ". Consider to set enforce_detection argument to False.")
        else:  # restore base image
            img = base_img.copy()

    # post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize image to expected shape

    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        # print(img)
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                         'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    try:
        # double check: if target image is not still the same size with target.
        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)
    except:
        pass

    # normalizing the image pixels

    img_pixels = image.img_to_array(img)  # what this line doing? must?
    # print("1",img_pixels.shape)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    # print("2",img_pixels.shape)
    img_pixels /= 255  # normalize input in [0, 1]

    if return_region == True:
        return img_pixels
    else:
        return img_pixels


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def process_data(f_cnic, f_selfie):

    # selfie_path = "dataset/" + f_selfie
    # cnic_path = "dataset/" + f_cnic #"dataset/" + f_cnic
    selfie_path=r"C:\Users\karachigamerz.com\Desktop\ahteshammn\KAIR-master\testsets\setmy/a.jpg"
    cnic_path=r"C:\Users\karachigamerz.com\Desktop\ahteshammn\KAIR-master\testsets\setmy/b.jpg"
    selfie = cv2.imread(selfie_path)
    cnic = cv2.imread(cnic_path)

    selfie = cv2.cvtColor(selfie, cv2.COLOR_BGR2RGB)
    cnic = cv2.cvtColor(cnic, cv2.COLOR_BGR2RGB)
    	
    selfie = cv2.cvtColor(selfie, cv2.COLOR_RGB2GRAY)
    cnic = cv2.cvtColor(cnic, cv2.COLOR_RGB2GRAY)

    selfie = np.stack((selfie,)*3, axis=-1)
    cnic = np.stack((cnic,)*3, axis=-1)
 
    selfie_embedding = process(selfie)

    if len(selfie_embedding)==512:
        selfie_embedding =[ i/3 for i in selfie_embedding]
        selfie_embedding=np.array(selfie_embedding)
    else:        
        return "  face not found","in "+selfie_path.split("/")[-1]

    cnic_embedding = process(cnic)

    if len(cnic_embedding)==512:
        cnic_embedding=[ j/3 for j in cnic_embedding]
        cnic_embedding=np.array(cnic_embedding)
    else:        
        return "  face not found","in "+cnic_path.split("/")[-1]

    dist = findCosineDistance(selfie_embedding, cnic_embedding)
    
    if dist<0.4:
        # print("Same Person Found")
        dcs = "Accepted"
    else:
        dcs = "Rejected"
        # print("Dissimilar Detected")
   
    if dist<1.5:
        dist_Cnic_selfie=int(0.5**((dist**2)/(1.5-dist))*100)
    else:
        dist_Cnic_selfie=0

    if dcs=="Rejected" and dist_Cnic_selfie > 35:
        dist_Cnic_selfie=dist_Cnic_selfie-30
   
    return  dcs, str(dist_Cnic_selfie)+"%"

