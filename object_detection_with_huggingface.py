import random
from ultralytics import YOLO
import cv2
model = YOLO("yolo11n-seg.pt")  # load an official model  yolo11n.pt
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.cluster import KMeans
import re
import string
import os.path
from staticvectors import StaticVectors



punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
glovemodel = StaticVectors("neuml/glove-6B-quantized")



def text_to_embedding(text):
    tokens = pre_process_query(text)
    embedding = None
    for token in tokens:
        embedding = glovemodel.embeddings([token])
        break
    return embedding
    

def compute_similarity(user_input_vector,program_input_vector,threashold):
    
    program_norm = np.linalg.norm(program_input_vector[0])
    try:
        user_norm = np.linalg.norm(user_input_vector[0])
        similarity = user_input_vector[0].dot(program_input_vector[0]) / (user_norm * program_norm)
    except:
        lmao_vector = text_to_embedding("verbosity")
        lmao_norm = np.linalg.norm(lmao_vector[0])
        similarity = lmao_vector.dot(program_input_vector[0]) / (lmao_norm * program_norm)
        
    print(f"similarity: {similarity}")
    return similarity > threashold
    
def pre_process_query(query: str):
    lower = query.lower()
    no_punc = punc_regex.sub('', lower)
    tokens = no_punc.split()
    return tokens

def image_to_data(image):
    results = model(image, save=True, project="uploads")
    # box_heights = []
    # xyxy_list = []
    # print(results)
    output_results = []

    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        # xyxy_list.append(xyxy)
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        
        # print(f"names test: {names}")
        # print(f"xyxy  test: {xyxy}")
        original_image = result.orig_img
        confs = result.boxes.conf  # confidence score of each box
        mask = result.masks
        output_results.append(result)

    index = random.randint(0, len(xyxy)-1)
    
    xyxy_out = xyxy[index]
    name_out = names[index]
    
    return original_image, xyxy_out, name_out, mask, index, result


def crop_image_and_pick(orig_image, xyxy):
    
    x1, y1, x2, y2 =  xyxy
    cropped = orig_image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
    # cropped_img = cv2.imdecode(cropped, cv2.IMREAD_COLOR)
    # cv2.imshow('cool image', cropped_img) #can comment out later
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return cropped

def filepath_or_image_to_all_data(image_or_filepath):

    orig_img, xyxy, name, mask, index, result = image_to_data(image_or_filepath)

    color_name, color_rgb_value = get_color(orig_img, xyxy, mask, index)

    cropped_image = crop_image_and_pick(orig_img,xyxy)

    arr = np.array(cropped_image)
    arr = arr.astype(np.uint8)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'img3.jpg')
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, arr_bgr)

    return color_name, color_rgb_value, orig_img, cropped_image, name, result


def get_color(orig_img, xyxy, mask, index):


    ndarraymask = mask[index].data.cpu().numpy()
    desired_mask_shape = (orig_img.shape[1],orig_img.shape[0])
    resized_mask = cv2.resize(ndarraymask[0],desired_mask_shape, interpolation=cv2.INTER_NEAREST)

    bool_mask = resized_mask.astype(bool)
    masked_img = bool_mask[..., None] * orig_img
    masked_cropped_image = crop_image_and_pick(masked_img,xyxy)


    kmeans = KMeans(n_clusters=3, random_state=0).fit(masked_cropped_image.reshape(-1,3)[~np.all(masked_cropped_image.reshape(-1,3) == 0, axis=1)])
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster_id = np.argmax(counts)

    dominant_color = kmeans.cluster_centers_[labels[dominant_cluster_id]].astype(int)
  
    min_dist = float('inf')
    closest_name = None
    for name, hexcode in colors.XKCD_COLORS.items():
        r_c, g_c, b_c = tuple(int(hexcode.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) 
        dist = (r_c - dominant_color[0])**2 + (g_c - dominant_color[1])**2 + (b_c - dominant_color[2])**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name.replace("xkcd:", ""), dominant_color