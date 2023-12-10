import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

#Autor Uros Nikolovski: RA 110 2020

def calculate_mae(correct, my):
    total_difference = 0
    for i in range(len(correct)):
        total_difference += abs(correct[i] - my[i])
    return total_difference/len(correct)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')




#Loading images

pos_imgs = []
neg_imgs = []

filepath = sys.argv[1]
picture_filepath = os.path.join(filepath, 'pictures/')
video_filepath = os.path.join(filepath, 'videos/')

for img_name in os.listdir(picture_filepath):
    img_path = os.path.join(picture_filepath, img_name)
    img = load_image(img_path)
    if 'p_' in img_name:
        pos_imgs.append(img)
        #pos_imgs.append(img[::-1, :])
        #pos_imgs.append(img[::-1, ::-1])
        #pos_imgs.append(img[:, ::-1])
    elif 'n_' in img_name:
        neg_imgs.append(img)
        #neg_imgs.append(img[::-1, :])
        #neg_imgs.append(img[::-1, ::-1])
        #neg_imgs.append(img[:, ::-1])
        
#print("Positive images #: ", len(pos_imgs))
#print("Negative images #: ", len(neg_imgs))




#Calculating features using hog descriptor

pos_features = []
neg_features = []
labels = []

nbins = 12
cell_size = (8, 8)
block_size = (3, 3)

hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)




#Training svm

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print('Train shape: ', x_train.shape, y_train.shape)
#print('Test shape: ', x_test.shape, y_test.shape)

clf_svm = SVC(kernel='linear', probability=True) 
clf_svm.fit(x_train, y_train)
y_train_pred = clf_svm.predict(x_train)
y_test_pred = clf_svm.predict(x_test)
#print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
#print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))




#Some comment

def classify_window(window):
    features = hog.compute(window).reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]

def get_predicitons_boxes(image, step_size, min_y, min_x, max_y, max_x, prediciton_threshold=0.96, window_size=(120, 60)):
    prediciton_boxes = []
    x_enlargment = 39
    y_enlargment = 68
    for y in range(max(min_y, 0), min(image.shape[0], max_y), step_size):
        for x in range(max(min_x, 0), min(image.shape[1], max_x), step_size):
            window = image[y:y+window_size[0]+y_enlargment, x:x+window_size[1]+x_enlargment]
            window = cv2.resize(window, (window_size[1], window_size[0]))
            if window.shape == (window_size[0], window_size[1]):
                score = classify_window(window)
                if score > prediciton_threshold:
                    prediciton_boxes.append([score, [y, x, y+window_size[0]+y_enlargment, x+window_size[1]+x_enlargment]])
    return prediciton_boxes

def jaccard_index(box1, box2):
    y_a = max(box1[0], box2[0])
    x_a = max(box1[1], box2[1])
    y_b = min(box1[2], box2[2])
    x_b = min(box1[3], box2[3])
    
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    
    true_area = (box1[3] - box1[1] + 1) * (box1[2] - box1[0] + 1)

    pred_area = (box2[3] - box2[1] + 1) * (box2[2] - box2[0] + 1)
    
    iou = inter_area / float(true_area + pred_area - inter_area)
    
    return max(iou, 0)

def get_maximum_score_box_index(prediction_boxes):
    max_score = prediction_boxes[0][0]
    max_index = 0
    for i in range(len(prediction_boxes)):
        if prediction_boxes[i][0] > max_score:
            max_score = prediction_boxes[i][0]
            max_index = i
    
    return max_index

def non_maximum_suppression(prediction_boxes, thresh_iou=0.4):
    true_predictions = []
    while len(prediction_boxes) > 0:
        max_index = get_maximum_score_box_index(prediction_boxes)
        best_box = prediction_boxes[max_index]
        true_predictions.append(best_box)
        del prediction_boxes[max_index]

        index_to_remove = []
        for i in range(len(prediction_boxes)):
            if(jaccard_index(best_box[1], prediction_boxes[i][1])) > thresh_iou:
                index_to_remove.append(i)
        
        index_to_remove.reverse()
        for index in index_to_remove:
            del prediction_boxes[index]

    return true_predictions

def get_box_centers(true_predictions):
    centers = []
    for i in range(len(true_predictions)):
        box = true_predictions[i]
        centers.append([int((box[1][2] - box[1][0])/2 + box[1][0]),
                        int((box[1][3] - box[1][1])/2 + box[1][1])])
    return centers

def process_image(image, step_size, min_y, min_x, max_y, max_x, prediciton_threshold=0.93, window_size=(120, 60), thresh_iou=0.2):
    prediction_boxes = get_predicitons_boxes(image, step_size, min_y, min_x, max_y, max_x, prediciton_threshold, window_size)
    true_predictions = non_maximum_suppression(prediction_boxes, thresh_iou)
    car_coordinates = get_box_centers(true_predictions)
    return car_coordinates

def detect_line(img):
    edges_img = cv2.Canny(img, 500, 700, apertureSize=3, L2gradient=True)
    
    min_line_length = 500
    
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=50, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=30)
    
    x1 = lines[0][0][0]
    y1 = img.shape[0] - lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = img.shape[0] - lines[0][0][3]
    
    return (x1, y1, x2, y2)

def get_line_params(line_coords):
    k = (float(line_coords[3]) - float(line_coords[1])) / (float(1500) - float(500))
    n = k * (float(-500)) + float(line_coords[1])
    return k, n

def detect_cross(x, y, k, n):
    yy = k*x + n
    return -90 <= (yy - y) <= 80

def process_video(video_path, process_every_nth=1):
    num_of_crosses = 0
    
    frame_num = 0
    process_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)

    line_detect_frequency = 20 #After 20 processed frames, I will detect new line
    
    last_run_detected_cars = []
    
    while True:
        process_num += 1
        grabbed, frame = cap.read()
        if not grabbed:
            break
        else:
            for i in range(process_every_nth - 1):
                frame_num += 1
                grabbed, new_frame = cap.read()
                if not grabbed:
                    break
                if grabbed:
                    frame = new_frame

                
        if process_num % line_detect_frequency == 1:
            line_coords = detect_line(frame)

            k, n = get_line_params(line_coords)
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = process_image(frame_gray, 12, line_coords[1]-150, 1350, line_coords[1]+90, 2450)
        current_detected_cars = []
        for car in cars:
            if detect_cross(car[1], car[0], k, n):
                detected_same_car = False
                current_detected_cars.append(car)
                for detected_car in last_run_detected_cars:
                    if abs(detected_car[1] - car[1]) < 25:
                        detected_same_car = True
                        break
                if(not detected_same_car):
                    num_of_crosses += 1
        last_run_detected_cars = current_detected_cars
    
    return num_of_crosses




videos = [
    'segment_1.mp4',
    'segment_2.mp4',
    'segment_3.mp4',
    'segment_4.mp4',
]
results = [
    13,
    10,
    15,
    17
]
my_result = []
for i in range(len(videos)):
    my_result.append(process_video(os.path.join(video_filepath, videos[i]), process_every_nth=5))
    print(f'{videos[i]}-{results[i]}-{my_result[-1]}')

print(calculate_mae(results, my_result))