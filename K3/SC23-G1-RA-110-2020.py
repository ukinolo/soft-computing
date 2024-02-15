import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import sys

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.cluster import KMeans

from tensorflow.keras.optimizers import SGD

#Data
image_names = ['captcha_1.jpg',
               'captcha_2.jpg',
               'captcha_3.jpg',
               'captcha_4.jpg',
               'captcha_5.jpg',
               'captcha_6.jpg',
               'captcha_7.jpg',
               'captcha_8.jpg',
               'captcha_9.jpg',
               'captcha_10.jpg',]

text = ['стучать стучать',
        'кто это',
        'кошатница',
        'фул давай',
        'изи катка',
        'беспозвоночное',
        'юность щенок',
        'ягода ёж',
        'голубой экран',
        'хороший въезд',]


#Parameters
input_image_size = 30

#Alphabet
alphabet = set(''.join(text))
alphabet.remove(' ')
alphabet = sorted(list(alphabet))


#Image manipulation function
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def erode(image):
    kernel = np.array([[1],
                       [1],
                       [1],])
    return cv2.erode(image, kernel, iterations=4)

def crop_image(image):
    return image[150:280, 250:830]

def prepare_image(image):
    return erode(image_bin(image_gray(crop_image(image))))

def prepare_without_erode(image):
    return image_bin(image_gray(image))

#Manipulation regions of interest
def resize_region(region):
    return cv2.resize(region, (input_image_size, input_image_size), interpolation=cv2.INTER_NEAREST)

def select_roi_with_distances(image_orig, image_bin):
    image_copy = prepare_without_erode(image_orig.copy())
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    y_sub = 5
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y += y_sub
        h -= 2 * y_sub
        area = cv2.contourArea(contour)
        if area > 100 and h < 100 and h > 20 and w > 20:
            region = image_copy[y:y+h, x:x+w]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

#Image preparation to nn input
def scale_to_range(image):
    return image/255

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(scale.flatten())
    return ready_for_ann

#NN
def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(512, input_dim=input_image_size**2, activation='relu'))
    ann.add(Dense(output_size, activation='softmax'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
    
    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann

#One hot encoding
def one_hot_encode(char):
    ret = np.zeros(len(alphabet))
    one_index = [i for i, c in enumerate(alphabet) if c == char]
    if len(one_index) != 1:
        raise Exception('Characted does not exist in alphaber, or there are duplicates in alphabet')
    one_index = one_index[0]
    ret[one_index] = 1
    return ret

def one_hot_decode(code):
    if len(code) != len(alphabet):
        return None
    char_index = [i for i, c in enumerate(code) if c == 1]
    if len(char_index) != 1:
        raise Exception('Array does not contain one encoded element')
    char_index = char_index[0]
    return alphabet[char_index]

#Making training data
def get_train_data(filePath):
    local_alphabet = set(''.join(text))
    local_alphabet.remove(' ')

    all_distances = []
    test_data = []

    for i, image_path in enumerate(image_names):
        image_text = text[i]
        image_text.strip()
        image_text = image_text.replace(" ", "")
        image = load_image(filePath + image_path)
        _, sorted_regions, distances = select_roi_with_distances(crop_image(image), prepare_image(image))
        all_distances.extend(distances)
        for char, region in zip(image_text, sorted_regions):
            if char in local_alphabet:
                local_alphabet.remove(char)
                test_data.append((char, region))

    return test_data, all_distances

#Processing NN output
def winner_index(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result_with_spaces(outputs, alphabet, k_means, count):
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner_index(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[count] == w_space_group:
            result += ' '
        result += alphabet[winner_index(output)]
        count += 1
    return result

if __name__ == '__main__':
    filePath = sys.argv[1]
    test_data, distances = get_train_data(filePath)

    #Creating k_means that detects spaces in text
    distances = np.array(distances).reshape(len(distances), 1)
    k_means = KMeans(n_clusters=2)
    k_means.fit(distances)

    #Preparing train data for ann
    X_train = []
    Y_train = []
    for data in test_data:
        X_train.append(data[1])
        Y_train.append(one_hot_encode(data[0]))
    X_train = prepare_for_ann(X_train)

    ann = create_ann(output_size=len(alphabet))
    ann = train_ann(ann, X_train, Y_train, epochs=1000)

    char_space_count = 0
    my_results = []
    #Analyzing all the images
    for i, image_name in enumerate(image_names):
        image = load_image(filePath + image_name)
        _, sorted_regions, _ = select_roi_with_distances(crop_image(image), prepare_image(image))

        sorted_regions = prepare_for_ann(sorted_regions)
        result = ann.predict(np.array(sorted_regions, np.float32))

        my_str = display_result_with_spaces(result, alphabet, k_means, char_space_count)
        my_results.append(my_str)
        char_space_count += len(result) - 1
        print(f'{image_name}-{text[i]}-{my_str}')
    
    print(hamming(my_results, text))
