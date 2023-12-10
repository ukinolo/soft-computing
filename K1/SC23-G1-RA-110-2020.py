import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sys

#Autor Uros Nikolovski: RA 110 2020

def calculate_mae(correct, my):
    total_difference = 0
    for i in range(len(correct)):
        total_difference += abs(correct[i] - my[i])
    return total_difference/len(correct)

columns = ['Naziv slike','Broj squirtle-ova']
data = [
['picture_1.jpg',4],
['picture_2.jpg',8],
['picture_3.jpg',6],
['picture_4.jpg',8],
['picture_5.jpg',8],
['picture_6.jpg',4],
['picture_7.jpg',6],
['picture_8.jpg',6],
['picture_9.jpg',6],
['picture_10.jpg',13],
]
results = pd.DataFrame(data, columns=columns)
#results = pd.read_csv('squirtle_count.csv')
pictures = results['Naziv slike']
correct_values = results['Broj squirtle-ova']

filepath = sys.argv[1]

my_values = []
for i in range(results.shape[0]):
    picture_name = pictures[i]
    correct_value = correct_values[i]

    img = cv2.imread(filepath + picture_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 50)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img_update = cv2.morphologyEx(img_bin, cv2.MORPH_ERODE, kernel, iterations = 4)

    contours, hierarchy = cv2.findContours(img_update, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    squirtle_countours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        center, size, angle = cv2.minAreaRect(contour)
        height, width = size

        if area > 320 and area < 5000 and abs(height - width) < 24 and width > 30 and height > 25:
            squirtle_countours.append(contour)

    # fig = plt.figure(figsize=(10, 10))
    
    # img_copy1 = img.copy()
    # cv2.drawContours(img_copy1, contours, -1, (255, 0, 0), 1)
    # img_copy2 = img.copy()
    # cv2.drawContours(img_copy2, squirtle_countours, -1, (255, 0, 0), 1)
    
    # fig.add_subplot(2, 1, 1)
    # plt.imshow(img_copy1)
    
    # fig.add_subplot(2, 1, 2)
    # plt.imshow(img_copy2)
    # plt.show()

    my_value = len(squirtle_countours)
    my_values.append(my_value)

    print(picture_name + '-' + str(correct_value) + '-' + str(my_value))

print(calculate_mae(correct_values, my_values))
