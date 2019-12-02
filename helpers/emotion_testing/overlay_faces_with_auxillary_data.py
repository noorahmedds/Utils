import os
import glob
import cv2
import numpy as np
import pprint
import dlib
from imutils.face_utils import visualize_facial_landmarks, shape_to_np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from models import CNNArchitecture
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
#           (168, 100, 168), (158, 163, 32),
#           (163, 38, 32), (180, 42, 220), (100, 100, 200)]
# colors = [(100, 100, 100), (100, 100, 100), (100, 100, 100),
#           (100, 100, 100), (100, 100, 200), (100, 100, 100), (100, 100, 200), (100, 100, 100)]


# predictor = dlib.shape_predictor(
#     "../smart_gandola/weights/shape_predictor_68_face_landmarks.dat")

# happy_path = glob.glob("./Data/processed/train/happy/*")
# neutral_path = glob.glob("./Data/processed/train/neutral/*")
# surprise_path = glob.glob("./Data/processed/train/surprise/*")

# happy_file_names = [tuple(img.split('/')[-1].split("_")) for img in happy_path]
# neutral_file_names = [tuple(img.split('/')[-1].split("_"))
#                       for img in neutral_path]
# surprise_file_names = [tuple(img.split('/')[-1].split("_"))
#                        for img in surprise_path]

# # happy_landmarks_path = [
# #     f"./Data/Landmarks/{folder}/{expression}/{folder}_{expression}_{text_file.split('.')[0]}_landmarks.txt" for folder, expression, text_file in happy_file_names]
# # neutral_landmarks_path = [
# #     f"./Data/Landmarks/{folder}/{expression}/{folder}_{expression}_{text_file.split('.')[0]}_landmarks.txt" for folder, expression, text_file in neutral_file_names]
# # surprise_landmarks_path = [
# #     f"./Data/Landmarks/{folder}/{expression}/{folder}_{expression}_{text_file.split('.')[0]}_landmarks.txt" for folder, expression, text_file in surprise_file_names]


# # with open(happy_landmarks_path[0], 'r') as file_obj:
# #     content = file_obj.readlines()
# #     content = [np.array(list(map(float, point.split()))) -
# #                np.array([165, 140]) for point in content]

# white = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
#          (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]

# for img in tqdm(happy_path):
#     name = img.split('/')[-1].split(".")[0]

#     image = cv2.imread(img)
#     image = cv2.resize(image, (224, 224))

#     landmarked_img = np.zeros_like(image)
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     shape = predictor(gray, dlib.rectangle(
#         0, 0, image.shape[0], image.shape[1]))
#     shape = shape_to_np(shape)
#     out = visualize_facial_landmarks(landmarked_img, shape, colors=white)

#     out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
#     out = out[:, :, np.newaxis]

#     stacked = np.append(image, out, axis=2)
#     stacked = stacked/255.0

#     with open(f"./Processed/Happy/{name}.pickle", "wb") as f_out:
#         pickle.dump(stacked, f_out)

# for img in tqdm(neutral_path):
#     name = img.split('/')[-1].split(".")[0]

#     image = cv2.imread(img)
#     image = cv2.resize(image, (224, 224))

#     landmarked_img = np.zeros_like(image)
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     shape = predictor(gray, dlib.rectangle(
#         0, 0, image.shape[0], image.shape[1]))
#     shape = shape_to_np(shape)
#     out = visualize_facial_landmarks(landmarked_img, shape, colors=white)

#     out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
#     out = out[:, :, np.newaxis]

#     stacked = np.append(image, out, axis=2)
#     stacked = stacked/255.0

#     with open(f"./Processed/Neutral/{name}.pickle", "wb") as f_out:
#         pickle.dump(stacked, f_out)

# for img in tqdm(surprise_path):
#     name = img.split('/')[-1].split(".")[0]

#     image = cv2.imread(img)
#     image = cv2.resize(image, (224, 224))

#     landmarked_img = np.zeros_like(image)
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     shape = predictor(gray, dlib.rectangle(
#         0, 0, image.shape[0], image.shape[1]))
#     shape = shape_to_np(shape)
#     out = visualize_facial_landmarks(landmarked_img, shape, colors=white)

#     out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
#     out = out[:, :, np.newaxis]

#     stacked = np.append(image, out, axis=2)
#     stacked = stacked/255.0

#     with open(f"./Processed/Surprise/{name}.pickle", "wb") as f_out:
#         pickle.dump(stacked, f_out)

# print("Exported 4 channel image")

# # cv2.imshow("out", stacked)
# # plt.show()
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# =============================

happy_pickle_path = glob.glob("./Processed/Happy/*")
neutral_pickle_path = glob.glob("./Processed/Neutral/*")
surprise_pickle_path = glob.glob("./Processed/Surprise/*")

labels = []
data = []
for path in happy_pickle_path:
    with open(path, "rb") as f_in:
        data.append(pickle.load(f_in))
        labels.append(0)

for path in neutral_pickle_path:
    with open(path, "rb") as f_in:
        data.append(pickle.load(f_in))
        labels.append(1)

for path in surprise_pickle_path:
    with open(path, "rb") as f_in:
        data.append(pickle.load(f_in))
        labels.append(2)


data = np.array(data)
print(data.shape)

labels = np.array(labels)
labels = to_categorical(labels)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.05)


inception, callback_list, name = CNNArchitecture(input_shape=(
    224, 224, 4), num_classes=3, loss="categorical_crossentropy", optimizer="adam").inception_v3()
history = inception.fit(X_train, y_train,
                        batch_size=16, epochs=5, shuffle=True,
                        validation_data=(X_test, y_test))


# =========================================
# train_datagen = ImageDataGenerator(
#     validation_split=0.2
# )
# train_generator = train_datagen.flow_from_directory(
#     "./Processed", target_size=(224, 224), batch_size=8, subset='training'
# )

# valid_generator = train_datagen.flow_from_directory(
#     "./Processed", target_size=(224, 224), batch_size=4, subset='validation'
# )
