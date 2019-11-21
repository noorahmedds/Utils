import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from shutil import copy
import random
import cv2
from yoloface import infer
import tqdm
# from mtcnn.mtcnn import MTCNN

sets = [('2012', 'person_train'),
        ('2007', 'person_train')]

test_sets = [('2012', 'person_val'),
             ('2007', 'person_val'), ('2007', 'person_test')]

classes = ["person", "face"]

emotion_dict = {0: 'happy', 1: 'neutral', 2: 'surprised'}


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

def convert2VOC(box):
    xmin = box[0]
    ymin = box[1]
    xmax = box[0] + box[2]
    ymax = box[1] + box[3]

    return (xmin, xmax, ymin, ymax)


def convert_annotation(year, image_id, faces, skip_difficult=True):
    # By default we skip difficult annots
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    out_file = open('obj/%s.txt' % (image_id), 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    ann_count = 0
    face_ann_count = 0
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        if skip_difficult:
            if int(difficult) == 1:
                continue

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        
        # Convert bb coordinates to xc,yc,w,h coordinates
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')

        # Here we need to write the bb coordinates for the face. If a face exists
        # face_bb = None
        # out_file.write("1" + " " +
        #                " ".join([str(a) for a in face_bb]) + '\n')
        ann_count += 1

    for face_bb in faces:
        # Convert yolo output to xc, yc, w, h
        face_bb = convert2VOC(face_bb)
        face_bb = convert((w,h), face_bb)

        out_file.write("1" + " " + " ".join([str(a) for a in face_bb]) + '\n')
        face_ann_count += 1

        
    return ann_count, face_ann_count


wd = getcwd()
print("Working Directory = ", wd)


def create_test_train_lists(image_ids, split=0.5):

    # now we iterate over all list_objects
    # Train
    train_file = open("train.txt", "w+")

    i = 0
    while i in range(int(len(image_ids) * split)):
        full_path = '%s\n' % (image_ids[i])
        train_file.write(full_path)
        i += 1
    train_file.close()

    test_file = open("test.txt", "w+")
    while i < len(image_ids):
        full_path = '%s\n' % (image_ids[i])
        test_file.write(full_path)
        i += 1
    test_file.close

    # for year, image_set in test_sets:
    #     if not os.path.exists('obj'):
    #         os.makedirs('obj')

    #     # if not os.path.exists('VOCdevkit/VOC%s/labels/' % (year)):
    #     #     os.makedirs('VOCdevkit/VOC%s/labels/' % (year))

    #     image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' %
    #                      (year, image_set)).read().strip().split()

    #     image_ids_new = []
    #     # remove extra info from image ids

    #     i = 0
    #     while i < len(image_ids):
    #         if (int(image_ids[i+1]) == 1):
    #             image_ids_new.append(image_ids[i])
    #         i += 2

    #     print(image_ids_new)

    #     list_file = open('train.txt', 'a+')
    #     for image_id in image_ids_new:
    #         # list_file.write('/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n' %
    #         #                 (year, image_id))
    #         list_file.write('data/obj/%s.jpg\n' % image_id)
    #         convert_annotation(year, image_id)
    #     list_file.close()


def populate_image_ids(file_path):
    # all_sets = [('2012', 'person_trainval'), ('2007', 'person_trainval')]
    # for year, image_set in all_sets:
    # image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' %
    #                  (year, image_set)).read().strip().split()

    image_ids = open(file_path).read().strip().split()

    return image_ids


def create_obj_foler():
    # this folder contains all pictures of persons and annotated data
    #  
    all_sets = [('2012', 'person_trainval'), ('2007', 'person_trainval'), ('2007', 'person_test')]
    true_counts = [4374, 2095, 2097]

    # detector = MTCNN()

    set_idx = 0
    for year, image_set in all_sets:
        if not os.path.exists('obj'):
            os.makedirs('obj')

        image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' %
                         (year, image_set)).read().strip().split()


        image_ids_new = []
        tf_idxs = []
        # remove extra info from image ids

        n_c = 0
        i = 0
        true_count = 0
        total_count = 0

        while i < len(image_ids):

            if (int(image_ids[i+1]) >= 0):
                # Appending images of the given class
                image_ids_new.append(image_ids[i])
                tf_idxs.append(True)
                true_count += 1
            else:
                # The number of negative frames should match the amount of true frames
                # Without any randomness. Assume randomness from the dataset itself
                if n_c < true_counts[set_idx]:
                    image_ids_new.append(image_ids[i])
                    tf_idxs.append(False)
                n_c += 1

            total_count += 1
            i += 2

        total_ann = 0
        face_ann_count = 0
        list_file = open('all.txt', 'a+')
        for i, image_id in tqdm.tqdm(enumerate(image_ids_new)):
            # here we copy the image to the obj folder
            full_path_abs = 'VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (
                year, image_id)
            out_path = 'obj/'
            copy(full_path_abs, out_path)

            # Test mtcnn and yolo_faces on the images
            faces = []
            if tf_idxs[i]:
                faces = infer(full_path_abs)
            # test_faces(full_path_abs, detector)
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     return
            #     cv2.destroyAllWindows()

            n_ann = convert_annotation(year, image_id, faces, skip_difficult=True)
            total_ann += n_ann[0]
            face_ann_count += n_ann[1]

            # if total_ann > 0:
            full_path = 'data/obj/%s.jpg\n' % (image_id)
            list_file.write(full_path)

        list_file.close()

        # Now list file contains all train val images from the voc person data
        # Now i need to create a splitter of this all.txt file for train test splits
        print("true_count: ", true_count)
        print("total_count: ", total_count)
        print("total_ann: ", total_ann)
        print("face_ann_count: ", face_ann_count)
        set_idx += 1

def test_faces(image_path, detector):
    # Test faces using mtcnn
    image = cv2.imread(image_path)
    detection = detector.detect_faces(image)
    for det in detection:
        # Get the bounding box
        bbox = det['box']
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
    cv2.imshow("Image", image)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     return False
    # return True

if __name__ == "__main__":
    create_obj_foler()
    # im_ids = populate_image_ids('all.txt')
    # random.shuffle(im_ids)
    # create_test_train_lists(im_ids, split=0.5)
