# We need to make a txt file which will contain for each image the bounding boxes for the persons in that image in the form xc, yc, w, h
# Then to that same file we need to add the faces
# This will take some time

# Open the annotions.csv file
# Find unique elements filenames
# For each filename

import numpy as np
import csv
import cv2
# from yoloface import infer
import tqdm
import glob

def process_bb(size, bb):

	dh = 1./size[0]
    dw = 1./size[1]

    x, y, w, h = bb
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x2 = x+w
    y2 = y+h

    xc = (x2 - x)/2.0
    yc = (y2 - y)/2.0

    return xc*dw, yc*dh, w*dw, h*dh


def write_to_files(annot_dict):
    # Here do a final run through over the dictionary and save to txt files
    # Annotations output file
    list_file = open("all.txt", "a+")
    for image_id in tqdm.tqdm(annot_dict.keys()):
        list_file.write(f"data/obj/{image_id}.jpg\n")

        out_path = f"annotations/{image_id}.txt"
        with open(out_path, 'w+') as out_file:
            # For annotations for a single image
            for ann in annot_dict[image_id]:
                # Write this annotation to the file
                out_file.write(" ".join([str(a) for a in ann]) + '\n')

    print("Done writing annotations to file")
    list_file.close()


def main():
    annotations_path = "annotations.csv"
    annot_dict = dict()
    faces_count = 0
    persons_count = 0

    # Reading annotations into a dictionary
    with open(annotations_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in tqdm.tqdm(csv_reader):

            if (line_count % 100) == 0:
                print("1000 images and written to files")
                write_to_files(annot_dict)
                annot_dict = {}

            file_path = row[0]
            image_id = file_path.split('/')[-1].split('.')[0]

            image = cv2.imread(file_path)
            if np.all(image == None):
                continue

            # The COCO bounding box format is [top left x position, top left y position, width, height].
            # Converting the bounding boxes to darkent format here
            size = image.shape
            xc, yc, w, h = process_bb(size, row[1:-1])
            class_id = int(row[-1])

            if image_id not in annot_dict.keys():
                annot_dict[image_id] = [[class_id, xc, yc, w, h]]
                line_count += 1

                # # Running yolo on the current image
                # # Saving annotations in the dictionary
                # faces = infer(file_path)
                # faces_count += len(faces)
                # # left, top, width, height

                # for face in faces:
                #     face_x, face_y, face_w, face_h = process_bb(size, face)
                #     face_ann = [1, face_x, face_y, face_w, face_h]
                #     annot_dict[image_id].append(face_ann)
            else:
                annot_dict[image_id].append([class_id, xc, yc, w, h])
            persons_count += 1

        print("Done populating annotation dictionary")
        print("Faces Count = ", faces_count)
        print("Persons Count = ", persons_count)
        # Now we just need to run yolo on each image. And add its annotations to the dict as well


def make_all_txt():
    # read through annotations
    # Get names and split based on .
    # get the image ids and write to the txt file
    in_dir = "./prepared_output/annotations/"
    out_path = "split/all.txt"

    all_anns = glob.glob("prepared_output/annotations/*.txt")

    with open(out_path, 'w+') as out:
        for ann in all_anns:
            new_ann = 'data/obj/' + \
                ann.split('.')[0].split('/')[2] + '.jpg' + '\n'
            out.write(new_ann)


if __name__ == "__main__":
    # main()
    make_all_txt()
