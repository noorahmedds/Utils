# Boiler plate from user: https://github.com/bluetooth12

from pycocotools.coco import COCO
import requests
import csv
import tqdm

cocos = ['cocoapi-master/annotations/instances_val2017.json', 'cocoapi-master/annotations/instances_train2017.json']
cocos = cocos[::-1]
img_files = ["val_images/", "train_images/"]
annot_files = ["val_annotations.csv", "train_annotations.csv"]

for coco_loc, annot_file, img_file in zip(cocos, annot_files, img_files):
    coco = COCO(coco_loc)

    # Get the category ID for person
    catIds = coco.getCatIds(catNms=['person'])

    # Get image ids associated with the category required
    imgIds = coco.getImgIds(catIds=catIds)

    # Get locations to those images 
    images = coco.loadImgs(imgIds)

    annot = open(annot_file, mode='w+', newline = "")
    annot_writer = csv.writer(annot)

    for im in tqdm.tqdm(images):
        # Downloading all images
        img_data = requests.get(im['coco_url']).content
        with open(img_file + im['file_name'], 'wb+') as f:
            f.write(img_data)

        # Writing annotations to the annotations file
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for i in range(len(anns)):  
            annot_writer.writerow([img_file + im['file_name'], int(round(anns[i]['bbox'][0])), int(round(anns[i]['bbox'][1])), int(round(anns[i]['bbox'][0] + anns[i]['bbox'][2])), int(round(anns[i]['bbox'][1] + anns[i]['bbox'][3])), 0])

    annot.close()
    print("Data has been downloaded!!")