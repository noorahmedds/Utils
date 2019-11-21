import grequests
from pycocotools.coco import COCO
import requests
import csv
import tqdm
import threading


class Test:
    def __init__(self, n_threads=100):
        self.n_threads = n_threads
        self.coco = COCO('cocoapi-master/annotations/instances_val2017.json')

        # Get the category ID for person
        self.catIds = self.coco.getCatIds(catNms=['person'])

        # Get image ids associated with the category required
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)

        # Get locations to those images
        self.images = self.coco.loadImgs(self.imgIds)

    def exception(self, request, exception):
        print("Problem: {}: {}".format(request.url, exception))

    def asyncs(self, id=0):
        n_images = len(self.images)

        batch_size = int(n_images / self.n_threads)
        left_idx = id*batch_size
        right_idx = min(((id+1)*batch_size) - 1, n_images - 1)

        urls = [im['coco_url'] for im in self.images[left_idx:right_idx]]

        results = grequests.map(
            (grequests.get(u) for u in tqdm.tqdm(urls)), exception_handler=self.exception, size=20)
        # print(results)

        # annot = open('annotations.csv', mode='w+', newline="")
        # annot_writer = csv.writer(annot)
        # Writing To file
        for res, im in tqdm.tqdm(zip(results, self.images[left_idx:right_idx])):
            try:
                with open('val_images/' + im['file_name'], 'wb+') as f:
                    f.write(res.content)

                # # Writing annotations to the annotations file
                # annIds = self.coco.getAnnIds(
                #     imgIds=im['id'], catIds=self.catIds, iscrowd=None)
                # anns = self.coco.loadAnns(annIds)
                # for i in range(len(anns)):
                #     annot_writer.writerow(['images/' + im['file_name'], int(round(anns[i]['bbox'][0])), int(round(anns[i]['bbox'][1])), int(
                #         round(anns[i]['bbox'][0] + anns[i]['bbox'][2])), int(round(anns[i]['bbox'][1] + anns[i]['bbox'][3])), 0])
            except:
                continue
        # annot.close()
        print(f"Async {id} Complete")


n_threads = 50
test = Test(n_threads)

for t_id in range(n_threads):
    test.asyncs(t_id)
