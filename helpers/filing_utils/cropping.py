import numpy as np
import cv2
import glob
import tqdm
import os
import shutil
​
def label2image(image_path, labels_path):
    img = cv2.imread(image_path)
    # labels_path = "./Updated Labels/"
    # label = labels_path + image_path.split(".")[-2].split("/")[-1] + ".txt"
​
    color = [(255,2,0), (0,255,0), (0,0,255)]
​
    with open(labels_path, "r") as f:
        for line in f:
            if line:
                cat, xc, yc, w, h = (float(i) for i in line.split())
​
                # Denormalise
                _h, _w, _ = img.shape
​
                xc  *= _w
                yc  *= _h
                w   *= _w
                h   *= _h
​
                p1 = (int(xc - w/2), int(yc - h/2))
                p2 = (int(xc + w/2), int(yc + h/2))
​
                # print(p1, p2, color[cat])
​
                cv2.rectangle(img, p1, p2, color[int(cat)], 1)
​
        cv2.imshow("Labelled Image: ", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
​
def write_image_and_annot(output_path, game_id, image_path, label_path):
    output_path = output_path+"/"+game_id
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
​
    shutil.copy(image_path, output_path)
    shutil.copy(label_path, output_path)
​
def main(output_path):
    stats_dict = {
        "ball_count":0,
        "player_count":0,
        "ref_count":0,
        "invalid_frame_count":0,
        "frames_cropped":0
    }
​
    game_ids = list(range(1, 30))
    game_paths = ["Game " + str(id) for id in game_ids]
​
    for game_path, game_id in zip(game_paths, game_ids):
        game_id = str(game_id)
​
        images = glob.glob(f"./Images/{game_path}/*.jpg")
        labels_path = f"./Updated Labels/{game_path}/"
​
        if not os.path.isdir(f"{output_path}/{game_id}"):
            os.mkdir(f"{output_path}/{game_id}")
​
        swap_ref_ball = False
        is720 = False
        
        if game_id in [10,15,16]:
            swap_ref_ball = True
​
        for image_path in tqdm.tqdm(images):
            # print(images)
            image_id = image_path.split(".")[-2].split("/")[-1]
            label = labels_path + image_id + ".txt"
            
            crop_shape = (620, 1100)
​
            if os.stat(label).st_size == 0:
                # Skip this image
                continue
            else:
                if is720:
                    # Directly write annotations and image to output path
                    write_image_and_annot(output_path, game_id, image_path, label)
                else:    
                    # We had some stuff in labels
                    img = cv2.imread(image_path)
                    _h, _w, _ = img.shape
                    imshape = np.array(img.shape)
                    
                    if  (_h >= 720 and _w > 720):
                        # print("IMAGE IS BEING RENORMALIZED")
                        updated_label = ""
​
                        # print(label)
                        
                        # Images Updating
                        c_h, c_w = crop_shape
                        crop_img = img[:c_h,:c_w].copy()
​
                        # READ FILE LINE BY LINE
                        f = open(label, "r")
                        # Labels Updating
                        for line in f:
                            
                            if line:
                                cat, xc, yc, w, h = (float(i) for i in line.split())
                                cat = int(cat)
​
                                if swap_ref_ball:
                                    if cat == 1:
                                        cat = 2
                                    elif cat == 2:
                                        cat = 1
​
                                # print(line)
​
                                # Denormalise and ReNormalize
                                xc  *= _w / c_w
                                yc  *= _h / c_h
                                w   *= _w / c_w
                                h   *= _h / c_h
​
                                updated_label += f"{int(cat)} {xc} {yc} {w} {h}\n"
​
                                if cat == 0:
                                    stats_dict['player_count'] += 1
                                elif cat == 1:
                                    stats_dict['ball_count'] += 1
                                else:                                
                                    stats_dict['ref_count'] += 1
​
                        # Here we write the image and updated_label to the output folder
                        out_label_path = output_path+ "/" + game_id +"/" +image_id+".txt"
                        out_image_path = output_path + "/" + game_id + "/" + image_id + ".jpg"
​
                        with open(out_label_path, "w") as out:
                            out.write(updated_label)
​
                        cv2.imwrite(out_image_path, crop_img)
                    else:
                        is720 = True
                        write_image_and_annot(output_path, game_id, image_path, label)
        # print(stats_dict)
​
    return
​
if __name__ == "__main__":
    main("./Output")
    # label2image("./Output/1/23090.jpg","./Output/1/23090.txt")

 
