from PIL import Image
import os

# for files in os.scandir("build/darknet/x64/data/obj"):
#     print(files)

with open("wider_face_train_bbx_gt.txt") as reader:
    img_name = reader.readline().strip("\n")
    while img_name != '':
        img_name = img_name.split(".")[0]
        print(img_name)
        img_name = img_name.split("/")[1]
        im = Image.open("build/darknet/x64/data/obj/" + img_name + ".jpg")
        # img_name = reader.readline().strip("\n")
        no_of_faces = reader.readline().strip("\n")
        coords = list()

        if no_of_faces == '0':
            string = "build/darknet/x64/data/obj/" + img_name + ".txt"
            reader.readline()
            with open(string, 'w') as writer:
                writer.write(str(0))
        else:
            for index in range(int(no_of_faces)):
                width, height = im.size
                text = reader.readline().strip("\n").split(" ")
                text = str(0) + " " + str(int(text[0]) / width) + " " +\
                       str(int(text[1]) / height) + " " +\
                       str(int(text[2]) / width) + " " +\
                       str(int(text[3]) / height)
                coords.append(text)

            string = "build/darknet/x64/data/obj/" + img_name + ".txt"
            with open(string, 'w') as writer:
                # img_name = img_name.split(".")[0] + ".jpg"
                # img_name = img_name[0] + ".jpg"
                # writer.write(img_name + "\n")
                # writer.write(no_of_faces + "\n")
                for coord in coords:
                    writer.write(coord + "\n")

        img_name = reader.readline().strip("\n")








# for images in os.scandir("build/darknet/x64/obj"):
#
#     print(images.name)
#
#     with open("build/darknet/x64/data/train.txt", "a") as writer:
#
#
#         writer.write("data/obj/" + images.name + "\n")
#         print("data/obj/" + images.name)