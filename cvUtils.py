import cv2 as cv
import numpy as np
import imutils
from collections import namedtuple
import face_recognition
from imutils import face_utils
import dlib
import time
import sys

Point2D = namedtuple("Point2D", "x y")


def clamp(value, between):
    """
    Clamp value between a range.
    """
    return max(between[0], min(value, between[1]))


def doOverlap(boxA, boxB):
    if (boxA[0] > boxB[2] or boxB[0] > boxA[2]):
        return False
    if (boxA[1] > boxB[3] or boxB[1] > boxA[3]):
        return False
    return True


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def iou(boxA, boxB):
    """
    Calculate iou between two bounding boxes
    """

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = (xB - xA + 1) * (yB - yA + 1)

    # Compute the area of both rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IOU
    iou = float(intersection_area) / \
        float(boxA_area + boxB_area - intersection_area)
    return iou


def find_dominant_color(img, verbose=False):
    """Finds Dominant color from an image. Returns palette and the dominant RGB value for that image"""

    # Average
    # average = img.mean(axis=0).mean(axis=0)

    pixels = np.float32(img.reshape(-1, 3))
    n_colors = 5
    # the final param is epsilon. It is the minimum amount the centroid has to move before the update is terminated
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 500, .05)
    flags = cv.KMEANS_RANDOM_CENTERS

    # Using kmeans to determine n_colors centroids from the complete set of pixel values
    _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = np.int0(palette[np.argmax(counts)])

    if verbose:
        print("Palette: ", palette)
        print("Labels: ", labels)
        print("Labels length: ", len(labels))
        print("Counts: ", counts)
        print("Dominant: ", dominant)

    # display_color(counts, img, palette)
    # Swapping R channel with B
    dominant = dominant[::-1]

    return [palette], [dominant]


def imshow2(im_name, image, location=[0, 0], resize=Point2D(x=350, y=600)):
    """
    Shows the image in the upper left side (by default) of the screen while using the cv2.imshow functionality

    Parameters
    ----------
    im_name : str
        The name of the window
    image : numpy array type (cv::Mat)
        The already-read image you want to show
    location: list(), optional
        This will dictate where the window appears on your screen
    resize: Point2D(), optional
        This will resize the window. By default will be 600 by 350

    Returns
    -------
    null : void

    """
    cv.namedWindow(im_name, cv.WINDOW_NORMAL)
    cv.moveWindow(im_name, location[0], location[1])
    cv.resizeWindow(im_name, resize.x, resize.y)
    cv.imshow(im_name, image)


def waitAfterShow(key='q', verbose=False):
    """
    Wait for the users input with the given key to cv.destroyAllWindows()

    Parameters
    ----------
    key: char
        The key which, if pressed, will destroy all currently open windows
    verbose : bool
        Dictates printing for debugging purposes

    Returns
    -------
    null : void

    """
    while True:
        wait = cv.waitKey(0)
        if verbose:
            print(wait)
        if wait & 0xFF == ord(key):
            cv.destroyAllWindows()
            return


def processFrame(image):
    return imutils.auto_canny(image)


def streamProcessedVideo(videoPath, processFrame, willProcess=True, showFPS=True):
    """
    The function will show a processed stream of a video

    Parameters
    ----------
    processFrame: a callable function object
        This function has to process a gray image e.g. Convert frame to canny, blur a frame etc
    willProcess : Bool
        This dictates whether the processing will be done
    showFPS: Bool
        Toggles FPS printing

    Returns
    -------
    null : void
    """
    cap = cv.VideoCapture(videoPath)

    # FPS
    if showFPS:
        fps = 0.0
        start = time.time()

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Processing occurs here
        if willProcess:
            out = processFrame(gray)
        else:
            out = gray

        imshow2('frame', out)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # FPS
        if showFPS:
            end = time.time()
            seconds = end-start
            fps = (fps + (1/seconds))/2
            cv2.putText(out, "{0} fps".format(fps), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print("{0} fps".format(fps))
            start = end

    cap.release()
    cv.destroyAllWindows()


def streamProcessedWebcam(processFrame, willProcess=True, showFPS=True):
    """
    The function will show a processed stream of the webcam capture

    Parameters
    ----------
    processFrame: a callable function object
        This function has to process a gray image e.g. Convert frame to canny, blur a frame etc
    willProcess : Bool
        This dictates whether the processing will be done
    showFPS: Bool
        Toggles FPS printing

    Returns
    -------
    null : void
    """

    cap = cv.VideoCapture(0)

    # FPS
    if showFPS:
        fps = 0.0
        start = time.time()

    while(True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Processing occurs here
        if willProcess:
            out = processFrame(gray)
        else:
            out = gray

        # cv.imshow('frame',gray)
        imshow2('frame', out)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # FPS
        if showFPS:
            end = time.time()
            seconds = end-start
            fps = (fps + (1/seconds))/2
            cv2.putText(out, "{0} fps".format(fps), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print("{0} fps".format(fps))
            start = end

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

def est_pose(im, bbox):
    """
    Adapted from learnopencv.com script to include dlibs shape predictors input. Returns image and whether it is a frontal face or not
    
    Parameters
    ----------
    im: frame from image or video
    bbox: the shape array converted to a numpy array using imutils shape_to_np

    Returns
    -------
    im: with gui
    frontal: flag whether frontal or not

    """
    gray = cv2.cvtColor(im, 7)
    cv2.imshow("crop", im[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])
    tl = np.array((bbox[0], bbox[1]))
    shape = predictor(gray, dlib.rectangle(
        bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]))
    # Shape becomes a 2 dim array
    shape = shape_to_np(shape)

    im = visualize_facial_landmarks(im, shape)

    size = im.shape

    # shape += tl
    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        shape[33],     # Nose tip
        shape[8],     # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corne
        shape[48],     # Left Mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")

    # Adding top_left offset
    # image_points += tl

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        # Left eye left corner
        (-225.0, 170.0, -135.0),
        # Right eye right corne
        (225.0, 170.0, -135.0),
        # Left Mouth corner
        (-150.0, -150.0, -125.0),
        # Right mouth corner
        (150.0, -150.0, -125.0)

    ])

    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    print("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                  image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array(
        [(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(im, p1, p2, (255, 0, 0), 2)

    # cv2.imshow("out", im)
    frontal = False
    if abs(rotation_vector[2]) < 0.7:
        frontal = True

    return im, frontal

# ================= Utilitiy functions
def get_size(obj, seen=None):
    """Recursively finds size of a python object
    N.B: If the object has several references to other objects then this may significantly slow down processing as time progresses
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def prepare_output_for_tracking(yolo_output, img_shape):
    # Returns a list of predictions [xcycwh] for each bounding box

    processed_bbox = []
    processed_conf = []
    for bbox in yolo_output:
        if (bbox[-1] is not None):
            proc_box = xcycwh(bbox, img_shape)

            if not(proc_box[2] == 0 and proc_box[3] == 0):
                processed_bbox.append(proc_box)
                processed_conf.append([bbox[5]])

    return np.array(processed_bbox), np.array(processed_conf)


def xcycwh(bb, img_shape):
    img_height, img_width = img_shape[:-1]

    x1 = bb[1]
    y1 = bb[2]

    x2 = bb[3]
    y2 = bb[4]

    x1 = clamp(x1, [0, img_width])
    x2 = clamp(x2, [0, img_width])
    y1 = clamp(y1, [0, img_height])
    y2 = clamp(y2, [0, img_height])

    xc = x1 + (x2-x1)/2
    yc = y1 + (y2-y1)/2

    w = x2-x1
    h = y2-y1

    return np.array([xc, yc, w, h])


def clamp(value, between):
    """
    Clamp a value between a range
    
    Parameters
    ----------
    value: an integer/float to be clamped
    between: 1-D Array or Tuple with length 2

    Returns
    -------
    clamped value
    """
    return max(between[0], min(value, between[1]))


def doOverlap(boxA, boxB):
    """
    Checks if two boxes have any overlapping section

    Parameters
    ----------
    boxA, boxB: Bounding box coordinated in the form (x1, y1, x2, y2)

    Returns
    -------
    Bool Flag
    """

    if (boxA[0] > boxB[2] or boxB[0] > boxA[2]):
        return False
    if (boxA[1] > boxB[3] or boxB[1] > boxA[3]):
        return False
    return True


def iou(boxA, boxB):
    """
    Returns the iou of two boxes
    """
    # boxA = boxB = [x1,y1,x2,y2]

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = (xB - xA + 1) * (yB - yA + 1)

    # Compute the area of both rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IOU
    iou = float(intersection_area) / float(boxA_area + boxB_area - intersection_area)
    return iou

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, :: -1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def prep_face(fa, img, rectangle):
    """
    Align face
    fa is a faceAligner object from the imutils library
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_aligned = fa.align(img, gray, rectangle)
    return np.copy(face_aligned)


def prep_crop(bb_tensor, img):
    # First lets make sure that x is in bounds
    # Gets the top_left and bottom_right points from the tensor
    # bb_tensor contains = [?, x, y, x1, y1]
    x, y = (bb_tensor[1].int().item(), bb_tensor[2].int().item())
    x1, y1 = (bb_tensor[3].int().item(), bb_tensor[4].int().item())

    input_height = img.shape[0]
    input_width = img.shape[1]

    # Clamping values
    x = clamp(x, (0, input_width))
    x1 = clamp(x1, (0, input_width))
    y = clamp(y, (0, input_height))
    y1 = clamp(y1, (0, input_height))

    # return np.copy(img[y:y1, x:x1, :])
    return img[y:y1, x:x1, :]

if __name__ == "__main__":
    """
    This main is only for testing purposed and not part of the utilities
    """

    # im_path = "images/can.jpeg"
    # color = cv.imread(im_path)
    # gray = cv.cvtColor(color, cv.COLOR_RGB2GRAY)
    # cannyMap = imutils.auto_canny(gray)
    # cannyMap_t = imutils.translate(cannyMap, 10, 10)
    # cannyMap_transpose = np.transpose(cannyMap)
    # # imshow2("Canny", cannyMap_transpose)
    # rot = []
    # rot.append(cv.cvtColor(gray, cv.COLOR_GRAY2RGB))
    # # rot.append(cannyMap)
    # for angle in range(0, 360, 90):
    #     # rotate the image and display it
    #     rotated = imutils.rotate(gray, angle=angle)
    #     rot.append(cv.cvtColor(rotated, cv.COLOR_GRAY2RGB))
    #     # cv.imshow("Angle=%d" % (angle), rotated)

    # # Refer to https://pyformat.info for formatting
    # print("Showing file {0}, {1:.5}".format(1, im_path))

    # mont = imutils.build_montages(rot, (200, 200), (5, 1))
    # for m, i in zip(mont, range(0, len(mont))):
    #     cv.imshow("Montage %i" % i, mont[i])

    # waitAfterShow()

    streamProcessedWebcam(processFrame)
