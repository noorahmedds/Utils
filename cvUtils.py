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

def find_dominant_color(img, verbose = False):
    """Finds Dominant color from an image. Returns palette and the dominant RGB value for that image"""

    # Average
    # average = img.mean(axis=0).mean(axis=0)

    pixels = np.float32(img.reshape(-1, 3))
    n_colors = 5
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 500, .05) #the final param is epsilon. It is the minimum amount the centroid has to move before the update is terminated
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

def imshow2(im_name, image, location = [0, 0], resize = Point2D(x=350, y=600)):
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
        if verbose: print(wait)
        if wait & 0xFF == ord(key):
            cv.destroyAllWindows()
            return

def processFrame(image):
    return imutils.auto_canny(image)

def streamProcessedVideo(videoPath, processFrame, willProcess = True, showFPS = True):
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

        imshow2('frame',out)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # FPS
        if showFPS:
            end = time.time()
            seconds = end-start
            fps = (fps + (1/seconds))/2
            cv2.putText(out, "{0} fps".format(fps), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            print("{0} fps".format(fps))
            start = end

    cap.release()
    cv.destroyAllWindows()


def streamProcessedWebcam(processFrame, willProcess = True, showFPS = True):
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
            cv2.putText(out, "{0} fps".format(fps), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            print("{0} fps".format(fps))
            start = end

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


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
