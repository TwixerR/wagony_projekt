import math
import io
import cv2
import numpy as np
# import skimage.morphology as morph
import copy
import net
import tensorflow as tf

DEBUG = False

LETTERS_DIR = "E:\\2021-wagony-final (1)\\wagony\\letters"
CONFIDENCE_THRESHOLD = 80
LOWER_CONT_AREA_LIMIT = 600
UPPER_CONT_AREA_LIMIT = 6000

def raw_response(resp):
    resp = [x for x in resp if x[1] > CONFIDENCE_THRESHOLD]
    resp = sorted(resp, key=lambda x: x[2][0])
    lines = []
    text = ""
    while len(resp) > 0:
        midpoint_vertical = int((resp[0][2][0] + resp[0][2][1]) / 2)
        height = resp[0][2][1] - resp[0][2][0]
        # assemble list of objects in same line
        line = [x for x in resp if midpoint_vertical in range(x[2][0], x[2][1])]
        # sort by x1 coordinate
        line = sorted(line, key=lambda x: x[2][2])
        for item in line:
            # remove from all_items_list
            resp.remove(item)
            # build text line
            text = text + item[0]
        # add separator on line break
        text = text + '-'
    # remove last dash character
    text = text[:-1]
    # remove underscores from small letters
    text = "".join(text.split('_'))
    return text

def format_response(resp):
    # get raw parsed response
    text = raw_response(resp)
    # cut off all false reads of 'Eoas' section
    if len(text.split('E')) > 1:
        left = "".join(text.split('E')[:-1])
        right = text.split('E')
    # discard all characters except for the numbers
    text = [x for x in text  if ord(x) in range(ord('0'), ord('9'))]
    # assemble single string from list of chars
    text = "".join(text)
    return text

def extract(filepath, catalog_root, model=None):
    # read img
    img = cv2.imread(f'{catalog_root}\\{filepath}.jpeg', flags=cv2.IMREAD_GRAYSCALE)
    # exit foo() on failed image read
    if type(img).__name__ == "NoneType":
        return
    if DEBUG:
        cv2.imshow("img", img)
    # binarize the image
    foo = lambda x: 255 if x > 200 else 0
    food = np.vectorize(foo)
    img = np.array(food(img), dtype=np.uint8)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found_contours = []
    moments = []
    for contour in contours:
        c_area = cv2.contourArea(contour)
        if LOWER_CONT_AREA_LIMIT < c_area < UPPER_CONT_AREA_LIMIT:
            found_contours.append(contour)
            mom = cv2.moments(contour)
            cx = int(mom['m10'] / mom['m00'])
            cy = int(mom['m01'] / mom['m00'])
    if DEBUG:
        tmp_img = cv2.imread(f'{catalog_root}\\{filepath}.jpeg')
        cv2.drawContours(tmp_img, found_contours, -1, (100, 100, 100), 8)
    extracted_tups = []
    for contour in found_contours:
        y1, y2, x1, x2 = np.min(contour[:,0,0]), np.max(contour[:,0,0]), np.min(contour[:,0,1]), np.max(contour[:,0,1])
        sub_img = np.zeros(shape=(img[x1:x2,y1:y2].shape), dtype=img.dtype)
        # get contour without position in img offset for drawing purpose
        pred_class, accuracy = net.predict(img[x1:x2,y1:y2], model)
        extracted_tups.append((pred_class, accuracy, (x1,x2,y1,y2)))
        if DEBUG:
            # cv2.imshow("cropped", img[x1:x2,y1:y2])
            # cv2.putText(tmp_img2, "{}:{:.2f}".format(pred_class, accuracy), (y2,x1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255))
            if accuracy > CONFIDENCE_THRESHOLD:
                cv2.putText(tmp_img, "{}".format(pred_class), (y1,x1), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100,255,0))
            else:
                cv2.putText(tmp_img, "{}".format(pred_class), (y1, x1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255))

    if DEBUG:
        cv2.imshow("LETTERS AND PREDS", tmp_img[200:,200:])
        del tmp_img

    return format_response(extracted_tups)

def extract_flask(file, model=None):
    memory_file = io.BytesIO()
    file.save(memory_file)
    data = np.fromstring(memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE)
    if model is None:
        model = tf.keras.models.load_model(net.BEST_MODEL_PATH)
    foo = lambda x: 255 if x > 200 else 0
    food = np.vectorize(foo)
    img = np.array(food(img), dtype=np.uint8)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found_contours = []
    moments = []
    for contour in contours:
        c_area = cv2.contourArea(contour)
        if LOWER_CONT_AREA_LIMIT < c_area < UPPER_CONT_AREA_LIMIT:
            found_contours.append(contour)
            mom = cv2.moments(contour)
            cx = int(mom['m10'] / mom['m00'])
            cy = int(mom['m01'] / mom['m00'])
    extracted_tups = []
    for contour in found_contours:
        y1, y2, x1, x2 = np.min(contour[:, 0, 0]), np.max(contour[:, 0, 0]), np.min(contour[:, 0, 1]), np.max(
            contour[:, 0, 1])
        pred_class, accuracy = net.predict(img[x1:x2, y1:y2], model)
        extracted_tups.append((pred_class, accuracy, (x1, x2, y1, y2)))
    confident_tups = [x for x in extracted_tups if x[1] > CONFIDENCE_THRESHOLD]
    return format_response(confident_tups)