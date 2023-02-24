import math
import cv2
import numpy as np
import skimage.morphology as morph
import matplotlib.pyplot as plt

DEBUG = False
LETTERS_DIR = "E:\\2021-wagony-final (1)\\wagony\\letters"



def preprocess(filename_core, catalog_root):
    # wczytanie danych i metadanych
    image = cv2.imread(f'{catalog_root}\\{filename_core}.jpeg')
    # exit foo() on failed image read
    if type(image).__name__ == "NoneType":
        return
    # crop 10% of borders, which can be safely performed without loss of significant data
    # img_crop = image[int(image.shape[0]*0.1):int(image.shape[0]*0.9), int(image.shape[1]*0.1):int(image.shape[1]*0.9)]
    with open(f'{catalog_root}\\{filename_core}.txt', "r") as f:
        file_data = f.read()
    # TRY CONTOUR:
    image_gray = cv2.imread(f'{catalog_root}\\{filename_core}.jpeg', flags=cv2.IMREAD_GRAYSCALE)
    # image_gray = image_gray[int(image_gray.shape[0] * 0.1):int(image_gray.shape[0] * 0.9), int(image_gray.shape[1] * 0.1):int(image_gray.shape[1] * 0.9)]

    # # OPTIONALLY blur the image
    # cv2.GaussianBlur(src=image_gray, dst=image_gray, ksize=(5,5), sigmaX=0, sigmaY=0)
    # threshold the img
    foo = lambda x: 255 if x > 200 else 0
    food = np.vectorize(foo)
    image_gray = np.array(food(image_gray), dtype=np.uint8)
    r = np.array(food(image[:,:,0]), dtype=np.uint8)
    g = np.array(food(image[:,:,1]), dtype=np.uint8)
    b = np.array(food(image[:,:,2]), dtype=np.uint8)
    # perform closing: dilation followed by erosion with kernels of equal size
    morph.binary_closing(image_gray, morph.disk(3), out=image_gray)
    morph.binary_closing(r, morph.disk(3), out=r)
    morph.binary_closing(g, morph.disk(3), out=g)
    morph.binary_closing(b, morph.disk(3), out=b)
    # perform opening: erosion followed by dilation with kernels of equal size
    morph.binary_opening(image_gray, morph.disk(3), out=image_gray)
    morph.binary_opening(r, morph.disk(3), out=g)
    morph.binary_opening(g, morph.disk(3), out=g)
    morph.binary_opening(b, morph.disk(3), out=b)
    # scale image values since preceeding lines yield a matrix containig 1s and 0s
    image_gray = image_gray*255
    r = r*255
    g = g*255
    b = b*255
    if DEBUG:
        cv2.imshow("image", image_gray)

    lower_cont_area_boundary = 600
    upper_cont_area_boundary = 6000
    contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found_contours = []
    moments = []
    for contour in contours:
        c_area = cv2.contourArea(contour)
        if lower_cont_area_boundary < c_area < upper_cont_area_boundary:
            found_contours.append(contour)
            mom = cv2.moments(contour)
            cx = int(mom['m10'] / mom['m00'])
            cy = int(mom['m01'] / mom['m00'])

    if DEBUG:
        tmp_img = copy.deepcopy(image_gray)
        cv2.drawContours(tmp_img, found_contours, -1, (100, 100, 100), 8)
        cv2.imshow("ALL CONTOURS", tmp_img)
        del tmp_img
    # calculate centers for found objects(contours)
    # calculate global center of mass
    # calculate distances
    # drop further half of objects
    x_centers = []
    y_centers = []
    dists = []
    for index, element in enumerate(found_contours):
        x_centers.append(np.average(found_contours[index][:, :, 0]))
        y_centers.append(np.average(found_contours[index][:, :, 1]))
        # found_contours[index] = np.expand_dims(found_contours[index], axis=3)
    center = (np.average(x_centers), np.average(y_centers))
    # looks like magic number, but this is the value that placed into the equation effectively approaches value of 0.5
    weight_factor_max = 50000
    for x, y, cont in zip(x_centers, y_centers, found_contours):
        # calculate centered weight
        _weight_x = 0.5 - (1 / math.sqrt(2 * math.pi)) * (math.e ** (-0.5 * (1/weight_factor_max)*((image_gray.shape[0]/2)-x)**2))
        _weight_y = 0.5 - (1 / math.sqrt(2 * math.pi)) * (math.e ** (-0.5 * (1/weight_factor_max)*((image_gray.shape[1]/2)-y)**2))
        _dist = math.sqrt((x - center[0])*(x - center[0]) + (y - center[1])*(y - center[1]))
        # calculate Pythagorean distance
        dists.append(_dist*_weight_x*_weight_y)
        # dists.append(_dist)
    _dists = sorted(list(enumerate(dists)), key=lambda x: x[1])
    # find and drop elements furthest from center

    cnt = max((len(found_contours)/2)-1, 30)
    topop = []
    for index, dist in _dists:
        if dist > 450:
            topop.append(index)
    # while cnt > 0:
    #     max = -1
    #     max_dist = -1
    #     for index, dist in enumerate(dists):
    #         # flatten last dimension to get the appropriate distance value
    #         if index not in topop and dist > max_dist:
    #             max_dist = dist
    #             max = index
    #     topop.append(max)
    #     cnt = cnt - 1
    topop = sorted(topop, reverse=True)
    for i in topop:
        found_contours.pop(i)

    crop_x1 = image_gray.shape[0]
    crop_x2 = 0
    crop_y1 = image_gray.shape[1]
    crop_y2 = 0
    for contour in found_contours:
        if np.min(contour[:,:,1]) < crop_x1:
            crop_x1 = np.min(contour[:,:,1])
        if np.max(contour[:,:,1]) > crop_x2:
            crop_x2 = np.max(contour[:,:,1])
        if np.min(contour[:,:,0]) < crop_y1:
            crop_y1 = np.min(contour[:,:,0])
        if np.max(contour[:,:,0]) > crop_y2:
            crop_y2 = np.max(contour[:,:,0])
        # cv2.imwrite(f"{LETTERS_DIR}\\{np.sum(contour)}.png", image_gray[np.min(contour[:,:,1]):np.max(contour[:,:,1]), np.min(contour[:,:,0]):np.max(contour[:,:,0])])

    image_gray = image_gray[min(crop_x1,crop_x2):max(crop_x1,crop_x2),min(crop_y1,crop_y2):max(crop_y1,crop_y2)]
    r = r[min(crop_x1,crop_x2):max(crop_x1,crop_x2),min(crop_y1,crop_y2):max(crop_y1,crop_y2)]
    g = g[min(crop_x1,crop_x2):max(crop_x1,crop_x2),min(crop_y1,crop_y2):max(crop_y1,crop_y2)]
    b = b[min(crop_x1,crop_x2):max(crop_x1,crop_x2),min(crop_y1,crop_y2):max(crop_y1,crop_y2)]

    # TRY ADAPTIVE THRESHOLDING
    # image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 11)

    # TEMPORARILY shrink dims of found_contours to allow drawing
    # np.squeeze(found_contours, axis=3)
    if DEBUG:
        cv2.drawContours(image_gray, found_contours, -1, (100, 100, 100), 8)
        cv2.imshow("REDUCED CONTOURS", image_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # # store channels separately,
    # # CONSIDER: change dtype to faciliatate further calculations on those arrays
    # # r = np.array(img_crop[:, :, 0], dtype=np.uint32)
    # r = np.array(img_crop[:, :, 0])
    # g = np.array(img_crop[:, :, 1])
    # b = np.array(img_crop[:, :, 2])
    # r_otsu, r_thresh = otsu_apply(r)
    # g_otsu, g_thresh = otsu_apply(g)
    # b_otsu, b_thresh = otsu_apply(b)

    # nakreślenie histogrmów częstotliwości dla każdego z kanałów koloru
    # fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1)
    # ax0.hist(img_crop[:, :, 0].ravel(), 255, [0, 255], color='blue')
    # ax0.vlines(r_thresh, 0, 100000)
    # ax1.hist(img_crop[:, :, 1].ravel(), 255, [0, 255], color='green')
    # ax1.vlines(g_thresh, 0, 40000)
    # ax2.hist(img_crop[:, :, 2].ravel(), 255, [0, 255], color='red')
    # ax2.vlines(b_thresh, 0, 25000)
    # fig.tight_layout()
    # fig.set_size_inches(2.5, 7.5)
    # plt.show()
    # if DEBUG:
    #     cv2.imshow("Red thresholded", r_otsu)
    #     cv2.imshow("Green thresholded", g_otsu)
    #     cv2.imshow("Blue thresholded", b_otsu)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # cv2.imwrite(f'{catalog_root}\\{filename_core}_R.jpeg', r)
    # cv2.imwrite(f'{catalog_root}\\{filename_core}_G.jpeg', g)
    # cv2.imwrite(f'{catalog_root}\\{filename_core}_B.jpeg', b)
    # cv2.imwrite(f'{catalog_root}\\{filename_core}_gray.jpeg', image_gray)

