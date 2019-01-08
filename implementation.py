import numpy
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import glob
from skimage.filters import threshold_otsu



dir_files = glob.glob("*.jpg")
for img in dir_files:
    # read orginal image
    im = scipy.misc.imread(img)

    # change type to int32
    im = im.astype('int32')

    # plot orginal image
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)

    # split RGB fron orginal image
    try:
        b, g, r = cv2.split(im)
    except:
        continue

    # change to gray scale by this formula
    gray_im = (3*r+6*g+b)/10

    # show gray_im
    ax1.set_title(img)
    ax1.imshow(gray_im, cmap="gray")

    # apply sobel filter in gray_im
    dx = ndimage.sobel(gray_im, 0)  # horizontal derivative
    dy = ndimage.sobel(gray_im, 1)  # vertical derivative
    mag = numpy.hypot(dx, dy)  # magnitude
    mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
    scipy.misc.imsave('sobel.jpg', mag)
    # mag is resualt of sobel filter
    ax2.set_title("Sobel Filtered")
    ax2.imshow(mag.astype(int), cmap="gray")

    # apply otsu threshold
        #it just a threshold that help to segmentation image into two part

    #change gray level fron 0,255 into 0,1   
    mag = mag/255
    threshold_value = threshold_otsu(mag)
    binary_car_image = mag > threshold_value
    # show gray_im whith threshold
    ax3.set_title("otsu thresold")
    ax3.imshow(binary_car_image, cmap="gray")


    #apply some morphology actions

    #dilation for binary_car_image
        #dialation make highlight white lines in binary_car_image 
    kernel = numpy.ones((11,11),numpy.uint8)
    dilation = cv2.dilate(binary_car_image.astype(float),kernel,iterations = 1)
    ax4.set_title("dilation")
    ax4.imshow(dilation, cmap="gray")


    #closing for binary_car_image
        #closing improve white segment in binary_car_image 
    kernel = numpy.ones((11,11),numpy.uint8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    ax5.set_title("closing")
    ax5.imshow(closing, cmap="gray")

    #erosion for binary_car_image
        #erosion is a easy way to remove noise but it is'n a good way 
    kernel = numpy.ones((35,35),numpy.uint8)
    erosion = cv2.erode(closing,kernel,iterations = 1)
    ax6.set_title("erosion")
    ax6.imshow(erosion, cmap="gray")

    plt.show()
