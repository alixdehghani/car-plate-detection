import numpy
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import glob
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches




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

    #detect car plate from erosion image that made in last morphology actions

    plate = erosion
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(plate, cmap="gray")

    #this for loop detect all rectangle in image which is area is bigger than 50 pixel
    # regionprops creates a list of properties of all the labelled regions
    label_image = measure.label(plate)
    for region in regionprops(label_image):

        if region.area < 50:
            #if the region is so small then it's likely not a license plate
            continue
        
        # the bounding box coordinates
        minRow, minCol, maxRow, maxCol = region.bbox
        rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
            # let's draw a red rectangle over those regions
    plt.show() 


    # in this section we want to find best rectangle that is similar to car plate

    #plate_dimensions is Percentage of the image that Determines a Approximaten of car plate size
    plate_dimensions = (0.05*label_image.shape[0], 0.3*label_image.shape[0], 0.1*label_image.shape[1], 0.5*label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    plate_objects_cordinates = []
    plate_like_objects = []
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(plate, cmap="gray")

    #minrow, mincol, maxrow, maxcol show the car plate position in the orgianl image at the end of for loop
    minrow=0
    mincol=0
    maxrow=0
    maxcol=0
    label_image = measure.label(plate)
    for region in regionprops(label_image):
        if region.area < 100:
            #if the region is so small then it's likely not a license plate
            continue

        # the bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        # ensuring that the region identified satisfies the condition of a typical license plate
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            minrow=min_row
            mincol=min_col
            maxrow=maxRow
            maxcol=max_col
            plate_like_objects.append(binary_car_image[min_row:max_row,
                                        min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                                    max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
            # let's draw a red rectangle over those regions

    plt.show()
    
    #this kernel is a matrix of zeros that is same size of orginal image
    kernel = numpy.zeros((480,640),numpy.uint8)
    kernel[minrow:maxrow+10,mincol:maxcol+10]=1

    #Multiplication of gray_im and kernel show just car plate
    plate_gray= gray_im*kernel
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(plate_gray, cmap="gray")
    plt.show()

