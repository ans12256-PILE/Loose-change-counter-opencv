#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:59:41 2020

@author: alexey_imac
"""

#  following along with basic openCV turorial at
# https://www.youtube.com/watch?v=Z78zbnLlPUA&feature=youtu.be&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq

# Duplicate defoinition of opencv in the error below after import cv2
# refer search "Class RunLoopModeTracker is implemented in both "
# for example
#

# objc[92655]: Class RunLoopModeTracker is implemented in both
# /Users/alexey_imac/opt/anaconda3/lib/libQt5Core.5.9.7.dylib (0x103876a80)
# and
# /Users/alexey_imac/opt/anaconda3/lib/python3.8/site-packages/cv2/.dylibs/QtCore (0x1219cb7f0).
# One of the two will be used. Which one is undefined.
#

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline #does not work in VS Code

# # image name    0 - "0" designates gray scale - faster processing
# img=cv2.imread('/Users/alexey_imac/Documents/CLOUD_GENIUS_CLASSES/PYTHON CLASS/capstone_coins.png',cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(
#     '/Users/alexey_imac/Documents/CLOUD_GENIUS_CLASSES/PYTHON CLASS/capstone_coins.png')
# # turns out CV2 places channels in BGR order, while matplotlib keeps them in RGB order
# # to make sure images loaded by cv2, and displayed by matplotlib do not look
# # "bluish", conversion has to be made as follows:
# # https://stackoverflow.com/questions/39316447/opencv-giving-wrong-color-to-colored-images-on-loading
# plt.figure(1)
# plt.imshow(img) # colors messed up due to the default swap of BGR / RGB

# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # cv2.imshow('image_window_name', img)
# # cv2.waitKey(0)        # wait for any key to be pressed INSIDE the window
# # cv2.destroyAllWindows()

# # plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.figure(2)
# plt.imshow(RGB_img)     # works - shows normal colors
# # plt.show() shows all figures
# # cv2.imshow('Original image',RGB_img)   hangs Jupiter Notebook

# # convert image to gray
# img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plt.figure(3)
# # https: // matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.imshow.html
# # Note: For actually displaying a grayscale image set up
# # the color mapping using the parameters cmap='gray', vmin=0, vmax=255.
# # for whatever reason messes up previous plt..imshow
# plt.imshow(img_g, cmap='gray', vmin=0, vmax=255)

# plt.plot([100,500],[500, 200],'r',linewidth=5)

# # cv2.imwrite('temp_cv2_imwrite.png', img)

# # edge detection using Canny filter to get contours
# edges = cv2.Canny(img_g, 33, 78)
# # display edges
# plt.figure(4)
# plt.imshow(edges, cmap='gray', vmin=0, vmax=255)
# plt.show()

# # https://stackoverflow.com/questions/25125670/best-value-for-threshold-in-canny
# # interactively selecting threshods for Canny


# def callback(x):
#     print(x)


# canny = cv2.Canny(img_g, 85, 255)


# cv2.namedWindow('image')  # make a window with name 'image'
# # lower threshold trackbar for window 'image
# cv2.createTrackbar('L', 'image', 0, 255, callback)
# # upper threshold trackbar for window 'image
# cv2.createTrackbar('U', 'image', 0, 255, callback)

# while(1):
#     numpy_horizontal_concat = np.concatenate(
#         (img_g, canny), axis=1)  # to display image side by side
#     # cv2.imshow('image', numpy_horizontal_concat)
#     # too large, rescale per
#     # https://stackoverflow.com/questions/58405119/how-to-resize-the-window-obtained-from-cv2-imshow
#     h, w = numpy_horizontal_concat.shape[0:2]  # h=890, w=1920*2=3840
#     h, w = 445, 1920
#     numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat, (w, h))
#     cv2.imshow('image', numpy_horizontal_concat)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:  # escape key
#         break
#     l = cv2.getTrackbarPos('L', 'image')
#     u = cv2.getTrackbarPos('U', 'image')

#     canny = cv2.Canny(img_g, l, u)

# cv2.destroyAllWindows()

# https://docs.opencv.org/3.4.12/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
# find circles in the image
# Finds circles in a grayscale image using the Hough transform.
# The function finds circles in a grayscale image using a modification of the Hough transform.
# circles	=	cv.HoughCircles(	image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]	)
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html

# cv2.IMREAD_GRAYSCALE	This flag is used to return the image in grayscale format.
# Alternatively, we can pass integer value 0 for this flag.
img = cv2.imread(
    '/Users/alexey_imac/Documents/CLOUD_GENIUS_CLASSES/PYTHON CLASS/capstone_coins.png', 0)

# Color image in cv2 format of BGR
BGR_img = cv2.imread(
    '/Users/alexey_imac/Documents/CLOUD_GENIUS_CLASSES/PYTHON CLASS/capstone_coins.png', 1)

# does not work just a pointer
# BGR_img_copy = BGR_img   # make a copy since original will be contaminated with text and lines
BGR_img_copy = cv2.imread(
    '/Users/alexey_imac/Documents/CLOUD_GENIUS_CLASSES/PYTHON CLASS/capstone_coins.png', 1)

# RGB_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
# plt.figure(1)
# plt.imshow(RGB_img)  # for matplotlib swap of BGR / RGB is required
# plt.show()

# edge detection using Canny filter to get contours
# img = cv2.Canny(img, 33, 78) # HoughCircles uses Canny edge
# https://docs.opencv.org/3.4.12/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
# 	dst	=	cv.medianBlur(	src, ksize[, dst]	)
#   ksize	aperture linear size;
# it must be odd and greater than 1, for example: 3, 5, 7 ...
# changed from 5 to
img = cv2.medianBlur(img, 9)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cimg=img
# https://docs.opencv.org/3.4.12/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
#                            param1=50, param2=30, minRadius=50, maxRadius=150)
#  circles	=	cv.HoughCircles(	image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]	)
# Changes:
# param1 (higher of Canny filters) changed from 50 to 78
# minDist changed from 20 to 160 = 2*minRadius
# param2 (the smaller it is, the more false circles ...) changed from 30 to 31
# to remove false circle
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 160,
                           param1=78, param2=57, minRadius=80, maxRadius=140)

# convert float to integer for whole pixels
circles = np.uint16(np.around(circles))

# draw circles, centers and add text
# https: // docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
font = cv2.FONT_HERSHEY_SIMPLEX
# for i in circles[0, :]:
#     # draw the outer circle
#     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     # draw the center of the circle
#     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
#     # no new line possible in putText
#     # https://stackoverflow.com/questions/27647424/opencv-puttext-new-line-character
#     cv2.putText(cimg, f'x,y={i[0]},{i[1]}', (i[0], i[1]), font, 1,
#                 (255, 255, 255), 2, cv2.LINE_AA)
#     # add separate r call out at 45 degrees i[2]/2**0.5
#     # x--> left->right, y top--> bottom ==> i[1]-diag
#     diag = int(i[2]/2**0.5)
#     cv2.putText(cimg, f'r={i[2]}', (i[0]+diag, i[1]-diag), font, 1,
#                 (255, 255, 255), 2, cv2.LINE_AA)
#     # annotation of parameters
#     cv2.putText(cimg, 'img = cv2.medianBlur(img, 5)', (1400, 120), font, 1,
#                 (255, 255, 255), 2, cv2.LINE_AA)
#     circles_param1 = 'circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 160, '
#     circles_param2 = '160, param1=78, param2=31, minRadius=80, maxRadius=150)'
#     cv2.putText(cimg, circles_param1, (850, 40), font, 1,
#                 (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(cimg, circles_param2, (850, 80), font, 1,
#                 (255, 255, 255), 2, cv2.LINE_AA)
j = 0   # circle number
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(BGR_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(BGR_img, (i[0], i[1]), 2, (0, 0, 255), 3)
    # no new line possible in putText
    # https://stackoverflow.com/questions/27647424/opencv-puttext-new-line-character
    cv2.putText(BGR_img, f'x,y={i[0]},{i[1]}', (i[0], i[1]), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    # add circle ID r/2 above center like Cr#j; int() required because or r/2 otherwise float, error
    cv2.putText(BGR_img, f'Cr#{j}', (i[0]-i[2], i[1]), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    j += 1    # increase circle ID counter
    # add separate r call out at 45 degrees i[2]/2**0.5
    # x--> left->right, y top--> bottom ==> i[1]-diag
    diag = int(i[2]/2**0.5)
    cv2.putText(BGR_img, f'r={i[2]}', (i[0]+diag, i[1]+diag), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

# annotation of parameters OUTSIDE the loop
cv2.putText(BGR_img, 'img = cv2.medianBlur(img, 9)', (1400, 120), font, 1,
            (255, 255, 255), 2, cv2.LINE_AA)
circles_param1 = 'circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 160,'
circles_param2 = 'param1=78, param2=57, minRadius=80, maxRadius=140)'
cv2.putText(BGR_img, circles_param1, (850, 40), font, 1,
            (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(BGR_img, circles_param2, (850, 80), font, 1,
            (255, 255, 255), 2, cv2.LINE_AA)


# cv2.imshow('detected circles', cimg)
# for cv2 original BGR image is required
cv2.imshow('detected circles', BGR_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# circles: vector of [x,y,radius]
# check output
print(circles)
# understanding structure of circles
# circles[0,:] - whole circle ?
# circles[0,:] - whole thing
# circles[0,0] - [1679  507  132] x,y,r of the first detected circle
# circle[0,0,0] - 1679 x coordinate of the center of the first detected circle
print(circles[0, 0, 0])

# Looked up British coins
# Ref. https://en.wikipedia.org/wiki/Coins_of_the_pound_sterling#Currently_circulating_coinage
# Two types of "copper" coins - 1 penny OD 20.3mm, 2 pence OD 25.9mm
# Two types of "silver" coins 5 pence OD 18.0mm, 10 pence OD 24.5mm
# Two types of 7 sided poligon "silver" coins
# 20 pence OD 21.4mm. 50 pence OD 27.3mm
# Once circular coins are separated by colod, there would be a binary choice by diameter
# Average color of the circular area of the image:
#
# The process appears to be a two-step process:
# Create a mask around each circle using info found in "circles", namely [x,y,r], and create
# a black mask around to remove color info other than area of interest
# calculate average color in the masked image,
# figure out if it is more "copperish" or "silverish". Maybe color diagram? 8-o
# There seems to be a function for circular mask creation named drawContours, let's see
# try for one circle
# tmp_mask = cv2.drawContours(BGR_img, circles)
# Dec. 16, 2020
# maskes - let's start simple - rectangular mask
# let's try first coin using results found in circles[0,:] meaning 1st(0) circle [x,y,r]
# range of x ==? x+-r; range of y==> y+-r, where
coin_i_BGR = []  # initialize a list of BGR values per coin
Nth = 0   # coin counter
for k in circles[0, :]:  # process colors inside of of each circle found

    x = k[0]
    y = k[1]
    r = k[2]
    print(x, y, r)   # 1679 507 132 2*r=264
# is it possible index ranges can not use expressions? Can they use variables ?
# coin_0 = BGR_img[x - r:x + r, y - r:y + r] # - seems array is empty
# coin_0 = BGR_img[x-r:x+r, y-r:y+r]
# for manual try x-r=1679-132=1547; x+r=1679+132=1811; y-r=507-132=375; y+r=507+132=639
# coin_0 = BGR_img[x-r:x+r, y-r:y+r]   # manual try
# manual try same result - 0 rows
# https: // www.docs.opencv.org / master / d3 / df2 / tutorial_py_basic_ops.html
# Image ROI - error was caused by the fact that image indices are listed as [y,x]
# and original backwards placing was causing out of data on y range
# why
# coin_0 = BGR_img[1547:1811, 375:639]
# BGR_img_copy[y-+r, x-+r]
    # re-assign for each step of the loop
    coin_i = BGR_img_copy[y-r:y+r, x-r:x+r]

    # now to modify out of circle pixels to [0, 0, 0] black
    for i in range(2*r):
        for j in range(2*r):
            # check if outside circle (x-r)^2+(y-r)^2>r^2 ==> assign black [0,0,0]
            if (i-r) ** 2 + (j-r) ** 2 > r**2:
                coin_i[i, j] = [0, 0, 0]  # mind you coin_0[y,x]

# seems to be working, now average color - "copper" or "silver" ?
# "Large copper" - 2 pence, "Small copper" - 1 penny
# "Large silver" - 10 pence, "Small silver" - 5 pence
# color analysis and average
# now every pixel in coin_0[0:2*r, 0:2*r] [y,x] has a (for BGR image) [B,G,R] values from 0~255
# coin_0[r,r] = [B,G,R]  average ? ... short of manual loop?
# REF: https: // stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
    average_color_per_row = np.average(coin_i, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    print(average_color, type(average_color))
    # accumulate color in list
    # parse average color into tuple (B,G,R)
    # https://www.tutorialspoint.com/numpy/numpy_iterating_over_array.htm
    tmp = []
    for color_component in average_color:
        # .item() is required to convert types from numpy.float64 to python native
        tmp.append(int(color_component.item()))
        # color component: 68.25352961432503, data type < class 'numpy.float64' >
        # print(f'color component: {color_component}, data type {type(color_component)}')
    # convert list to tuple
    # average_color_list = tuple(tmp)
    # coin_i_BGR.append(average_color_tuple)
    coin_i_BGR.append(tmp)
    print(f'Accumulated BGR values of coin_i_BGR:  {coin_i_BGR}')
    # display RECTANGULAR subimage
    # cv2.imshow(f'{Nth} detected coin', coin_i)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()   # errors ?
    # tuple number of rows, columns, and channels (0, 264, 3) 0 rows ?!
    print(coin_i.shape)
    print(coin_i.dtype)  # uint8 -seems to be OK WTF ?!
# plt.figure(1)
# plt.imshow(coin_0)  # for matplotlib swap of BGR / RGB is required
# plt.show()
# display average color
    # coin_i[:,:] = average_color
    # cv2.imshow(f'{Nth} detected coin average color', coin_i)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  # errors ?

    Nth += 1   # increase coin counter


# Bar plot since Histogram does not work - distribution
# ref: https://izziswift.com/how-to-plot-a-histogram-using-matplotlib-in-python-with-a-list-of-data/

print(f'list of BGR values of the coins = {coin_i_BGR}')
# check list parsing
print(f'extract last coin R channel value should be 115: ? {coin_i_BGR[7][2]}')
# x = np.arange(3)
# plt.bar(x, height=average_color)
# plt.xticks(x, ['B', 'G', 'R'])
# # show plot
# plt.show()

# color detection
# === step 1  even out brightness by normalizing to the total
# BGR = B/sum(BGR), G/sum(BGR), R/sum(BGR)
# === step 2  first differential B-min(BGR), G-min(BGR), R-min(BGR)
# turns out it illiminates blue as the weakestB=0
# === step 3 2nd differential R-G
# shameless analysis in Excel revealed that for "silver" coins
# differential was 0.8 and 2.5, while for "copper" coins
# it ranged from 10 to 16.43

# let's assume we know the threshold of "silver"-->"copper" as being 5
# how to automatically determine that for more verieties of colors
# is to be left for the next assignment
coin_color = []   # initialize list to store "copper" / "silver" decision
copper_index = []  # save indices of copper coins while we are at it
silver_index = []  # save indices of silver coins while we are at it
index_counter = 0
for i in coin_i_BGR:  # 8 triple tuples
    # normalize brightness per step 1
    tmp_total = sum(i)
    # print(f'sum of the channels {tmp_total}')
    # tuples are unmutable can not use in calculations
    i[0] = int(i[0] / tmp_total*100)
    i[1] = int(i[1] / tmp_total*100)
    i[2] = int(i[2] / tmp_total*100)  # normalized brightness
    # first differential -min() per step 2
    tmp_min = min(i)
    i[0] = i[0] - tmp_min
    i[1] = i[1] - tmp_min
    i[2] = i[2] - tmp_min
    # assuming we know the next channel is Green
    # perform 2nd differential per step 3 R=G, and determine color
    # if differential is above 5 => "copper", append list with colors
    if i[2] - i[1] > 5:
        coin_color.append("copper")
        copper_index.append(index_counter)
    else:
        coin_color.append("silver")
        silver_index.append(index_counter)

    index_counter += 1

print(f'Coins color detection results are as follows coin_color: {coin_color}')
print(f'Indices of copper coins are : {copper_index}')
print(f'Indices of silver coins are : {silver_index}')
# collect and compare diameters in color categories to
# make a binary choice for "copper" 1 penny small, 2 pence large
# and for "silver" 5 pence small, 10 pence large

# extract all radii
radii = circles[0, :, 2]
print(f'detected radii values are radii[]:  {radii}')  # works

# extract list of diameters in one shot
# ref https://stackoverflow.com/questions/22412509/getting-a-sublist-of-a-python-list-with-the-given-indices
# You can use list comprehension to get that list:

# a = [0, 1, 2, 3, 4, 5, 6]
# b = [0, 2, 4, 5]
# c = [a[index] for index in b]
# print c
# [0, 2, 4, 5]
copper_radii = [radii[index] for index in copper_index]
silver_radii = [radii[index] for index in silver_index]
print(f'copper radii are copper_radii:  {copper_radii}')
print(f'silver radii are copper_radii:  {silver_radii}')


total_change = 0  # And now about something completely different ...
# :o)

# copper denominations determination
# 1 penny has OD 20.3 mm, 2 pense has OD 25.9
# therefore 1 penny is 100*20.3/25.9 = 78.37% of the 2 pence size
# plus OD approximation error say max OD - 10% - half of the difference
# would be a determination threshold
copper_threshold = 0.9 * max(copper_radii)
# count change
# memorize denominations
copper_denominations = []
for i in copper_radii:
    if i > copper_threshold:  # found 2 pence, add 2
        total_change += 2
        copper_denominations.append('2 pence')
    else:  # must be a penny, add 1
        total_change += 1
        copper_denominations.append('1 penny')

# silver denominations determination
# 5 pence has OD 18.0 mm, 10 pense has OD 24.5
# therefore 5 pence is 100*18.0/24.5 = 73.46% of the 10 pence size
# plus OD approximation error say max OD - 13% - half of the difference
# would be a determination threshold
silver_threshold = 0.87 * max(copper_radii)
# count change
# memorize denominations
silver_denominations = []
for i in silver_radii:
    if i > silver_threshold:  # found 10 pence, add 10
        total_change += 10
        silver_denominations.append('10 pence')
    else:  # must be 5 pence, add 5
        total_change += 5
        silver_denominations.append('5 pence')

print(f'Total change in the picture {total_change}')
# now have to "glue/splice" list of denominations back together in
# a single list

print(f'len(radii) = {len(radii)}')
denominations = [None]*len(radii)  # empty list of denominations
print(f'Denominations initialization:  {denominations}')
# populate silver denominations
i = 0  # for steps thorough silver_denominations
for index in silver_index:
    denominations[index] = silver_denominations[i]
    i += 1

# populate copper denominations
i = 0  # for steps thorough copper_denominations
for index in copper_index:
    denominations[index] = copper_denominations[i]
    i += 1

print(f'Denominations initialization:  {denominations}')


# add total to figure
j = 0   # circle number
for i in circles[0, :]:
    # add separate r call out at 45 degrees i[2]/2**0.5
    # x--> left->right, y top--> bottom ==> i[1]-diag
    diag = int(i[2]/2**0.5*0.6)  # 0.6 to avoid text overlap on Cr#3
    cv2.putText(BGR_img, f'{denominations[j]}', (i[0]-diag, i[1]+diag), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    j += 1    # cycle to the next circle index

cv2.putText(BGR_img, f'Total change in the picture {total_change}', (1411, 800), font, 1,
            (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow('detected circles and total', BGR_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
