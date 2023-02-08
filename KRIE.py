"""
This file contains all the funcitons for creating an image plus with projected lines of a predefined height from radar data.
The height can either be predefined or calculated by the radar elevation field of view.
This file has been completely reworked on 2019-01-23 for best functionalities. Some function arguments changed, so please verify if you referr to this file.

"""

# Standard libraries
import os
import os.path as osp
import sys
import math
import time

# 3rd party libraries
import cv2
import json
import numpy as np
from pyquaternion import Quaternion
from PIL import Image

# Local modules
# Allow relative imports when being executed as script.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    # import crfnet.raw_data_fusion  # noqa: F401
    __package__ = "crfnet.raw_data_fusion"

from nuscenes.utils.data_classes import PointCloud
from crfnet.utils import radar
# from nuscenes.utils.geometry_utils import view_points


def _resize_image(image_data, target_shape):

    stupid_confusing_cv2_size_because_width_and_height_are_in_wrong_order = (target_shape[1], target_shape[0])
    resized_image = cv2.resize(image_data, stupid_confusing_cv2_size_because_width_and_height_are_in_wrong_order)
    resize_matrix = np.eye(3, dtype=resized_image.dtype)
    resize_matrix[1, 1] = target_shape[0]/image_data.shape[0]
    resize_matrix[0, 0] = target_shape[1]/image_data.shape[1]

    # image_data:(900,1600,3)
    # target_shape:(450, 450)
    # resized_image:(450,450,3)
    return resized_image, resize_matrix


def _radar_transformation(radar_data, height=None):


    # Field of view (global)
    ELEVATION_FOV_SR = 20
    ELEVATION_FOV_FR = 14

    # initialization
    num_points = radar_data.shape[1]

    # Radar points for the endpoint
    radar_xyz_endpoint = radar_data[0:3,:].copy()


    RADAR_HEIGHT = 0.5



    # sigma =1.2*(radar_data[18, :] - np.min(radar_data[18,:])) / (np.max(radar_data[18,:])-np.min(radar_data[18,:]))
    #assert radar_data[0, :].size != 0
    if radar_data[0, :].size != 0:
        r = np.max(abs(radar_data[0, :]))
    
        if r !=0:
            #print('d_std:',d_std)
            
            sigma = 1.5*(radar_data[0, :]-np.mean(radar_data[0, :]))/r
            sigma = sigma+0.3
        else:
            #print('d_std:',d_std)
            sigma = 0.5
           
        if height:
            if num_points !=0:
                radar_data[2, :] = np.ones((num_points,)) * (sigma) # lower points
                radar_xyz_endpoint[2, :] = np.ones((num_points,)) * (sigma) # upper points
            else:
                pass

        else:
            dist = radar_data[-1,:]
            count = 0
            for d in dist:
            # short range mode
                if d <= 70:
                    radar_xyz_endpoint[2, count] = -d * np.tan(ELEVATION_FOV_SR/2)
            # long range mode
                else:
                    radar_xyz_endpoint[2, count] = -d * np.tan(ELEVATION_FOV_FR/2)

                count += 1

        return radar_data, radar_xyz_endpoint
    else:
        print('111111111111111111111111111111')
        pass



def _create_vertical_line(P1, P2, img):


    imageH = img.shape[0]
    imageW = img.shape[1]
    # imageH=imageW=450


    N = P1[24]
    P1_y = int(P1[1])
    P2_y = int(P2[1])
    if P1[1] is not np.nan:
        P1_y = int(P1[1])
    else:
        pass
    if P2[1] is not np.nan:
        P2_y = int(P2[1])
    else:
        pass
    if N is not np.nan:
        N = N
    else:
        N = 0.5

    P1_y = P1_y-int(P1[23] *(1-N))
    P2_y = P2_y+int(P1[23] *N)


    dX = 0
    dY = P2_y - P1_y
    # P2_Y=92 P1_y=393
    if dY == 0:
        dY = 1
    dXa = np.abs(dX)
    dYa = np.abs(dY)


    itbuffer = np.empty(
        shape=(np.maximum(int(dYa), int(dXa)), 2), dtype=np.float32)
    itbuffer.fill(np.nan)

    # vertical line segment
    itbuffer[:, 0] = int(P1[0])
    if P1_y > P2_y:
        # Obtain coordinates along the line using a form of Bresenham's algorithm
        itbuffer[:, 1] = np.arange(P1_y - 1, P1_y - dYa - 1, -1)
    else:
        itbuffer[:, 1] = np.arange(P1_y+1, P1_y+dYa+1)

    # Remove points outside of image
    colX = itbuffer[:, 0].astype(int)
    colY = itbuffer[:, 1].astype(int)
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) &
                        (colX < imageW) & (colY < imageH)]
    # print('itbuffer:', itbuffer)
    # img:(450, 450)  itbuffer:(301,2)
    return itbuffer

def KRIE(image_data, radar_data, radar_xyz_endpoints, clear_radar=False):


    radar_meta_count = radar_data.shape[0]-3
    # radar_meta_count=18
    # radar_data:(450,450,3)
    radar_extension = np.zeros(
        (image_data.shape[0], image_data.shape[1], radar_meta_count), dtype=np.float32)
    # radar_extension:(450,450,18)
    no_of_points = radar_data.shape[1]
    # no_of_points=30

    if clear_radar:
        pass # we just don't add it to the image
    else:
        for radar_point in range(0, no_of_points):
            # radar_point=29
            projection_line = _create_vertical_line(
                radar_data[:, radar_point], radar_xyz_endpoints[0:2, radar_point], image_data)
            # projection_line:(20,2)
            # radar_xyz_endpoints:(3,30)


            scale = 20 * radar_data[-1, radar_point]

            s = int(scale)

            for i in range(0, s):

                for pixel_point in range(0, projection_line.shape[0]):

                    y = projection_line[pixel_point, 1].astype(int)
                    x = projection_line[pixel_point, 0].astype(int) + i
                    # y=232,x=382
                    if y < 360 and x < 640 and x>0 and y>0:

                        if not np.any(radar_extension[y, x]) or radar_data[-1, radar_point] < radar_extension[y, x, -1]:
                            radar_extension[y, x] = radar_data[3:, radar_point]
                        else:
                            pass
                    else:
                        pass


    image_plus = np.concatenate((image_data, radar_extension), axis=2)
    # image_plus:(450,450,21)
    # print('radar_data:', radar_data)

    return image_plus

