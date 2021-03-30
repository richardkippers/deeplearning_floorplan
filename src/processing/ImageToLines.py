"""
Postprocessing from binary images to polygons
@author: r.kippers, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import shapes
import cv2
from processing.utils.point_to_line_dist import point_to_line_dist
from itertools import cycle


class ImageToLines():
    
    def __init__(self):
        """
        This function converts binary images to individual walls (polygons)
        """
        # define color cycle
        self.cycol = cycle('bgrcmky')
    
    def morphological_operations(self,image):
        """
        Apply morphological closing, dilation, erosion and canny 
        to input image, which is output of sem. segmentation

        Parameters
        ----------
        image : ndarray
            Array of size 512x512
        """

        kernel = np.ones((15,15),np.uint8)
        closing = cv2.morphologyEx(np.float32(image), cv2.MORPH_CLOSE, kernel)

        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(closing,kernel,iterations = 1)

        kernel = np.ones((6,6),np.uint8)
        erosion = cv2.erode(dilation,kernel,iterations = 1)

        gray = cv2.cvtColor(np.flip(np.repeat(erosion.reshape(512,512,1),3, axis=2),axis=0), cv2.COLOR_BGR2GRAY).astype('uint8')
        edges = cv2.Canny(gray,80,80,apertureSize = 7)

        return edges
    
    def do_hough_transform(self,image):
        """
        Do Hough transform and return lines

        Parameters
        ---------    
        """

        minLineLength = 20
        maxLineGap = 30

        lines = cv2.HoughLinesP(image, rho = 1,theta = 1*np.pi/180, threshold = 20,minLineLength=minLineLength,maxLineGap=maxLineGap)

        return lines 
    
    def do_hough_transform(self,image):
        """
        Do Hough transform and return lines

        Parameters
        ---------    
        """

        minLineLength = 20
        maxLineGap = 30

        lines = cv2.HoughLinesP(image, rho = 1,theta = 1*np.pi/180, threshold = 20,minLineLength=minLineLength,maxLineGap=maxLineGap)

        return lines 

    def calculate_angles(self,lines):
        """
        Calculate angle (degrees) for each line

        Parameters
        ----------
        todo
        """
        angles = []
        for line in lines: 

            x1 = line[0][[0,2]].min()
            x2 = line[0][[0,2]].max()
            y1 = line[0][[1,3]].min()
            y2 = line[0][[1,3]].max()

            dx = x2 - x1 
            dy = y2 - y1 

            # div. by 0 
            if np.round(dx, 0) == 0: 
                angles.append(90)
                continue
            
            theta = np.abs(np.arctan(dy/dx)) * (180/np.pi)

            angles.append(theta)
        
        return angles
    
    def get_to_merge_lines(self,lines):
    
        """
        Decide which lines need to be merged
        based on slope and distance

        Parmeters
        ---------
        todo
        """

        angle_treshhold = 20
        distance_treshold = 30
        to_merge = []
        processed = [False for _ in range(len(lines))]

        angles = self.calculate_angles(lines)

        for i, line in enumerate(lines): 
            x1,y1,x2,y2 = line[0]
            p1 = (x1,y1)
            p2 = (x2,y2)

            # compare with other lines 
            for j, line_j in enumerate(lines):
                if i == j or processed[j]: 
                    continue 
                if np.abs(angles[i] - angles[j]) < angle_treshhold:
                    # calculate distance between lines 
                    x1_j,y1_j,x2_j,y2_j = line_j[0]
                    p3 = (x1_j, y1_j)
                    p4 = (x2_j, y2_j)

                    distances = []
                    distances.append(point_to_line_dist(np.array(p3),np.array([p1,p2])))
                    distances.append(point_to_line_dist(np.array(p4),np.array([p1,p2])))
                    distances.append(point_to_line_dist(np.array(p1),np.array([p3,p4])))
                    distances.append(point_to_line_dist(np.array(p2),np.array([p3,p4])))

                    if min(distances) < distance_treshold:
                        to_merge.append((i,j))

            processed[i] = True 

        return np.array(to_merge)

    def merge_lines_to_polygons(self,lines):
        """

        Create polygons of lines

        Parameters
        ----------
        todo

        Output
        ------
        polygons : list 
        """

        to_merge = self.get_to_merge_lines(lines)

        def find_new_recursive_indices(index,found,depth=0):
            # recursive function to find related indices
            a = to_merge[to_merge[:,0] == index][:,1]
            b = to_merge[to_merge[:,1] == index][:,0]
            to_do = list(np.hstack((a,b)))
            for j in to_do:
                if j not in found: 
                    found.append(j)
                    found = find_new_recursive_indices(j, found, depth+1)
            return found

        # Do
        processed_indices = [False for _ in range(np.max(to_merge)+1)]
        polygons = []

        for l in np.unique(to_merge):
            if processed_indices[l] == True:
                continue

            line_element_indices = find_new_recursive_indices(l, [l])

            # order = x1,y1,x2,y2
            min_x = lines[line_element_indices][:,0][:,[0,2]].min()
            min_y = lines[line_element_indices][:,0][:,[1,3]].min()
            max_x = lines[line_element_indices][:,0][:,[0,2]].max()
            max_y = lines[line_element_indices][:,0][:,[1,3]].max()

            polygons.append([(min_x,min_y),(max_x,min_y), (max_x, max_y), (min_x, max_y)])

            processed_indices[l] = True
            for i in line_element_indices:
                processed_indices[i] = True

        # Create polygons for all other lines 
        for i, line in enumerate(lines):
            x1,y1,x2,y2 = line[0]
            if i not in np.unique(to_merge):
                #continue
                polygons.append([(x1,y1),(x2,y2)])#,color=next(self.cycol)))

        return polygons
