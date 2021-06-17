"""
Postprocessing from binary images to polygons
@author: r.kippers, 2021
"""

import numpy as np

import cv2
from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection

from processing.utils.line_angle import calculate_angle

class ImageToWalls():
    
    def __init__(self, image_size=512, kernel_size=5):
        """
        This function converts binary images to individual walls (polygons)
        """
        self.image_size=image_size
        self.kernel_size = kernel_size
    
    def do_steps(self, image_prediction):
        """
        Convert image_prediction to Shapely Polygons

        Parameters
        ----------
        image_prediction : ndarray
            Binary mask, 1 = wall, 0 else 
        
        Output
        ------
        walls 
        """

        morph_result = self.morphology_steps(image_prediction)

        linestrings = self.get_contours_as_linestring(morph_result)
        linestrings = self.fix_lines_angles(linestrings,25)
        
        wall_polys = self.get_base_poly(linestrings)
        
        return self.all_walls_poly_to_walls(wall_polys)


    def morphology_steps(self, image):
        
        """
        OpenCV Morphology Steps for 
        
        Parameters
        ----------
        image : ndarray
            Input image
            
        Output
        ------
        ndarray
            512*512 binary image
        """
        
        kernel = np.ones((self.kernel_size,self.kernel_size),np.uint8)
        closing = cv2.morphologyEx(np.float32(image), cv2.MORPH_CLOSE, kernel)

        dilation = cv2.dilate(closing,kernel,iterations = 1)

        closing2 = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

        erosion = cv2.erode(closing2,kernel,iterations = 1)

        dilation2 = cv2.dilate(erosion,np.ones((2,2)),iterations = 3)

        gray = cv2.cvtColor(np.flip(np.repeat(dilation2.reshape(self.image_size,self.image_size,1),3, axis=2),axis=0), cv2.COLOR_BGR2GRAY).astype('uint8')

        gray = np.flip(gray, axis=0)
        
        return gray

    def get_contours_as_linestring(self, image):
        """
        Get contours from binary image
        
        Parameters
        ----------
        image : ndarray
            Binary image
        
        Output
        ------
        List of linestrings
        """
        
        # Find contours 
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        linestrings = []
        
        for i,contour in enumerate(contours):
            
            epsilon = 5 # pixels
            approx = cv2.approxPolyDP(contour,epsilon,True)

            x = list(approx[:,0,0])
            y = list(approx[:,0,1])

            xy = np.array((x,y))
            linestrings.append(xy.T)
        
        return linestrings

    def fix_lines_angles(self, linestrings, deviation):
        """
        If line is close to horizontal or vertical: make them
        
        Parameters, Output
        ------------------
        linestrings : list 
            List of linestrings
        deviation : float 
            Threshold, deviation in degrees
        """

        for linestring_i, linestring in enumerate(linestrings):

            fixed_x, fixed_y = [False for _ in linestring], [False for _ in linestring]
            
            for i,_ in enumerate(linestring):

                i_next = 0 if  i == len(linestring)-1 else i+1

                linestring = linestrings[linestring_i]

                this_point = linestring[i,:]
                next_point= linestring[i_next,:]

                angle = calculate_angle(this_point, next_point)
                
                # 0 degrees fix : average on y-axis
                if 0 < angle < deviation:

                    if fixed_y[i]: 
                        fixed_y[i_next] = True
                        linestrings[linestring_i][i_next,1] = linestrings[linestring_i][i,1]
                    else: 
                        fixed_y[i] = fixed_y[i_next] = True
                        linestrings[linestring_i][[i,i_next],1] = linestrings[linestring_i][[i,i_next],1].mean()

                # 90 degrees fix -> average on x-axis
                if 90-deviation < angle <= 90: 
                    if fixed_x[i]:
                        fixed_y[i_next] = True
                        linestrings[linestring_i][i_next,0] = linestrings[linestring_i][i,0]
                    else: 
                        fixed_x[i] = fixed_x[i_next] = True
                        linestrings[linestring_i][[i,i_next],0] = linestrings[linestring_i][[i,i_next],0].mean()
                
        return linestrings

    def get_base_poly(self, linestrings):
        """
        Create large area from linestrings that represent
        the contour polygons 

        Parameters
        ----------
        linestrings : list 
            List of linestrings
        """
        
        # Find largest area
        largest_area = 0
        largest_area_i = -1
        for i,linestring in enumerate(linestrings):
            if len(linestring) < 3: 
                # skip TODO check
                continue
            p = Polygon(linestring)
            if p.area > largest_area: 
                largest_area_i = i
                largest_area = p.area

        # Remove all areas from largest area
        base_poly = Polygon(linestrings[largest_area_i])
        base_poly.simplify(tolerance=1)
        
        if not base_poly.is_valid:
            base_poly = base_poly.buffer(0)
        
        for i,linestring in enumerate(linestrings):
            if i == largest_area_i or len(linestring) < 3:
                continue 
            p = Polygon(linestring)
            
            if not base_poly.is_valid:
                base_poly = base_poly.buffer(0)
                
            if not p.is_valid:
                p = p.buffer(0)
            
            base_poly = base_poly.symmetric_difference(p)

        return base_poly

    def find_top_left_point_in_linestring(self, linestrings):
        """
        Find top left point in linestrings by first finding top 
        y, then finding the most left point for this y
        
        Parameters
        ----------
        linestrings : list
            All linestrings
            
        Output
        ------
        point : tuple
        
        """
        max_y = -np.inf 
        min_x_for_y = np.inf

        for linestring in linestrings:
            linestring_top_y = np.array(linestring.coords)[:,1].max()
            max_y = linestring_top_y if linestring_top_y > max_y else max_y

        for linestring in linestrings:
            points = np.array(linestring.coords)
            points = points[points[:,1] == max_y]
            if points.shape[0] > 0:
                linestring_min_x = points[:,0].min()
                min_x_for_y = linestring_min_x

        return (min_x_for_y, max_y)

    def find_max_x_delta_and_highest_angle(self, linestrings, top_left):
        """
        Find line to point with highest angle 
        and find line to point with largest dx

        Parameters
        ----------
        linestrings : list
            Linestrings
        top_left : tuple 
            Top left point 

        Output
        ------
        point_with_highest_angle : list
        max_x_delta_point : list
        """

        point_with_highest_angle = None
        max_x_delta_point = None
        highest_angle = -1
        max_x_delta = -1

        for linestring in linestrings:
            i = None
            points = np.array(linestring.coords)
            # find index
            for j,point in enumerate(points): 
                if tuple(point) == top_left:
                    i = j
                    break

            if i != None:
                prev_point = points[i-1] if i != 0 else points[points.shape[0]-2]
                next_point = points[i+1] if i != points.shape[0]-1 else points[0]

                # define point with highest angle
                angle_prev = calculate_angle(top_left, prev_point)
                if angle_prev > highest_angle: 
                    point_with_highest_angle = list(prev_point)
                    highest_angle = angle_prev
                angle_next = calculate_angle(top_left, next_point)
                if angle_next > highest_angle: 
                    point_with_highest_angle = list(next_point)
                    highest_angle = angle_next

                # set point with largest x delta 
                if np.abs(top_left[0] - prev_point[0]) > max_x_delta: 
                    max_x_delta = np.abs(top_left[0] - prev_point[0])
                    max_x_delta_point = prev_point

                if np.abs(top_left[0] - next_point[0]) > max_x_delta: 
                    max_x_delta_point = next_point
                    max_x_delta = np.abs(top_left[0] - next_point[0])
        
        return point_with_highest_angle, max_x_delta_point

    def all_walls_poly_to_walls(self, all_walls_poly):
        
        """
        Convert a large polygon representing all walls
        to individual wall elements
        
        Parameters
        ----------
        all_walls_poly : Shapely object
            Polygon or other Shapely object
        
        Output
        ------
        walls : list
            List containing walls
        """
        
        if type(all_walls_poly) != Polygon:
            max_poly = None
            for poly in all_walls_poly:
                if max_poly is None or max_poly.area < poly.area: 
                    max_poly = poly
            all_walls_poly = max_poly
        
        # Remove unnecessary points on lines by simplifying
        all_walls_poly = all_walls_poly.simplify(tolerance=0.4)
        
        walls = [] # result

        for _iterator in range(int(1e5)): # range to prevent infinite things
            
            if all_walls_poly.area < 0.1:
                break
            
            linestrings = all_walls_poly.boundary
            if type(linestrings) == LineString:
                linestrings = [linestrings]

            # Get top left point
            top_left = self.find_top_left_point_in_linestring(linestrings)

            # Get max x_delta and highest angle points
            point_with_highest_angle, max_x_delta_point = self.find_max_x_delta_and_highest_angle(linestrings,top_left)

            # Define wall
            wall = None
            
            # Create line with same angle, move x (to left and to right, stop at first intersection) 
            # direction depends on the other point's x coordinate
            direction = -1 if point_with_highest_angle[0] < top_left[0] else 1 

            for x_indent in range(1,512):
                
                x_indent = x_indent * direction

                top_y = top_left[1]
                top_x = top_left[0] + x_indent
                bottom_y =  point_with_highest_angle[1]
                bottom_x =  point_with_highest_angle[0] + x_indent

                move_poly = Polygon((  
                    (point_with_highest_angle[0], point_with_highest_angle[1]),
                    (top_left[0], top_left[1]),
                    (top_x, top_y),
                    (bottom_x,bottom_y),
                    (point_with_highest_angle[0], point_with_highest_angle[1])
                ))
                
                if not move_poly.is_valid: 
                    move_poly = move_poly.buffer(0)

                if np.abs(all_walls_poly.intersection(move_poly).area - move_poly.area) > 0.01:
                    x_indent-= (1*direction) # set 1 back
                    break
            
            # Sucesfully found new wall, generate the new wall
            top_y = top_left[1]
            top_x = top_left[0] 
            bottom_y =  point_with_highest_angle[1]
            bottom_x =  point_with_highest_angle[0] 
            wall = Polygon((
                (bottom_x, bottom_y), 
                (top_x, top_y), 
                (top_x + x_indent, top_y), 
                (bottom_x+x_indent,bottom_y)
            ))

            if wall.area < 0.5:
                # Not possibble to create wall, 
                # new wall is intersection for bbox around 
                # extreme points around + top_left (pois)

                pois = np.array((
                    top_left,
                    point_with_highest_angle,
                    max_x_delta_point
                ))
                
                bbox = Polygon((
                    (pois[:,0].min(), pois[:,1].min()),
                    (pois[:,0].min(), pois[:,1].max()),
                    (pois[:,0].max(), pois[:,1].max()),
                    (pois[:,0].max(), pois[:,1].min()),
                    (pois[:,0].min(), pois[:,1].min())
                ))
                
                wall = all_walls_poly.intersection(bbox)
                
                if type(wall) == GeometryCollection or type(wall) == MultiPolygon:
                    # Don't combine objects, so only pick largest area
                    max_wall_area = 0
                    max_wall_area_i = 0
                    for o_i, o in enumerate(wall): 
                        if o.area > max_wall_area: 
                            max_wall_area_i = o_i
                            max_wall_area = o.area
                    wall = wall[max_wall_area_i]
                
            

            all_walls_poly = all_walls_poly.symmetric_difference(wall)
            wall_unsimplified = wall
            wall = wall.simplify(2,preserve_topology=False)
            
            if wall.is_empty:
                # Simplifying made polygon empty
                wall = wall_unsimplified
            
            walls.append(wall)
            
        
        return walls

        