"""
Get objects from Tensorflow Detections API
@author: r.kippers, 2021
"""

import numpy as np
from shapely.geometry import Polygon

def process_openings_from_detections(detections, walls, image_size):
    """
    Process openings from detections
    
    Parameters
    ----------
    detections : dict
        Tensorflow Object detection API results
    walls : list
        List with Shapely objects
    image_size : float
        Input image size
    
    Output
    ------
    dict
        Objects
    
    """
    openings = []

    for i in range(detections['detection_boxes'].shape[1]):
        score = detections['detection_scores'].numpy()[0,i]
        if score >= 0.3:

            target_class = detections['detection_classes'].numpy()[0][i]

            bbox = detections['detection_boxes'].numpy()[0,i,:]

            # use var image_size to scale 
            y1 = bbox[0] * image_size
            x1 = bbox[1] * image_size
            y2 = bbox[2] * image_size 
            x2 = bbox[3] * image_size

            opening_box = Polygon(((x1, y1), (x1,y2), (x2,y2), (x2,y1), (x1,y1)))

            # find wall with largest intersection
            related_wall = None
            related_wall_overlap = 0
            for wall in walls: 
                if related_wall_overlap < wall.intersection(opening_box).area:
                    related_wall_overlap = wall.intersection(opening_box).area
                    related_wall = wall

            if related_wall != None:

                opening_poly = related_wall.intersection(opening_box)
                opening_poly_center = opening_poly.centroid
                opening_center = opening_box.centroid
                dx = opening_center.xy[0][0] - opening_poly_center.xy[0][0]
                dy = opening_center.xy[1][0] - opening_poly_center.xy[1][0]

                # direction: top, right, bottom, left
                if np.abs(dx) < np.abs(dy):
                    direction = 2 if dx > 0 else 0
                else: 
                    direction = 3 if dy > 0 else 1


                openings.append({
                    "polygon": opening_poly,
                    "direction": None if max(np.abs(dx), np.abs(dy)) < 3 else direction,
                    "type": "window" if target_class == 0 else "door"
                })

    return openings       