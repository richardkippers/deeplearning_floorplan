"""
Get objects from Tensorflow Detections API
@author: r.kippers, 2021
"""

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.affinity import scale 

def process_openings_from_detections(detections, walls, image_size, score_th=0.4):
    
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
    score_th : float
        Score treshhold
    
    Output
    ------
    dict
        Objects
    
    """
    openings = []

    for i in range(detections['detection_boxes'].shape[1]):
        score = detections['detection_scores'].numpy()[0,i]
        if score >= score_th:

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
            related_wall_j = -1
            for j,wall in enumerate(walls): 
                if related_wall_overlap < wall.intersection(opening_box).area:
                    related_wall_overlap = wall.intersection(opening_box).area
                    related_wall = wall
                    related_wall_j = j

            if related_wall != None:
                
                # Make opening fit nice on wall
                exterior =  zip(related_wall.exterior.xy[0], related_wall.exterior.xy[1])
                exterior = [x for x in exterior]
                largest_intersection_edge = None
                largest_intersection_length = 0
                for vertice_i in range(len(exterior)-1):
                    edge = LineString((exterior[vertice_i], exterior[vertice_i+1]))
                    intersection = edge.intersection(opening_box).length
                    if intersection > largest_intersection_length:
                        largest_intersection_length = intersection
                        largest_intersection_edge = edge 
                if largest_intersection_edge != None:
                    edge_dx = abs(largest_intersection_edge.xy[0][0]-largest_intersection_edge.xy[0][1])
                    edge_dy = abs(largest_intersection_edge.xy[1][0]-largest_intersection_edge.xy[1][1])
                    if edge_dx < edge_dy:
                        opening_box = scale(opening_box, xfact=5)
                    else: 
                        opening_box = scale(opening_box, yfact=5)

                opening_poly = related_wall.intersection(opening_box)
                
                # Find direction
                opening_poly_center = opening_poly.centroid
                opening_center = opening_box.centroid
                dx = opening_center.xy[0][0] - opening_poly_center.xy[0][0]
                dy = opening_center.xy[1][0] - opening_poly_center.xy[1][0]

                # direction: top, right, bottom, left
                if np.abs(dx) < np.abs(dy):
                    direction = 2 if dx > 0 else 0
                else: 
                    direction = 3 if dy > 0 else 1
                    
                direction = None if target_class == 0 else direction
        
                openings.append({
                    "polygon": opening_poly,
                    "direction": None if max(np.abs(dx), np.abs(dy)) < 3 else direction,
                    "type": "window" if target_class == 0 else "door",
                    "related_wall_i": related_wall_j
                })

    return openings       