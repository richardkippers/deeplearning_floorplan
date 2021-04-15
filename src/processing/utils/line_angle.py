"""
Function to calculate the angle between two points
@author: r.kippers, 2021
"""
import numpy as np 

def calculate_angle(p1,p2):
    x1 = min(p1[0], p2[0])
    y1 = min(p1[1], p2[1])
    x2 = max(p1[0], p2[0])
    y2 = max(p1[1], p2[1])
    
    dx = x2 - x1 
    dy = y2 - y1 
    
    # div. by 0 
    if np.round(dx, 0) == 0: 
        return 90
    
    theta = np.abs(np.arctan(dy/dx)) * (180/np.pi)
    return theta
