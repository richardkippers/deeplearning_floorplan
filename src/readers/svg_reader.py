"""
SVG parser 
@author: r.kippers, 2021 
"""

from readers.floortrans_loaders.house import House
import numpy as np

class CubiCasaSvgReader():

  def __init__(self, svg_path, shape):
    """
    Wrapper around CubiCasa5K functions, to only get the
    functions I need 

    Parameters
    ----------
    svg_path : string
      Path to .svg file 
    shape : tuple
      Corresponding image shape 
    """
    self.svg_path = svg_path 
    self.shape = shape 
    self.house = None

  def read(self):
    """
    Parse svg to memory
    """

    self.house = House(self.svg_path, self.shape[0], self.shape[1])

  def get_walls(self):
    """
    Get walls from house (binary)

    Output
    ------
    ndarray
      Non-zero for wall points. Shape = self.shape 
    """

    walls = self.house.walls.copy() 
    walls[walls > 0] = 1
    return walls

