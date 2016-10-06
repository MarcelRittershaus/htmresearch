# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Actual implementation of physical objects.

Note that because locations are integers, rather large object sizes should be
used.
"""

import random
from math import pi, cos, sin, sqrt

import numpy as np
import matplotlib.pyplot as plt

from htmresearch.frameworks.layers.physical_object_base import PhysicalObject



class Sphere(PhysicalObject):
  """
  A classic sphere.

  It's particularity is that it has only one feature.

  Example:
    sphere = Sphere(radius=20, dimension=3, epsilon=1)

  It's only feature is its surface.

  """

  _FEATURES = ["surface"]

  def __init__(self, radius, dimension=3, epsilon=None):
    """
    The only key parameter to provide is the sphere's radius.

    Supports arbitrary dimensions.

    Parameters:
    ----------------------------
    @param    radius (int)
              Sphere radius.

    @param    dimension (int)
              Space dimension. Typically 3.

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON

    """
    self.radius = radius
    self.dimension = dimension

    if epsilon is None:
      self.epsilon = self.DEFAULT_EPSILON
    else:
      self.epsilon = epsilon


  def getFeatureID(self, location):
    """
    Returns the feature index associated with the provided location.

    In the case of a sphere, it is always the same if the location is valid.
    """
    if not self.contains(location):
      return self.EMPTY_FEATURE

    return self.SPHERICAL_SURFACE


  def contains(self, location):
    """
    Checks that the provided point is on the sphere.
    """
    return self.almostEqual(
      sum([coord ** 2 for coord in location]), self.radius ** 2
    )


  def sampleLocation(self):
    """
    Samples from the only available feature.
    """
    return self.sampleLocationFromFeature(self._FEATURES[0])


  def sampleLocationFromFeature(self, feature):
    """
    Samples a location from the provided specific feature.

    In the case of a sphere, there is only one feature.
    """
    if feature == "surface":
      coordinates = [random.gauss(0, 1.) for _ in xrange(self.dimension)]
      norm = sqrt(sum([coord ** 2 for coord in coordinates]))
      return [self.radius * coord / norm for coord in coordinates]
    elif feature == "random":
      return self.sampleLocation()
    else:
      raise NameError("No such feature in {}: {}".format(self, feature))


  def plot(self, numPoints=100):
    """
    Specific plotting method for cylinders.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # generate sphere
    phi, theta = np.meshgrid(
      np.linspace(0, pi, numPoints),
      np.linspace(0, 2 * pi, numPoints)
    )
    x = self.radius * np.sin(phi) * np.cos(theta)
    y = self.radius * np.sin(phi) * np.sin(theta)
    z = self.radius * np.cos(phi)

    # plot
    ax.plot_surface(x, y, z, alpha=0.2, rstride=20, cstride=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("{}".format(self))
    return fig, ax


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(R={})"
    return template.format(self.radius)



class Cylinder(PhysicalObject):
  """
  A classic cylinder.

  Example:
    cyl = Cylinder(height=20, radius=5, epsilon=1)

  It has five different features to sample locations from: topDisc, bottomDisc,
  topEdge, bottomEdge, and side.
  """

  _FEATURES = ["topDisc", "bottomDisc", "topEdge", "bottomEdge", "side"]


  def __init__(self, height, radius, epsilon=None):
    """
    The two key parameters are height and radius.

    Does not support arbitrary dimensions.

    Parameters:
    ----------------------------
    @param    height (int)
              Cylinder height.

    @param    radius (int)
              Cylinder radius.

    @param    dimension (int)
              Space dimension. Typically 3.

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON

    """
    self.radius = radius
    self.height = height
    self.dimension = 3  # no choice for cylinder dimension

    if epsilon is None:
      self.epsilon = self.DEFAULT_EPSILON
    else:
      self.epsilon = epsilon


  def getFeatureID(self, location):
    """
    Returns the feature index associated with the provided location.

    Three possibilities:
      - top discs
      - edges
      - lateral surface

    """
    if not self.contains(location):
      return self.EMPTY_FEATURE

    if self.almostEqual(abs(location[2]), self.height / 2.):
      if self.almostEqual(location[0] ** 2 + location[1] ** 2,
                          self.radius ** 2):
        return self.CYLINDER_EDGE
      else:
        return self.FLAT
    else:
      return self.CYLINDER_SURFACE


  def contains(self, location):
    """
    Checks that the provided point is on the cylinder.
    """
    if self.almostEqual(location[0] ** 2 + location[1] ** 2, self.radius ** 2):
      return abs(location[2]) < self.height / 2.
    if self.almostEqual(location[2], self.height / 2.):
      return location[0] ** 2 + location[1] ** 2 < self.radius ** 2
    return False


  def sampleLocation(self):
    """
    Simple method to sample uniformly from a cylinder.
    """
    areaRatio = self.radius / (self.radius + self.height)
    if random.random() < areaRatio:
      return self._sampleLocationOnDisc()
    else:
      return self._sampleLocationOnSide()


  def sampleLocationFromFeature(self, feature):
    """
    Samples a location from the provided specific features.
    """
    if feature == "topDisc":
      return self._sampleLocationOnDisc(top=True)
    elif feature == "topEdge":
      return self._sampleLocationOnEdge(top=True)
    elif feature == "bottomDisc":
      return self._sampleLocationOnDisc(top=False)
    elif feature == "bottomEdge":
      return self._sampleLocationOnEdge(top=False)
    elif feature == "side":
      return self._sampleLocationOnSide()
    elif feature == "random":
      return self.sampleLocation()
    else:
      raise NameError("No such feature in {}: {}".format(self, feature))


  def _sampleLocationOnDisc(self, top=None):
    """
    Helper method to sample from the top and bottom discs of a cylinder.

    If top is set to True, samples only from top disc. If top is set to False,
    samples only from bottom disc. If not set (defaults to None), samples from
    both discs.
    """
    if top is None:
      z = random.choice([-1, 1]) * self.height / 2.
    else:
      z = self.height / 2. if top else - self.height / 2.
    sampledAngle = 2 * random.random() * pi
    sampledRadius = self.radius * sqrt(random.random())
    x, y = sampledRadius * cos(sampledAngle), sampledRadius * sin(sampledAngle)
    return [x, y, z]


  def _sampleLocationOnEdge(self, top=None):
    """
    Helper method to sample from the top and bottom edges of a cylinder.

    If top is set to True, samples only from top edge. If top is set to False,
    samples only from bottom edge. If not set (defaults to None), samples from
    both edges.
    """
    if top is None:
      z = random.choice([-1, 1]) * self.height / 2.
    else:
      z = self.height / 2. if top else - self.height / 2.
    sampledAngle = 2 * random.random() * pi
    x, y = self.radius * cos(sampledAngle), self.radius * sin(sampledAngle)
    return [x, y, z]


  def _sampleLocationOnSide(self):
    """
    Helper method to sample from the lateral surface of a cylinder.
    """
    z = random.uniform(-1, 1) * self.height / 2.
    sampledAngle = 2 * random.random() * pi
    x, y = self.radius * cos(sampledAngle), self.radius * sin(sampledAngle)
    return [x, y, z]


  def plot(self, numPoints=100):
    """
    Specific plotting method for cylinders.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # generate cylinder
    x = np.linspace(- self.radius, self.radius, numPoints)
    z = np.linspace(- self.height / 2., self.height / 2., numPoints)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(self.radius ** 2 - Xc ** 2)

    # plot
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=20, cstride=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("{}".format(self))
    return fig, ax


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(H={}, R={})"
    return template.format(self.height, self.radius)



class Box(PhysicalObject):
  """
  A box is a classic cuboid.

  Example:
    box = Box(dimensions=[10, 10, 5], dimension=3, epsilon=1)

  It has three features to sample locations from: face, edge, and vertex.
  """

  _FEATURES = ["face", "edge", "vertex"]


  def __init__(self, dimensions, dimension=3, epsilon=None):
    """
    The only key parameter is the list (or tuple) of dimensions, which can be
    of any size as long as its length is equal to the "dimension" parameter.

    Parameters:
    ----------------------------
    @param    dimensions (list(int))
              List of the box's dimensions.

    @param    dimension (int)
              Space dimension. Typically 3.

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON

    """
    self.dimensions = dimensions
    self.dimension = dimension

    if epsilon is None:
      self.epsilon = self.DEFAULT_EPSILON
    else:
      self.epsilon = epsilon


  def getFeatureID(self, location):
    """
    There are three possible features for a box:
      - flat for a face
      - pointy for a vertex
      - edge

    """
    if not self.contains(location):
      return self.EMPTY_FEATURE  # random feature

    numFaces = sum(
      [self.almostEqual(abs(coord), self.dimensions[i] / 2.) \
       for i, coord in enumerate(location)]
    )

    if numFaces == 1:
      return self.FLAT
    if numFaces == 2:
      return self.EDGE
    if numFaces == 3:
      return self.POINTY


  def contains(self, location):
    """
    A location is on the box if one of the dimension is "satured").
    """
    for i, coord in enumerate(location):
      if self.almostEqual(abs(coord), self.dimensions[i] / 2.):
        return True
    return False


  def sampleLocation(self):
    """
    Random sampling fron any location corresponds to sampling form the faces.
    """
    return self._sampleFromFaces()


  def sampleLocationFromFeature(self, feature):
    """
    Samples a location from one specific feature.

    This is only supported with three dimensions.
    """
    if feature == "face":
      return self._sampleFromFaces()
    elif feature == "edge":
      return self._sampleFromEdges()
    elif feature == "vertex":
      return self._sampleFromVertices()
    elif feature == "random":
      return self.sampleLocation()
    else:
      raise NameError("No such feature in {}: {}".format(self, feature))


  def _sampleFromFaces(self):
    """
    We start by sampling a dimension to "max out", then sample the sign and
    the other dimensions' values.
    """
    coordinates = [random.uniform(-1, 1) * dim / 2. for dim in self.dimensions]
    dim = random.choice(range(self.dimension))
    coordinates[dim] = self.dimensions[dim] / 2. * random.choice([-1, 1])
    return coordinates


  def _sampleFromEdges(self):
    """
    We start by sampling dimensions to "max out", then sample the sign and
    the other dimensions' values.
    """
    dimensionsToChooseFrom = range(self.dimension)
    random.shuffle(dimensionsToChooseFrom)
    dim1, dim2 = dimensionsToChooseFrom[0], dimensionsToChooseFrom[1]
    coordinates = [random.uniform(-1, 1) * dim / 2. for dim in self.dimensions]
    coordinates[dim1] = self.dimensions[dim1] / 2. * random.choice([-1, 1])
    coordinates[dim2] = self.dimensions[dim2] / 2. * random.choice([-1, 1])
    return coordinates


  def _sampleFromVertices(self):
    """
    We start by sampling dimensions to "max out", then sample the sign and
    the other dimensions' values.
    """
    coordinates = [
      self.dimensions[0] / 2. * random.choice([-1, 1]),
      self.dimensions[1] / 2. * random.choice([-1, 1]),
      self.dimensions[2] / 2. * random.choice([-1, 1]),
    ]
    return coordinates

  def plot(self, numPoints=100):
    """
    Specific plotting method for boxes.

    Only supports 3-dimensional objects.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # generate cylinder
    x = np.linspace(- self.dimensions[0]/2., self.dimensions[0]/2., numPoints)
    y = np.linspace(- self.dimensions[1]/2., self.dimensions[1]/2., numPoints)
    z = np.linspace(- self.dimensions[2]/2., self.dimensions[2]/2., numPoints)

    # plot
    Xc, Yc = np.meshgrid(x, y)
    ax.plot_surface(Xc, Yc, -self.dimensions[2]/2,
                    alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(Xc, Yc, self.dimensions[2]/2,
                    alpha=0.2, rstride=20, cstride=10)
    Yc, Zc = np.meshgrid(y, z)
    ax.plot_surface(-self.dimensions[0]/2, Yc, Zc,
                    alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(self.dimensions[0]/2, Yc, Zc,
                    alpha=0.2, rstride=20, cstride=10)
    Xc, Zc = np.meshgrid(x, z)
    ax.plot_surface(Xc, -self.dimensions[1]/2, Zc,
                    alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(Xc, self.dimensions[1]/2, Zc,
                    alpha=0.2, rstride=20, cstride=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("{}".format(self))
    return fig, ax


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(dim=({}))"
    return template.format(", ".join([str(dim) for dim in self.dimensions]))



class Cube(Box):
  """
  A cube is a particular box where all dimensions have equal length.


  Example:
    cube = Cube(width=100, dimension=3, epsilon=2)
  """


  def __init__(self, width, dimension=3, epsilon=None):
    """
    We simply pass the width as every dimension.

    Parameters:
    ----------------------------
    @param    width (int)
              Cube width.

    @param    dimension (int)
              Space dimension. Typically 3.

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON

    """
    self.width = width
    dimensions = [width] * dimension
    super(Cube, self).__init__(dimensions=dimensions,
                               dimension=dimension,
                               epsilon=epsilon)


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(width={})"
    return template.format(self.width)
