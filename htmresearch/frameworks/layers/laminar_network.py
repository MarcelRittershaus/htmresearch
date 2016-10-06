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
The methods here contain factories to create networks of multiple layers
and for experimenting with different laminar structures.

The first network type supported, "L4L2Column", is a single cortical column
containing and L4 and an L2 layer. L4 gets two inputs and feeds into L2. The L2
column feeds back to L4.

             L2Column  <------|
               ^  |           |
               |  |           |
               |  v           |
        --->  L4Column <------|
        |          ^          |
        |          |        reset
        |          |          |
externalInput  sensorInput -->|

The second network type supported, "MultipleL4L2Columns", allows you to create
N L4L2 columns with the above structure. In this case the L2 columns will also
be laterally connected to one another (each one receives input from all other
columns.)

                ----------------------------------
                |                                |
                v                                v
             L2Column  <------|               L2Column  <------|
               ^  |           |                 ^  |           |
               |  |           |                 |  |           |
               |  v           |                 |  v           |
        --->  L4Column <------|          --->  L4Column <------|
        |          ^          |          |          ^          |
        |          |        reset        |          |        reset
        |          |          |          |          |          |
externalInput  sensorInput -->|  externalInput  sensorInput -->|


For both network types, regions will be named as shown above plus a suffix
indicating column number, such as "externalInput_0", "L2Column_3", etc. The
reset signal from sensorInput is sent to the other regions.

Also, how do you like my ascii art?

"""
import json

from nupic.engine import Network
from htmresearch.support.register_regions import registerAllResearchRegions


def createL4L2Column(network, networkConfig, suffix=""):
  """
  Create a a single column containing one L4 and one L2.

  networkConfig is a dict that must contain the following keys (additional keys
  ok):

    {
      "externalInputSize": 1024,
      "sensorInputSize": 1024,
      "L4Params": {
        <constructor parameters for ExtendedTMRegion
      },
      "L2Params": {
        <constructor parameters for ColumnPoolerRegion>
      }
    }

  Region names are externalInput, sensorInput, L4Column, and ColumnPoolerRegion.
  Each name has an optional string suffix appended to it.
  """
  externalInputName = "externalInput" + suffix
  sensorInputName = "sensorInput" + suffix
  L4ColumnName = "L4Column" + suffix
  L2ColumnName = "L2Column" + suffix

  # Create the two sensors, L4 column, and L2 column
  network.addRegion(
    externalInputName, "py.RawSensor",
    json.dumps({"outputWidth": networkConfig["externalInputSize"]}))
  network.addRegion(
    sensorInputName, "py.RawSensor",
    json.dumps({"outputWidth": networkConfig["sensorInputSize"]}))
  network.addRegion(
    L4ColumnName, "py.ExtendedTMRegion",
    json.dumps(networkConfig["L4Params"]))
  network.addRegion(
    L2ColumnName, "py.ColumnPoolerRegion",
    json.dumps(networkConfig["L2Params"]))

  # Set phases appropriately so regions are executed in the proper sequence
  # This is required when we create multiple columns - the order of execution
  # is not the same as the order of region creation.
  network.setPhases(externalInputName,[0])
  network.setPhases(sensorInputName,[0])
  network.setPhases(L4ColumnName,[1])
  network.setPhases(L2ColumnName,[2])

  # Link sensors to L4
  network.link(externalInputName, L4ColumnName, "UniformLink", "",
               srcOutput="dataOut", destInput="externalInput")
  network.link(sensorInputName, L4ColumnName, "UniformLink", "",
               srcOutput="dataOut", destInput="feedForwardInput")

  # Link L4 to L2, and L2's feedback to L4
  network.link(L4ColumnName, L2ColumnName, "UniformLink", "",
               srcOutput="feedForwardOutput", destInput="feedforwardInput")
  network.link(L2ColumnName, L4ColumnName, "UniformLink", "",
               srcOutput="feedForwardOutput", destInput="apicalInput")

  # Link reset output to L4 and L2
  network.link(sensorInputName, L4ColumnName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")
  network.link(sensorInputName, L2ColumnName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  enableProfiling(network)

  return network


def createMultipleL4L2Columns(network, networkConfig):
  """
  Create a network consisting of multiple columns.  Each column contains one L4
  and one L2, is identical in structure to the network created by
  createL4L2Column. In addition all the L2 columns are fully connected to each
  other through their lateral inputs.

  Region names have a column number appended as in externalInput_0,
  externalInput_1, etc.

  networkConfig must be of the following format:

    {
      "networkType": "MultipleL4L2Columns",
      "numCorticalColumns": 3,
      "externalInputSize": 1024,
      "sensorInputSize": 1024,
      "L4Params": {
        <constructor parameters for ExtendedTMRegion
      },
      "L2Params": {
        <constructor parameters for ColumnPoolerRegion>
      }
    }
  """
  # TODO: for now, each L2 column should have a different seed
  if "seed" not in networkConfig["L2Params"]:
    networkConfig["L2Params"]["seed"] = 42

  # Create each column
  for i in range(networkConfig["numCorticalColumns"]):
    suffix = "_" + str(i)
    network = createL4L2Column(network, networkConfig, suffix)
    networkConfig["L2Params"]["seed"] += 1

  # Now connect the L2 columns laterally
  for i in range(networkConfig["numCorticalColumns"]):
    suffixSrc = "_" + str(i)
    for j in range(networkConfig["numCorticalColumns"]):
      if i != j:
        suffixDest = "_" + str(j)
        network.link(
            "L2Column" + suffixSrc, "L2Column" + suffixDest,
            "UniformLink", "",
            srcOutput="feedForwardOutput", destInput="lateralInput")

  enableProfiling(network)

  return network


def enableProfiling(network):
  """Enable profiling for all regions in the network."""
  for region in network.regions.values():
    region.enableProfiling()


def createNetwork(networkConfig):
  """
  Create and initialize the specified network instance.

  @param networkConfig: (dict) the configuration of this network.
  @return network: (Network) The actual network
  """

  registerAllResearchRegions()

  network = Network()

  if networkConfig["networkType"] == "L4L2Column":
    return createL4L2Column(network, networkConfig, "_0")
  elif networkConfig["networkType"] == "MultipleL4L2Columns":
    return createMultipleL4L2Columns(network, networkConfig)

