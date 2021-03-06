#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

helpStr = """
Methods and data for running NLP model API tests. The intent here is
to ensure that changes to the models does not decrease their classification
accuracies (see NLP_MODEL_ACCURACIES below). Three tests are supported:

  hello classification: Very simple, hello world classification test. There are
    two categories that can be discriminated using bag of words. The training
    set is 8 docs, and the test set is an additional 2 (i.e. 10 total) -- first
    is an incorrectly labeled version of a training sample, second is
    semantically similar to one of the training samples.
  simple queries: Qualitative test where we query a trained model to see which
    data samples are the most and least similar.
  simple labels: Less simple classification test. The dataset used here must be
    specified in the command line args.

"""
import argparse
import numpy

from prettytable import PrettyTable
from textwrap import TextWrapper
from tqdm import tqdm

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.model_factory import (
  createModel, getNetworkConfig)



# There should be one "htm" model for each htm config entry.
NLP_MODEL_TYPES = [
  "docfp",
  "cioword",
  "htm",
  "htm",
  "htm",
  "keywords"]

# Network models use 4k retina.
HTM_CONFIGS = [
  ("HTM_sensor_knn", "../data/network_configs/sensor_knn.json"),
  ("HTM_sensor_simple_tp_knn", "../data/network_configs/sensor_simple_tp_knn.json"),
  ("HTM_sensor_tm_knn", "../data/network_configs/sensor_tm_knn.json"),
]

# Some values of k we know work well.
K_VALUES = { "keywords": 21, "docfp": 1}

NLP_MODEL_ACCURACIES = {
  "hello_classification": {
    "docfp": 90.0,
    "cioword": 90.0,
    "HTM_sensor_knn": 80.0,
    "HTM_sensor_simple_tp_knn": 90.0,
    "HTM_sensor_tm_knn": 90.0,
    "keywords": 80.0,
  },
  "simple_queries": {
    "docfp": "good but not great",
    "cioword": "passable",
    "HTM_sensor_knn": "passable",
    "HTM_sensor_simple_tp_knn": "idk, no results yet!",
    "HTM_sensor_tm_knn": "idk, no results yet!",
    "keywords": "passable",
  },
  "simple_labels": {
    "docfp": 100.0,
    "cioword": 100.0,
    "HTM_sensor_knn": 66.2,
    "HTM_sensor_simple_tp_knn": 99.7,
    "HTM_sensor_tm_knn": 0.0,
    "keywords": 16.9,
  },
}

_WRAPPER = TextWrapper(width=80)



def executeModelLifecycle(args, trainingData, labelRefs):
  """ Execute model lifecycle: create a model, train it, save it, reload it.

  @param args (argparse) Arguments used in classification model API experiments.
  @param trainingData (dict) Keys are document numbers, values are three-tuples
      of the document (str), labels (list), and document ID (int).
  @param labelRefs (list) Label names (str) corresponding to label indices.

  @return (two-tuple) Original and new models.
  """
  model = instantiateModel(args)
  model = trainModel(model, trainingData, labelRefs, args.verbosity)
  model.save(args.modelDir)
  newModel = ClassificationModel.load(args.modelDir)
  return model, newModel


def instantiateModel(args):
  """
  Set some specific arguments and return an instance of the model we will use.
  """
  args.networkConfig = getNetworkConfig(args.networkConfigPath)
  args.k = K_VALUES.get(args.modelName, 1)

  return createModel(**vars(args))


def trainModel(model, trainingData, labelRefs, verbosity=0):
  """
  Train the given model on trainingData. Return the trained model instance.
  """
  modelName = repr(model).split()[0].split(".")[-1]
  print
  print "===================Training {} on sample text================".format(
    modelName)
  if verbosity > 0:
    printTemplate = PrettyTable(["ID", "Document", "Label"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"
  for (document, labels, docId) in tqdm(trainingData):
    if verbosity > 0:
      docStr = unicode(document, errors="ignore")
      printTemplate.add_row(
        [docId, _WRAPPER.fill(docStr), labelRefs[labels[0]]])
    model.trainDocument(document, labels, docId)
  if verbosity > 0:
    print printTemplate

  return model


def testModel(model, testData, labelRefs, docCategoryMap=None, verbosity=0):
  """
  Test the given model on testData, print out and return accuracy percentage.

  Accuracy is calculated as follows. Each token in a document votes for a single
  category; it's possible for a token to contribute no votes. The document is
  classified with the category that received the most votes. Note that it is
  possible for a document to receive no votes, in which case it is counted as a
  misclassification.
  """
  modelName = repr(model).split()[0].split(".")[-1]
  print
  print "===================Testing {} on sample text==================".format(
    modelName)
  if verbosity > 0:
    print
    printTemplate = PrettyTable(
      ["ID", "Document", "Actual Label(s)", "Predicted Label"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"

  numCorrect = 0
  labelRefs.append("none")
  for (document, labels, docId) in tqdm(testData):

    categoryVotes, _, _ = model.inferDocument(document)

    if categoryVotes.sum() > 0:
      # We will count classification as correct if the best category is any
      # one of the categories associated with this docId
      predicted = categoryVotes.argmax()
      if predicted in docCategoryMap[docId]:
        numCorrect += 1
    else:
      # No classification possible for this doc
      predicted = -1

    if verbosity > 0:
      docStr = unicode(document, errors="ignore")
      printTemplate.add_row(
        [docId,
         _WRAPPER.fill(docStr),
         [labelRefs[l] for l in labels],
         labelRefs[predicted]]
      )

  accuracyPct = numCorrect * 100.0 / len(testData)

  if verbosity > 0:
    print printTemplate
  print
  print "Total correct =", numCorrect, "out of", len(testData), "documents"
  print "Accuracy =", accuracyPct, "%"

  return accuracyPct


def printSummary(testName, accuracies):
  """ Print comparison of the new acuracies against the current values.
  @param testName (str) One of the NLP_MODEL_ACCURACIES keys.
  @param accuracies (dict) Keys are model names, values are accuracy percents.
  """
  try:
    currentAccuracies = NLP_MODEL_ACCURACIES[testName]
  except KeyError as e:
    print "No accuracy values for test '{}'".format(testName)
    raise

  printTemplate = PrettyTable(["NLP Model", "Current Accuracy", "New Accuracy"])
  printTemplate.align = "l"
  printTemplate.header_style = "upper"

  for modelName, accuracyPct in accuracies.iteritems():
    currentPct = currentAccuracies.get(modelName, "which model?")
    printTemplate.add_row([modelName, currentPct, accuracyPct])

  print
  print "Results summary:"
  print printTemplate


def assertResults(testName, accuracies):
  """ Assert the new acuracies against the current values.
  @param testName (str) One of the NLP_MODEL_ACCURACIES keys.
  @param accuracies (dict) Keys are model names, values are accuracy percents.
  """
  try:
    currentAccuracies = NLP_MODEL_ACCURACIES[testName]
  except KeyError as e:
    print "No accuracy values for test '{}'".format(testName)
    raise e

  for modelName, accuracyPct in accuracies.iteritems():
    currentPct = currentAccuracies.get(modelName, 0.0)
    assert accuracyPct >= currentPct, \
      "{} does not pass the test! Current accuracy is {}, new is {}.".format(
      modelName, currentPct, accuracyPct)
