import csv
import os
import scipy
import random
import time
import numpy as np
from matplotlib import pyplot as plt

from htmresearch.frameworks.classification.utils.traces import (
  loadTraces, convertAnomalyScore)
import htmresearch.frameworks.clustering.online_agglomerative_clustering as oac

from htmresearch.frameworks.clustering.distances import (kernel_dist,
                                                         euclidian)
from htmresearch.frameworks.clustering.kernels import (
  normalized_gaussian_kernel)



def generate_points(num_classes, points_per_class, noise, dim,
                    merge_1_class=True):
  points = []
  labels = []
  # create three random 2D gaussian clusters
  for i in range(num_classes):
    center = [i for _ in range(dim)]
    for _ in range(points_per_class):
      point = scipy.array([center[k] + random.normalvariate(0, noise)
                           for k in range(dim)])
      points.append(point)
      if merge_1_class:
        if i == num_classes - 1:
          labels.append(0)
        else:
          labels.append(i)
      else:
        labels.append(i)

  # shuffle
  shuf_indices = range(len(points))
  random.shuffle(shuf_indices)
  points = [points[i] for i in shuf_indices]
  labels = [labels[i] for i in shuf_indices]

  return points, labels



def cluster_category_frequencies(cluster):
  labels = []
  for point in cluster.points:
    labels.append(point['label'])

  unique, counts = np.unique(labels, return_counts=True)
  frequencies = []
  for actualCategory, numberOfPoints in np.asarray((unique, counts)).T:
    frequencies.append({
      'actual_category': actualCategory,
      'num_points': numberOfPoints
    })

  return frequencies



def moving_average(last_ma, new_point_value, rolling_window_size):
  """
  Online computation of moving average.
  From: http://www.daycounter.com/LabBook/Moving-Average.phtml
  """

  ma = last_ma + (new_point_value - last_ma) / float(rolling_window_size)
  return ma



def clustering_stats(record_number,
                     winning_clusters,
                     clusters,
                     closest,
                     actual_label,
                     num_correct,
                     accuracy_ma,
                     rolling_window):
  if closest:
    # info about predicted cluster
    freqs = cluster_category_frequencies(closest)
    cluster_category = freqs[0]['actual_category']

    # compute accuracy      
    if cluster_category == actual_label:
      accuracy = 1
    else:
      accuracy = 0
    num_correct += accuracy
    accuracy_ma = moving_average(accuracy_ma, accuracy, rolling_window)
    cluster_id = closest.id
    cluster_size = closest.size

    print('Record: %s | Accuracy MA: %s' % (record_number, accuracy_ma))
    print("Winning clusters: %s | Total clusters: %s | "
          "Closest: {id=%s, size=%s, category=%s} | Actual category: %s"
          % (len(winning_clusters), len(clusters), cluster_id, cluster_size,
             cluster_category, actual_label))

  return accuracy_ma



def run(max_num_clusters,
        cluster_size_cutoff,
        trim_clusters,
        rolling_window,
        distance_func,
        points,
        labels):
  num_points = len(points)
  start = time.time()
  model = oac.OnlineAgglomerativeClustering(max_num_clusters,
                                            distance_func,
                                            cluster_size_cutoff)
  clusters_history = []
  winning_clusters_history = []
  accuracy_ma_history = []
  num_correct = 0
  accuracy_ma = 0
  for i in range(num_points):
    point = points[i]
    actual_label = labels[i]
    winning_clusters, closest = model.cluster(point, trim_clusters,
                                              actual_label)

    clusters = model._clusters
    clusters_history.append(clusters)
    winning_clusters_history.append(winning_clusters)

    accuracy_ma = clustering_stats(i,
                                   winning_clusters,
                                   clusters,
                                   closest,
                                   actual_label,
                                   num_correct,
                                   accuracy_ma,
                                   rolling_window)
    accuracy_ma_history.append(accuracy_ma)

  print ("%d points clustered in %.2f s." % (num_points, time.time() - start))
  return accuracy_ma_history, clusters_history, winning_clusters_history



def plot_2d_clusters(points_history, clusters_history,
                     winning_clusters_history):
  plt.ion()  # interactive mode on
  last_cx = []
  last_cy = []
  last_winning_cx = []
  last_winning_cy = []
  num_points = len(clusters_history)
  for i in range(num_points):
    clusters = clusters_history[i]
    winning_clusters = winning_clusters_history[i]
    point = points_history[i]
    plt.plot(point[0], point[1], 'bo')

    winning_cx = [x.center[0] for x in winning_clusters]
    winning_cy = [y.center[1] for y in winning_clusters]

    cx = [x.center[0] for x in clusters]
    cy = [y.center[1] for y in clusters]

    plt.plot(last_cx, last_cy, "bo")
    plt.plot(cx, cy, "yo")
    plt.plot(last_winning_cx, last_winning_cy, "bo")
    plt.plot(winning_cx, winning_cy, "ro")
    plt.pause(0.01)

    last_cx = cx
    last_cy = cy



def plot_accuracy(accuracy_moving_averages,
                  rolling_window,
                  points,
                  labels,
                  anomalyScores,
                  title,
                  anomalyScoreType,
                  xlim):
  fig, ax = plt.subplots(nrows=3, figsize=(15, 7))

  # plot sensor value and class labels
  t = range(xlim[0], xlim[1] + 1)
  ax[0].plot(t, points, label='signal')
  ax[0].set_xlabel('Time step')
  ax[0].set_ylabel('Signal amplitude')
  ax[0].set_xlim(xmin=xlim[0], xmax=xlim[1])
  categoryColors = ['grey', 'blue', 'yellow', 'red', 'green', 'orange']
  previousLabel = labels[0]
  start = 0
  labelCount = 0
  numPoints = len(labels)
  categoriesLabelled = []
  for label in labels:
    if previousLabel != label or labelCount == numPoints - 1:

      categoryColor = categoryColors[int(previousLabel)]
      if categoryColor not in categoriesLabelled:
        labelLegend = 'Cat. %s' % int(previousLabel)
        categoriesLabelled.append(categoryColor)
      else:
        labelLegend = None

      end = labelCount
      ax[0].axvspan(start, end, facecolor=categoryColor, alpha=0.4,
                    label=labelLegend)
      start = end
      previousLabel = label

    labelCount += 1

  ax[0].set_title(title)
  ax[0].legend(ncol=4)
  # clustering accuracy
  ax[1].plot(accuracy_moving_averages)
  ax[1].set_title('Clustering Accuracy Moving Average (Window = %s)'
                  % rolling_window)
  ax[1].set_xlabel('Time step')
  ax[1].set_ylabel('Accuracy MA')

  # plot anomaly score
  ax[2].set_title(anomalyScoreType)
  ax[2].plot(anomalyScores)

  plt.savefig('clustering_accuracy.png')



def plot_clustering_results(clusters):
  fig, ax = plt.subplots(figsize=(15, 7))
  # cluster sizes
  num_clusters = len(clusters)
  categories_to_num_points = {}
  for i in range(num_clusters):
    cluster = clusters[i]
    cluster_id = cluster.id
    freqs = cluster_category_frequencies(cluster)
    for freq in freqs:
      num_points = int(freq['num_points'])
      category = int(freq['actual_category'])
      if category not in categories_to_num_points:
        categories_to_num_points[category] = {}
      categories_to_num_points[category][cluster_id] = num_points

  cluster_ids = []
  for clusters_to_num_points in categories_to_num_points.values():
    cluster_ids.extend(clusters_to_num_points.keys())
  cluster_ids = list(set(cluster_ids))

  # Get some pastel shades for the colors. Note: category index start at 0 
  num_categories = max(categories_to_num_points.keys()) + 1
  colors = plt.cm.BuPu(np.linspace(0, 0.5, num_categories))
  bottom = np.array([0 for _ in range(len(cluster_ids) + 1)])
  # Plot bars and create text labels for the table
  cell_text = []
  for category, clusters_to_num_points in categories_to_num_points.items():
    bars = []
    for cid in cluster_ids:
      if cid in clusters_to_num_points:
        bars.append(clusters_to_num_points[cid])
      else:
        bars.append(0)
    bars.append(sum(bars))

    # draw the bars for this category
    bar_width = 0.4
    ax.bar(np.array([i for i in range(len(cluster_ids) + 1)]) + 0.3,
           bars,
           bar_width,
           bottom=bottom,
           color=colors[category])
    bottom += np.array(bars)
    cell_text.append([x for x in bottom])

  ax.set_title('Number of points in each category by Cluster')
  ax.set_ylabel('Number of points')

  # Reverse colors and text labels to display the last value at the top.
  colors = colors[::-1]
  cell_text.reverse()

  # Add a table at the bottom of the axes
  rowLabels = ['category %s' % c for c in categories_to_num_points.keys()]
  colLabels = ['cluster %s' % c for c in cluster_ids]
  colLabels.append('Tot. pts')
  the_table = plt.table(cellText=cell_text,
                        rowLabels=rowLabels,
                        rowColours=colors,
                        colLabels=colLabels,
                        loc='bottom')
  the_table.auto_set_font_size(False)
  the_table.set_fontsize(9)
  the_table.scale(1, 2)
  ax.set_xticks([])
  plt.tight_layout(pad=6)
  plt.savefig('cluster_assignments.png')



def get_file_name(exp_name, network_config):
  trace_csv = 'traces_%s_%s.csv' % (exp_name, network_config)
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      os.pardir, 'classification', 'results', trace_csv)



def convertToSDRs(patterNZs, input_width):
  sdrs = []
  for i in range(len(patterNZs)):
    patternNZ = patterNZs[i]
    sdr = np.zeros(input_width)
    sdr[patternNZ] = 1
    sdrs.append(sdr)
  return sdrs



def convertToSum(points, window_size):
  num_points = len(points)
  sums = []
  for i in range(num_points):
    if i < window_size - 1:
      idx = range(0, i + 1)
      assert len(idx) < window_size
    elif i > num_points - window_size:
      idx = range(i, num_points)
      assert len(idx) < window_size
    else:
      idx = range(i - window_size + 1, i + 1)
      assert len(idx) == window_size

    points_to_sum = []
    for k in idx:
      points_to_sum.append(points[k])
    # sum = np.sum(points_to_sum, axis=0)
    sum = np.sum(points_to_sum, axis=0) / len(points_to_sum)
    sums.append(sum)

  return sums



def load_csv(input_file):
  with open(input_file, 'r') as f:
    reader = csv.reader(f)
    headers = reader.next()
    points = []
    labels = []
    for row in reader:
      dict_row = dict(zip(headers, row))
      points.append(scipy.array([float(dict_row['x']),
                                 float(dict_row['y'])]))
      labels.append(int(dict_row['label']))

    return points, labels



def demo_gaussian_noise():
  max_num_clusters = 6
  cluster_size_cutoff = 0.5
  trim_clusters = True
  rolling_window = 10
  distance_func = kernel_dist(normalized_gaussian_kernel)
  points, labels = load_csv('gaussian_2d_noise=0.1.csv')

  # input data
  input_width = 2
  # num_classes = 4
  # num_points_per_class = 200
  # noise_level = 0.1
  # points, labels = generate_points(num_classes,
  #                                  num_points_per_class,
  #                                  noise_level,
  #                                  input_width)
  # 
  # with open('gaussian_2d.csv', 'w+') as f:
  #   writer = csv.writer(f)
  #   writer.writerow(['x', 'y', 'label'])
  #   for i in range(len(points)):
  #     writer.writerow([points[i][0], points[i][1], labels[i]])

  (accuracy_ma_history,
   clusters_history,
   winning_clusters_history) = run(max_num_clusters,
                                   cluster_size_cutoff,
                                   trim_clusters,
                                   rolling_window,
                                   distance_func,
                                   points,
                                   labels)

  # TODO: we can't plot in 2D for now if dim > 2. Will fix with projection algo
  # if input_width == 2:
  #  plot_2d_clusters(points, clusters_history, winning_clusters_history)
  last_clusters = clusters_history[-1]
  plot_clustering_results(last_clusters)



def demo_htm():
  # exp_name = 'binary_ampl=10.0_mean=0.0_noise=0.0'
  # exp_name = 'binary_ampl=10.0_mean=0.0_noise=1.0'
  exp_name = 'sensortag_z'

  anomalyScoreType = 'rawAnomalyScore'
  start_idx = 200
  end_idx = 300
  rolling_window = 10
  distance_func = euclidian

  input_width = 2048 * 32

  max_num_clusters = 15
  trim_clusters = False
  cluster_size_cutoff = 0.5

  # predictive and active cells
  activeCellsWeight = 0
  predictedActiveCellsWeight = 1

  # sum SDRs
  sum_sdrs = True
  sum_window = 5

  # threshold anomaly scores
  convert_anomaly_scores = False
  anomaly_score_probationary_period = 1
  anomaly_score_cutoff = 0.5

  network_config = 'sp=True_tm=True_tp=False_SDRClassifier'
  file_name = get_file_name(exp_name, network_config)
  traces = loadTraces(file_name)

  labels = traces['actualCategory'][start_idx:end_idx]
  anomalyScores = traces[anomalyScoreType][start_idx:end_idx]
  activeCells = traces['tmActiveCells'][start_idx:end_idx]
  predictedActiveCells = traces['tmPredictedActiveCells'][start_idx:end_idx]
  activeCellsSDRs = convertToSDRs(activeCells, input_width)
  predictedActiveCellsSDRs = convertToSDRs(predictedActiveCells,
                                           input_width)
  points = (activeCellsWeight * np.array(activeCellsSDRs)
            + predictedActiveCellsWeight * np.array(predictedActiveCellsSDRs))

  if convert_anomaly_scores:
    anomalyScores = convertAnomalyScore(anomalyScores,
                                          anomaly_score_cutoff,
                                          anomaly_score_probationary_period)
  
  pointsToCluster = []
  labelsToCluster = []
  for i in range(len(anomalyScores)):
    anomalyScore = anomalyScores[i]
    if anomalyScore <= anomaly_score_cutoff:
      labelsToCluster.append(labels[i])
      pointsToCluster.append(points[i])
  points = pointsToCluster
  labels = labelsToCluster
  if sum_sdrs:
    points = convertToSum(points, sum_window)

  (accuracy_ma_history,
   clusters_history,
   winning_clusters_history) = run(max_num_clusters,
                                   cluster_size_cutoff,
                                   trim_clusters,
                                   rolling_window,
                                   distance_func,
                                   points,
                                   labels)

  last_clusters = clusters_history[-1]
  plot_clustering_results(last_clusters)

  # plot clustering accuracy over time
  xlim = [0, len(labels) - 1]
  # plot_accuracy(accuracy_ma_history,
  #               rolling_window,
  #               points,
  #               labels,
  #               anomalyScores,
  #               exp_name,
  #               anomalyScoreType,
  #               xlim)



if __name__ == "__main__":
  # demo_gaussian_noise()
  demo_htm()
