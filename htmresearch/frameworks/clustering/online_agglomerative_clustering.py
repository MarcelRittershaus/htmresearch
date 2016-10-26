import heapq
import operator
import scipy
import logging

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)
_LOGGER.addHandler(logging.StreamHandler())



class Cluster(object):
  def __init__(self, cluster_id, center, distance_func):
    self.id = cluster_id
    self.center = center
    self.size = 0
    self.distance_func = distance_func
    self.points = []


  def add(self, e, label):
    self.size += 1
    self.center += e / float(self.size)
    self.points.append({'point': e, 'label': label})


  def merge(self, c):
    self.center = (self.center * self.size + c.center * c.size) / float(
      self.size + c.size)
    self.size += c.size
    self.points.extend(c.points)


  def resize(self, dim):
    extra = scipy.zeros(dim - len(self.center))
    self.center = scipy.append(self.center, extra)


  def __str__(self):
    return "Cluster( %s, %s, %.2f )" % (self.id, self.center, self.size)



class Dist(object):
  """
  this is just a tuple,
  but we need an object so we can define cmp for heapq
  """


  def __init__(self, c1, c2, d):
    self.c1 = c1
    self.c2 = c2
    self.d = d


  def __cmp__(self, o):
    return cmp(self.d, o.d)


  def __str__(self):
    return "Dist(%.10f, %s, %s)" % (self.d, self.c1.id, self.c2.id)



class OnlineAgglomerativeClustering(object):
  def __init__(self,
               max_num_clusters,
               distance_func,
               cluster_size_cutoff):
    """
    N-1 is the largest number of clusters that can be found.
    Higher N makes clustering slower.
    """

    self._num_points_processed = 0
    self._total_num_clusters_created = 0
    self._max_num_clusters = max_num_clusters

    self._distance_func = distance_func
    self._cluster_size_cutoff = cluster_size_cutoff

    self._clusters = []
    # max number of dimensions we've seen so far
    self._dim = 0

    # cache inter-cluster distances
    self._dist = []


  def _resize(self, dim):
    for c in self._clusters:
      c.resize(dim)
    self._dim = dim


  def find_closest_cluster(self, point, clusters):
    cluster_id_to_dist = []
    for i, cluster in enumerate(clusters):
      d = self._distance_func(cluster.center, point)
      cluster_id_to_dist.append((i, d))

    closest_id, distance_to_closest = min(cluster_id_to_dist,
                                          key=operator.itemgetter(1))
    return clusters[closest_id], distance_to_closest


  def delete_cluster(self, cluster):
    for c in self._clusters:
      if cluster.id == c.id:
        self._clusters.remove(c)


  def cluster(self, new_point, trim_clusters, label=None):

    if len(new_point) > self._dim:
      self._resize(len(new_point))

    while (len(self._clusters) >= self._max_num_clusters
           and len(self._clusters) > 1):
      inter_cluster_dist = heapq.heappop(self._dist)
      cluster_to_merge = inter_cluster_dist.c2
      winning_cluster = inter_cluster_dist.c1
      assert cluster_to_merge.id > winning_cluster.id
      winning_cluster.merge(cluster_to_merge)
      self.delete_cluster(cluster_to_merge)
      # update inter-cluster distances      
      self._remove_dist(cluster_to_merge)
      self._update_dist(winning_cluster)

    if len(self._clusters) > 0:
      # compare new point to each existing cluster
      closest, distance_to_closest = self.find_closest_cluster(new_point,
                                                               self._clusters)
      closest.add(new_point, label)
      # invalidate dist-cache for this cluster
      self._update_dist(closest)
    else:
      closest = None

    # make a new cluster for this point
    self._total_num_clusters_created += 1
    cluster_id = self._total_num_clusters_created
    new_cluster = Cluster(cluster_id, new_point, self._distance_func)
    self._clusters.append(new_cluster)
    self._update_dist(new_cluster)

    self._num_points_processed += 1

    if trim_clusters:
      winning_clusters = self._trim_clusters()
      # closest cluster might not be in the list of trimmed clusters
      self.find_closest_cluster(new_point, winning_clusters)
      return winning_clusters, closest
    else:
      return self._clusters, closest


  def _remove_dist(self, cluster_to_delete):
    """Invalidate inter-cluster distance cache for c"""
    c_id = cluster_to_delete.id
    inter_cluster_dist_to_remove = [d for d in self._dist
                                    if d.c1.id == c_id or d.c2.id == c_id]
    for x in inter_cluster_dist_to_remove:
      self._dist.remove(x)


  def _update_dist(self, c):
    """Cluster c has changed, re-compute all inter-cluster distances"""
    self._remove_dist(c)
    for x in self._clusters:
      if x == c: continue
      d = self._distance_func(x.center, c.center)
      # order by ID number
      if x.id < c.id:
        inter_cluster_dist = Dist(x, c, d)
      else:
        inter_cluster_dist = Dist(c, x, d)
      heapq.heappush(self._dist, inter_cluster_dist)


  def _trim_clusters(self):
    """Return only clusters over threshold"""
    mean_cluster_size = scipy.mean([x.size for x in self._clusters])
    t = mean_cluster_size * self._cluster_size_cutoff
    return filter(lambda x: x.size >= t, self._clusters)
