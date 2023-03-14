import numpy as np
import matplotlib
from matplotlib import cm
import ripser
import persim
import itertools
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from rank_persistence.persistence.persistence_diagram import PersistenceDiagram


def bottleneck(dgm1, dgm2, proper = True, matching = False):
    if isinstance(dgm1, PersistenceDiagram):
        dgm1 = dgm1.get_cornerpoints(proper=proper)
        dgm2 = dgm2.get_cornerpoints(proper=proper)
    return persim.bottleneck(dgm1, dgm2, matching=matching)


def compute_pairwise_distance(dgms, dist = bottleneck):
    pairs = itertools.combinations(range(len(dgms)), 2)
    return [bottleneck(dgms[i], dgms[j]) for (i, j) in pairs]

def get_dendrogram(distances, labels):
    dist_mat = squareform(distances)
    linkage_matrix = linkage(distances, "ward")
    dendrogram(linkage_matrix, labels=labels)

def get_spaced_colors(n, norm = False, black = True, cmap = 'jet'):
    rgb_tuples = cm.get_cmap(cmap)
    if norm:
        colors = [rgb_tuples(i / n) for i in range(n)]
    else:
        rgb_array = np.asarray([rgb_tuples(i / n) for i in range(n)])
        brg_array = np.zeros(rgb_array.shape)
        brg_array[:,0] = rgb_array[:,2]
        brg_array[:,1] = rgb_array[:,1]
        BRG_array[:,2] = rgb_array[:,0]
        colors = [tuple(brg_array[i,:] * 256) for i in range(n)]
    if black:
        black = (0., 0., 0.)
        colors.insert(0, black)
    return colors


def get_n_colors(n, cmap = 'jet'):
    rgb_tuples = cm.get_cmap(cmap)
    colors = [rgb_tuples(i / n)[:-1] for i in range(n)]
    return colors
