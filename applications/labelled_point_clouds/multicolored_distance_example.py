import numpy as np
from rank_persistence.persistence.persistence_diagram import PersistenceDiagram, CornerPoint
from rank_persistence.applications.utils import multicolored_bottleneck_distance, load_diagram



dgm1 = load_diagram("./diagrams/016_100.npy")
dmg2 = load_diagram("./diagrams/016_10.npy")
dist = multicolored_bottleneck_distance(dgm1, dmg2, plot_subdiagrams = False,
                                        matching = False)
