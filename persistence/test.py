from rank_persistence.persistence.graph_persistence import Wgraph

example = [('a', 'b', 14),
           ('b', 'c',  4),
           ('c', 'd',  6),
           ('d', 'e',  5),
           ('e', 'f', 12),
           ('f', 'g',  8),
           ('g', 'h',  9),
           ('i', 'f', 11),
           ('i', 'a', 13),
           ('i', 'b',  2),
           ('i', 'c',  3),
           ('c', 'f', 15),
           ('c', 'e',  7),
           ('h', 'f', 10)]

graph = Wgraph(weighted_edges = example, nx_graph = None)
# graph.build_graph()
graph.build_filtered_subgraphs()
graph.build_filtered_subgraphs(weight_transform = None)
graph.edge_block_persistence(edges_to_remove=1) 
