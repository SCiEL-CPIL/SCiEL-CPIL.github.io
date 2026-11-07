import itertools

def construct():
    # Construct a single K_{6,6} graph on vertices 0-11
    # Partition vertices into two sets of 6
    set_a = range(0, 6)
    set_b = range(6, 12)
    
    # Create edges connecting every vertex in set_a to every vertex in set_b
    edge_list = [[u, v] for u, v in itertools.product(set_a, set_b)]
    
    return edge_list
