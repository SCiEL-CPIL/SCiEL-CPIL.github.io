"""Rewritten program using NumPy-accelerated Neighbor-Joining."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance(a, b):
    """Calculate Jukes-Cantor corrected distance using NumPy."""
    diff = np.sum(a != b)
    L = len(a)
    p = diff / L
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_nj_tree(labels, D):
    """
    Neighbor-Joining algorithm.
    D: numpy array of shape (n, n)
    Returns a Newick string.
    """
    n = len(labels)
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})();"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"

    active_nodes = list(range(n))
    newick_map = {i: labels[i] for i in range(n)}
    
    # Use a dictionary for distances to handle dynamic node addition easily
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = D[i, j]
    
    current_id = n
    while len(active_nodes) > 3:
        m = len(active_nodes)
        r = {i: 0.0 for i in active_nodes}
        for i in active_nodes:
            for j in active_nodes:
                if i != j:
                    r[i] += dist[(i, j)]
        
        min_q = float('inf')
        min_i, min_j = -1, -1
        for idx_i, i in enumerate(active_nodes):
            for idx_j, j in enumerate(active_nodes):
                if idx_i < idx_j:
                    q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                    if q < min_q:
                        min_q = q
                        min_i, min_j = i, j
        
        i, j = min_i, min_j
        new_node = current_id
        current_id += 1
        
        newick_map[new_node] = f"({newick_map[i]},{newick_map[j]})"
        
        for k in active_nodes:
            if k not in (i, j):
                new_dist = 0.5 * (dist[(i, k)] + dist[(j, k)] - dist[(i, j)])
                dist[(new_node, k)] = new_dist
                dist[(k, new_node)] = new_dist
        
        active_nodes = [x for x in active_nodes if x not in (i, j)]
        active_nodes.append(new_node)
        
    a, b, c = active_nodes
    return f"({newick_map[a]},({newick_map[b]},{newick_map[c]}));"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    data = {}
    for label in labels:
        seq = sequences[label]
        data[label] = np.array([seq_map[c] for c in seq], dtype=np.int8)
    
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(data[labels[i]], data[labels[j]])
            D[i, j] = d
            D[j, i] = d
    
    newick = _build_nj_tree(labels, D)
    
    return newick

# EVOLVE-BLOCK-END
