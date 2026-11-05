"""Reconstruct unrooted phylogenetic topology from simulated Jukes-Cantor sequences.

Strategy:
1. Compute Jukes-Cantor corrected pairwise distances using NumPy.
2. Build a tree using Neighbor-Joining (NJ), implemented efficiently.
3. Return the tree as a Newick string.
"""
import time
import math
import numpy as np

_DEADLINE = 265.0  # seconds, leave 5s safety margin before 270s kill


def _jc_distances(sequences):
    """Compute Jukes-Cantor corrected distance matrix."""
    n = len(sequences)
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[0])
    M = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(sequences):
        for j, c in enumerate(s):
            M[i, j] = base_map[c]
    
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(M[i] != M[j])
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(max(1e-10, 1.0 - 4.0 * p / 3.0))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _neighbor_joining(dist):
    """
    Neighbor-Joining algorithm.
    Returns (root, tree_nodes) where tree_nodes maps node_id -> (left, right) for internal,
    or leaf index for leaves.
    """
    n0 = dist.shape[0]
    # We'll use a dict for tree_nodes: node_id -> (left, right)
    tree_nodes = {}
    for i in range(n0):
        tree_nodes[i] = i  # leaf
    
    D = dist.copy()
    active = list(range(n0))
    next_id = n0
    
    while len(active) > 3:
        m = len(active)
        # Row sums
        r = {}
        for i in active:
            r[i] = sum(D[i, j] for j in active if i != j)
        
        # Find best pair
        best_q = float('inf')
        best_i = best_j = -1
        for idx_i, i in enumerate(active):
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                q = (m - 2) * D[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        i, j = best_i, best_j
        new_id = next_id
        next_id += 1
        
        # Update distances
        for x in active:
            if x not in (i, j):
                D[new_id, x] = 0.5 * (D[i, x] + D[j, x] - D[i, j])
                D[x, new_id] = D[new_id, x]
        
        active = [x for x in active if x not in (i, j)]
        active.append(new_id)
        tree_nodes[new_id] = (i, j)
    
    # Final 3 nodes
    if len(active) == 3:
        a, b, c = active
        id1 = next_id; next_id += 1
        tree_nodes[id1] = (a, b)
        id2 = next_id; next_id += 1
        tree_nodes[id2] = (id1, c)
        return id2, tree_nodes
    elif len(active) == 2:
        a, b = active
        id1 = next_id; next_id += 1
        tree_nodes[id1] = (a, b)
        return id1, tree_nodes
    else:
        return active[0], tree_nodes


def _to_newick(root, tree_nodes, label_map):
    """Convert tree to Newick string."""
    if root in label_map:
        return label_map[root]
    left, right = tree_nodes[root]
    return "({},{})".format(_to_newick(left, tree_nodes, label_map),
                            _to_newick(right, tree_nodes, label_map))


def reconstruct(sequences):
    start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    label_map = {i: labels[i] for i in range(n)}
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "({},{});".format(labels[0], labels[1])
    if n == 3:
        return "(({},{}),{});".format(labels[0], labels[1], labels[2])
    
    if time.time() - start_time > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    dist = _jc_distances(sequences)
    
    if time.time() - start_time > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    root, tree_nodes = _neighbor_joining(dist)
    
    if time.time() - start_time > _DEADLINE:
        try:
            return "(" + _to_newick(root, tree_nodes, label_map) + ");"
        except:
            return "(" + ",".join(labels) + ");"
    
    return "(" + _to_newick(root, tree_nodes, label_map) + ");"
