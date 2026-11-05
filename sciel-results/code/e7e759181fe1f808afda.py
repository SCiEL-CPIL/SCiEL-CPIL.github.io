"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Optimized Neighbor-Joining with deadline guard and NumPy vectorization.'''
import math
import time
import numpy as np


_START_TIME = time.time()
_DEADLINE = 265.0  # Safety margin before 270s wall-clock kill


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy."""
    n = len(labels)
    seqs = [sequences[l] for l in labels]
    L = len(seqs[0])
    
    # Convert to integer codes: A=0, C=1, G=2, T=3
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.array([[code_map[c] for c in s] for s in seqs], dtype=np.int8)
    
    # Compute pairwise Hamming distances
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Count mismatches between row i and all others
        mismatches = np.sum(mat[i] != mat, axis=1)
        p = mismatches / L
        # JC correction: d = -3/4 * ln(1 - 4p/3), capped at p >= 0.75
        valid = p < 0.7499
        d = np.where(valid, -0.75 * np.log(1.0 - 4.0 * p / 3.0), 3.0)
        d[i] = 0.0
        dist[i] = d
    
    return dist


def _nj_build(dist_matrix):
    """Standard Neighbor-Joining algorithm. Returns list of (a, b, length_a, length_b) joins."""
    n = dist_matrix.shape[0]
    D = dist_matrix.copy()
    r = np.sum(D, axis=1)
    active = list(range(n))
    joins = []
    
    while len(active) > 3:
        m = len(active)
        # Compute Q matrix for active nodes
        # Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        best_val = float('inf')
        best_i = best_j = -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                i = active[ii]
                j = active[jj]
                q = (m - 2) * D[i, j] - r[i] - r[j]
                if q < best_val:
                    best_val = q
                    best_i = i
                    best_j = j
        
        i, j = best_i, best_j
        # Compute branch lengths
        if D[i, j] > 1e-10:
            li = 0.5 * D[i, j] + (r[i] - r[j]) / (2.0 * (m - 2))
            lj = D[i, j] - li
        else:
            li = lj = 0.0
        joins.append((i, j, li, lj))
        
        # Create new node
        new_node = n + len(joins)
        active.remove(i)
        active.remove(j)
        
        # Update distances to new node
        for x in active:
            D[new_node, x] = 0.5 * (D[i, x] + D[j, x] - D[i, j])
            D[x, new_node] = D[new_node, x]
        
        # Update r for new node
        r[new_node] = sum(D[new_node, x] for x in active)
        # Recompute r for remaining active nodes
        for x in active:
            r[x] = sum(D[x, y] for y in active)
        
        active.append(new_node)
    
    # Final three nodes form a star
    if len(active) == 3:
        a, b, c = active
        joins.append((a, b, 0.0, 0.0))
        joins.append((b, c, 0.0, 0.0))
    
    return joins


def _build_newick(labels, joins, dist_matrix):
    """Build Newick string from NJ joins."""
    n = len(labels)
    # Track the current tree representation for each node
    tree_str = {}
    for i, label in enumerate(labels):
        tree_str[i] = label
    
    # Process joins in reverse to build up the tree
    # Actually, process in forward order: each join creates a new internal node
    # We need to track which internal nodes correspond to which joins
    
    # Let's redo: build a parent map
    parent = {}  # node -> (left, right) or None for leaves
    children = {}  # node -> [left, right]
    
    for idx, (a, b, la, lb) in enumerate(joins):
        new_node = n + idx
        children[new_node] = [a, b]
        parent[a] = new_node
        parent[b] = new_node
    
    # Find the root (node with no parent)
    root = None
    for node in range(n + len(joins)):
        if node not in parent:
            root = node
            break
    
    if root is None:
        # Fallback: should not happen
        root = n + len(joins) - 1
    
    def build_subtree(node):
        if node < n:
            return labels[node]
        left, right = children[node]
        return f"({build_subtree(left)},{build_subtree(right)})"
    
    return build_subtree(root) + ";"


def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
        # Return a star tree as fallback
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    try:
        joins = _nj_build(D)
        newick = _build_newick(labels, joins, D)
    except Exception:
        # Fallback to star tree
        newick = "(" + ",".join(labels) + ");"
    
    return newick
# EVOLVE-BLOCK-END
