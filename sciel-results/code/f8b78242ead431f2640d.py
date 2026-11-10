import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    # Convert sequences to numpy arrays of integers 0-3
    # A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = []
    for label in labels:
        seq = sequences[label]
        arr = np.array([char_map[c] for c in seq], dtype=np.int8)
        seq_arrays.append(arr)
    
    # Stack into 2D array: (n, L)
    seq_matrix = np.stack(seq_arrays, axis=0)
    L = seq_matrix.shape[1]
    
    # Vectorized Hamming distance computation
    # For each pair (i,j), count positions where seq_matrix[i] != seq_matrix[j]
    dist_mat = np.zeros((n, n), dtype=np.float64)
    
    # Compute all pairwise differences
    # Using broadcasting: seq_matrix[:, np.newaxis, :] vs seq_matrix[np.newaxis, :, :]
    # This creates a (n, n, L) array which might be memory intensive for large n*L
    # Better approach: use matrix multiplication trick or loop over pairs with vectorization
    
    # For moderate n (up to 32), we can use the following efficient approach:
    # Count mismatches using np.sum over the sequence dimension
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_matrix[i] != seq_matrix[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat

def _nj_tree(dist_mat, labels):
    """
    Neighbor Joining algorithm to reconstruct unrooted binary tree.
    Returns a dictionary of edges and the root node for Newick construction.
    """
    n = len(labels)
    if n <= 2:
        return None, None
    
    # D is a 2D array of size (2n) x (2n) to accommodate new nodes
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            D[i, j] = dist_mat[i, j]
    
    # Active nodes list
    active = list(range(n))
    
    # Tree structure: dictionary mapping parent node -> (child1, child2, len1, len2)
    tree_edges = {}
    next_id = n
    
    while len(active) > 3:
        k = len(active)
        
        # Calculate row sums using the active nodes
        r = np.zeros(k)
        for i, node in enumerate(active):
            for j, other in enumerate(active):
                if i != j:
                    r[i] += D[node, other]
        
        # Find best pair (i, j) that minimizes Q(i,j) = (k-2)*D[i,j] - r[i] - r[j]
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i + 1, k):
                node_i = active[i]
                node_j = active[j]
                q = (k - 2) * D[node_i, node_j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_a = active[best_i]
        node_b = active[best_j]
        
        # Calculate branch lengths
        delta = (r[best_i] - r[best_j]) / (k - 2)
        len_a = 0.5 * (D[node_a, node_b] + delta)
        len_b = 0.5 * (D[node_a, node_b] - delta)
        
        # Create new internal node
        new_node = next_id
        next_id += 1
        
        tree_edges[new_node] = (node_a, node_b, max(0.0, len_a), max(0.0, len_b))
        
        # Update distances: d(new_node, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
        for x in active:
            if x != node_a and x != node_b:
                D[new_node, x] = 0.5 * (D[node_a, x] + D[node_b, x] - D[node_a, node_b])
                D[x, new_node] = D[new_node, x]
        
        # Remove a and b, add new_node
        active = [x for x in active if x != node_a and x != node_b]
        active.append(new_node)
    
    # Handle the last 3 nodes
    if len(active) == 3:
        u, v, w = active
        # Compute branch lengths from the central node
        len_u = 0.5 * (D[u, v] + D[u, w] - D[v, w])
        len_v = 0.5 * (D[u, v] + D[v, w] - D[u, w])
        len_w = 0.5 * (D[u, w] + D[v, w] - D[u, v])
        
        # Create binary tree: ((u, v), w)
        new_node_1 = next_id
        next_id += 1
        tree_edges[new_node_1] = (u, v, max(0.0, len_u), max(0.0, len_v))
        
        new_node_2 = next_id
        next_id += 1
        # The edge from new_node_1 to w has length 0 in the binary representation
        # (the "central" node is absorbed)
        tree_edges[new_node_2] = (new_node_1, w, 0.0, max(0.0, len_w))
        
        return tree_edges, new_node_2
    else:
        return tree_edges, active[0] if active else 0

def reconstruct(sequences):
    start_time = time.time()
    # Be conservative: use 260s out of 270s to account for overhead
    time_limit = 260.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return f"{labels[0]};"
    elif n == 2:
        return f"({labels[0]},{labels[1]})"
    elif n == 3:
        # For 3 taxa, return a simple star-like binary tree
        return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    # 1. Compute Distance Matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # Check time
    if time.time() - start_time > time_limit * 0.3:
        # Fallback: return a star tree
        return "({})".format(",".join(labels))
    
    # 2. Run Neighbor Joining
    tree_edges, root_node = _nj_tree(dist_mat, labels)
    
    if tree_edges is None or root_node is None:
        # Fallback: return a star tree
        return "({})".format(",".join(labels))
    
    # Build Newick string
    def get_newick(node):
        if node < n:
            return labels[node]
        else:
            if node not in tree_edges:
                return labels[node % n]  # Fallback
            child_a, child_b, len_a, len_b = tree_edges[node]
            return f"({get_newick(child_a)}:{len_a:.4f},{get_newick(child_b)}:{len_b:.4f})"
    
    newick = get_newick(root_node)
    
    # Final time check
    if time.time() - start_time > time_limit:
        return newick
    
    return newick
