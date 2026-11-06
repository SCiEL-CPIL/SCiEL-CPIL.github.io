import time
import math
import numpy as np
import random

_START_TIME = None
_TIME_LIMIT = 260.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    return dists

def _spectral_cluster_tree(dists, labels):
    """
    Construct a tree using spectral clustering.
    1. Compute eigenvectors of the distance matrix (or its Laplacian).
    2. Project taxa onto the first few eigenvectors (spectral embedding).
    3. Perform agglomerative clustering on these coordinates (or use the 
       structure of the eigenvectors directly to define splits).
    
    For phylogenetics, the "spectral tree" often refers to using the 
    eigenvectors of the distance matrix to define the splits. 
    Specifically, the first non-trivial eigenvector of the Laplacian 
    (or distance matrix) often reveals the primary bifurcation.
    
    We will use a recursive spectral bisection approach:
    - At each step, take the current set of taxa.
    - Compute the distance matrix for this subset.
    - Find the eigenvector corresponding to the second smallest eigenvalue 
      of the Laplacian (or largest magnitude of the distance matrix, 
      excluding the trivial constant vector).
    - Split the taxa into two groups based on the sign of this eigenvector.
    - Recurse on each group.
    """
    n = len(labels)
    if n < 3:
        if n == 1:
            return (labels[0], None, None)
        else:
            # For 2 taxa, it's a trivial split
            return (None, labels[0], labels[1])

    # We will build a binary tree structure
    # Node: (label, left_child, right_child)
    # Leaf: (label, None, None)
    
    def build_tree(subset_labels, subset_dists):
        m = len(subset_labels)
        if m == 1:
            return (subset_labels[0], None, None)
        if m == 2:
            return (None, subset_labels[0], subset_labels[1])
        
        # Compute Laplacian of the distance matrix
        # L = D - A, where A is adjacency (0 on diag, d on off-diag? No, usually L = D_deg - A)
        # For distance matrices, we can just use the eigenvectors of D directly.
        # The vector with the largest variance (2nd eigenvector) usually splits the tree.
        
        # Symmetrize just in case
        D = (subset_dists + subset_dists.T) / 2.0
        np.fill_diagonal(D, 0.0)
        
        try:
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(D)
        except np.linalg.LinAlgError:
            # Fallback to NJ if spectral fails (rare)
            return _nj_build_subset(subset_labels, subset_dists)
        
        # Sort by eigenvalue magnitude? 
        # Usually the smallest eigenvalue is 0 (or close) and corresponds to the constant vector.
        # The next one (index 1) often gives the main split.
        # Let's pick the eigenvector with the second smallest absolute eigenvalue 
        # that is not the constant vector.
        
        # Find the index of the smallest eigenvalue (should be ~0)
        # We want the "Fiedler vector" equivalent for distance matrices.
        # Let's just take the eigenvector with index 1 (second smallest).
        # If the matrix is not connected or has issues, we might need to check.
        
        # Ensure we don't pick the trivial constant vector.
        # The constant vector has all entries equal.
        # Let's check the variance of the eigenvectors.
        
        # Sort indices by eigenvalue
        idx = np.argsort(eigenvalues)
        
        # The first one is usually the trivial one.
        # Let's take the next one.
        vec_idx = 1
        if m > 2:
             # Sometimes the first few are trivial or noisy.
             # We'll stick with index 1 for now as it's the standard spectral bisection choice.
             pass
             
        v = eigenvectors[:, idx[vec_idx]]
        
        # Split based on sign of v
        # To ensure balanced splits (phylogenetic trees are usually roughly balanced in terms of 
        # number of taxa in subtrees, though not necessarily), we might need to adjust the threshold.
        # However, for topology recovery, the sign of the eigenvector is the standard approach.
        # Let's try to split into two non-empty sets.
        
        # If all values are positive or negative, we have a problem.
        # Subtract the mean to center.
        v_centered = v - np.mean(v)
        
        # Split: left = v_centered < 0, right = v_centered >= 0
        left_mask = v_centered < 0
        right_mask = ~left_mask
        
        # Ensure both sets are non-empty
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            # Fallback: split by median
            median_val = np.median(v_centered)
            left_mask = v_centered < median_val
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                 # Extreme fallback: arbitrary split
                 mid = m // 2
                 left_mask = np.zeros(m, dtype=bool)
                 left_mask[:mid] = True
                 right_mask = ~left_mask
        
        left_labels = subset_labels[left_mask]
        right_labels = subset_labels[right_mask]
        
        # Extract sub-matrices
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]
        
        left_dists = subset_dists[np.ix_(left_indices, left_indices)]
        right_dists = subset_dists[np.ix_(right_indices, right_indices)]
        
        # Recurse
        left_tree = build_tree(left_labels, left_dists)
        right_tree = build_tree(right_labels, right_dists)
        
        return (None, left_tree, right_tree)
        
    # Initial call
    return build_tree(labels, dists)

def _nj_build_subset(labels, dists):
    """Fallback NJ for a subset, returning the same tuple structure."""
    n = len(labels)
    if n < 3:
        if n == 1:
            return (labels[0], None, None)
        else:
            return (None, labels[0], labels[1])
            
    # Simplified NJ for subset
    current_d = dists.copy()
    active_ids = list(range(n))
    node_map = {i: (labels[i], None, None) for i in range(n)}
    node_id_counter = n
    
    while len(active_ids) > 2:
        n_curr = len(active_ids)
        r = np.sum(current_d, axis=1)
        q_matrix = (n_curr - 2) * current_d - r[:, None] - r[None, :]
        np.fill_diagonal(q_matrix, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        i, j = min_idx
        
        id_i = active_ids[i]
        id_j = active_ids[j]
        
        node_i = node_map[id_i]
        node_j = node_map[id_j]
        
        d_ij = current_d[i, j]
        alpha = (r[i] - r[j]) / (2.0 * (n_curr - 2))
        
        # Create new internal node
        new_id = node_id_counter
        node_id_counter += 1
        new_node = (None, node_i, node_j)
        node_map[new_id] = new_node
        
        new_active_ids = [k for k in range(n_curr) if k != i and k != j]
        new_active_ids.append(len(new_active_ids))
        
        n_new = len(new_active_ids)
        new_d = np.zeros((n_new, n_new))
        
        pos_to_id = {}
        for pos, old_pos in enumerate(new_active_ids[:-1]):
            pos_to_id[pos] = active_ids[old_pos]
        pos_to_id[n_new - 1] = new_id
        
        for p1 in range(n_new):
            for p2 in range(p1 + 1, n_new):
                id1 = pos_to_id[p1]
                id2 = pos_to_id[p2]
                
                if id1 == new_id or id2 == new_id:
                    if id1 == new_id:
                        x_id = id2
                    else:
                        x_id = id1
                    
                    old_pos_x = active_ids.index(x_id)
                    d_ix = current_d[i, old_pos_x]
                    d_jx = current_d[j, old_pos_x]
                    
                    d_new_x = 0.5 * (d_ix + d_jx - d_ij)
                    d_new_x = max(0.0, d_new_x)
                    
                    new_d[p1, p2] = d_new_x
                    new_d[p2, p1] = d_new_x
                else:
                    old_pos1 = active_ids.index(id1)
                    old_pos2 = active_ids.index(id2)
                    new_d[p1, p2] = current_d[old_pos1, old_pos2]
                    new_d[p2, p1] = current_d[old_pos1, old_pos2]
                    
        current_d = new_d
        active_ids = list(pos_to_id.values())
        
    if len(active_ids) == 2:
        id1 = active_ids[0]
        id2 = active_ids[1]
        node1 = node_map[id1]
        node2 = node_map[id2]
        return (None, node1, node2)
        
    return node_map[active_ids[0]]

def _tree_to_newick(node):
    """Convert tuple tree to Newick string."""
    if node is None:
        return ""
    label, left, right = node
    if label is not None:
        return label
    left_str = _tree_to_newick(left)
    right_str = _tree_to_newick(right)
    return f"({left_str},{right_str})"

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
            
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Check time
    if _time_left() < 5.0:
         # Not enough time for spectral, fallback to a very fast NJ or star tree?
         # Star tree is terrible for RF. Let's just do NJ quickly.
         # Actually, if time is tight, return a star tree? No, that scores 0.
         # Let's try NJ which is O(N^3) and fast for N<=32.
         tree = _nj_build_subset(labels, dists)
    else:
        # Use Spectral Clustering
        try:
            tree = _spectral_cluster_tree(dists, labels)
        except Exception:
            # Fallback to NJ if spectral fails
            tree = _nj_build_subset(labels, dists)
            
    # Convert to Newick
    newick = _tree_to_newick(tree)
    
    return newick + ";"

# EVOLVE-BLOCK-END
