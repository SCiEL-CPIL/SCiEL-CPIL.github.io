"""Reconstruct unrooted phylogenetic topology from simulated DNA sequences."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 255.0
_start_time = None


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    seq_list = [sequences[lab] for lab in labels]
    L = len(seq_list[0])
    
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    ints = np.array([[char_to_int[c] for c in s] for s in seq_list], dtype=np.int8)
    
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = (ints[i] != ints).sum(axis=1)
        dist[i] = diff / L
    
    p = dist.copy()
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _neighbor_joining(dist):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns a Newick string (unrooted, no branch lengths).
    Uses vectorized Q-matrix computation for speed.
    """
    n = dist.shape[0]
    if n == 1:
        return f"t{n-1};"
    if n == 2:
        return f"(t{n-2},t{n-1});"
    if n == 3:
        return f"(t0,(t1,t2));"
    
    labels_list = [f"t{i}" for i in range(n)]
    dm = dist.copy()
    
    while len(labels_list) > 2:
        m = len(labels_list)
        
        r = dm.sum(axis=1)
        Q = (m - 2) * dm - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        best_i, best_j = idx[0], idx[1]
        if best_i > best_j:
            best_i, best_j = best_j, best_i
        
        a, b = best_i, best_j
        new_label = f"({labels_list[a]},{labels_list[b]})"
        
        other_indices = [i for i in range(m) if i not in (a, b)]
        new_dists = 0.5 * (dm[a, :m] + dm[b, :m] - dm[a, b])
        new_dists_other = np.array([new_dists[k] for k in other_indices])
        
        new_m = m - 1
        new_dm = np.zeros((new_m, new_m), dtype=np.float64)
        new_labels = []
        
        for i_idx, i_orig in enumerate(other_indices):
            new_labels.append(labels_list[i_orig])
            for j_idx, j_orig in enumerate(other_indices):
                if i_idx != j_idx:
                    new_dm[i_idx, j_idx] = dm[i_orig, j_orig]
        
        new_labels.append(new_label)
        for i_idx, i_orig in enumerate(other_indices):
            new_dm[i_idx, new_m - 1] = new_dists_other[i_idx]
            new_dm[new_m - 1, i_idx] = new_dists_other[i_idx]
        
        labels_list = new_labels
        dm = new_dm
    
    return f"({labels_list[0]},{labels_list[1]});"


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree from DNA sequences.
    Returns a Newick string with taxa t0..t(n-1).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        n = len(sequences)
        labels = sorted(sequences, key=lambda x: int(x[1:]))
        return "(" + ",".join(labels) + ");"
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return f"{labels[0]};"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    dist = _jc_distance_matrix(sequences, labels)
    newick = _neighbor_joining(dist)
    
    return newick


# EVOLVE-BLOCK-END
