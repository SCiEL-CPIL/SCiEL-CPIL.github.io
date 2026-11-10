"""Phylogenetic topology recovery from simulated sequences (Jukes-Cantor)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations

# Deadline guard: evaluator kills after 270s wall-clock; leave safety margin
_DEADLINE = 265.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distance matrix using NumPy."""
    n = len(labels)
    # Convert sequences to integer arrays: A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arr = np.array([[char_map[c] for c in sequences[lab]] for lab in labels], dtype=np.int8)
    length = seq_arr.shape[1]
    
    # Compute pairwise Hamming distances using broadcasting
    # seq_arr shape: (n, length)
    # diff: (n, n, length) would be too large for large n*length, use chunked approach
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Compare seq_arr[i] with all others
        diffs = (seq_arr[i] != seq_arr).sum(axis=1)
        p = diffs / length
        # Apply JC correction
        p = np.clip(p, 0, 0.7499)
        d[i] = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return d


def _nj_reconstruct(d, labels):
    """Neighbour-Joining algorithm. Returns Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # d is n x n numpy array
    d = d.copy()
    trees = list(labels)
    active = list(range(n))
    n_active = n
    
    while n_active > 3:
        if _remaining() < 5.0:
            # Not enough time; return a star tree
            return _star_tree(trees, active)
        
        # Compute row sums
        row_sums = d[np.ix_(active, active)].sum(axis=1)
        
        # Find best pair to join using NJ criterion
        best_score = float('inf')
        best_i = best_j = -1
        for ii in range(n_active):
            for jj in range(ii + 1, n_active):
                i, j = active[ii], active[jj]
                m = n_active
                score = (m - 2) * d[i, j] - row_sums[ii] - row_sums[jj]
                if score < best_score:
                    best_score = score
                    best_i, best_j = i, j
        
        # Join best_i and best_j
        i, j = best_i, best_j
        # Compute branch lengths
        r = (row_sums[active.index(i)] - row_sums[active.index(j)]) / (2 * (n_active - 2))
        li = 0.5 * d[i, j] + r
        lj = 0.5 * d[i, j] - r
        
        # Create new node
        new_idx = n  # use index beyond original
        new_tree = f"({trees[i]},{trees[j]})"
        
        # Compute distances from new node to all other active nodes
        others = [k for k in active if k not in (i, j)]
        for k in others:
            dk = 0.5 * (d[i, k] + d[j, k] - d[i, j])
            d[new_idx, k] = dk
            d[k, new_idx] = dk
        d[new_idx, new_idx] = 0.0
        
        # Remove i and j, add new node
        active = others + [new_idx]
        trees.append(new_tree)
        n_active = len(active)
    
    # Final 3 nodes: make a trichotomy or resolve
    a, b, c = active
    return f"({trees[a]},({trees[b]},{trees[c]}));"


def _star_tree(trees, active):
    """Create a star tree from remaining nodes."""
    if len(active) == 0:
        return ""
    if len(active) == 1:
        return trees[active[0]] + ";"
    if len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    inner = ",".join(trees[k] for k in active)
    return f"({inner});"


def _upgma_reconstruct(d, labels):
    """UPGMA as a fallback (unrooted topology from rooted tree)."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    
    d = d.copy()
    trees = list(labels)
    active = list(range(n))
    
    while len(active) > 2:
        if _remaining() < 5.0:
            return _star_tree(trees, active)
        
        # Find closest pair
        best_dist = float('inf')
        best_i = best_j = -1
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                i, j = active[ii], active[jj]
                if d[i, j] < best_dist:
                    best_dist = d[i, j]
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        new_idx = len(trees)
        new_tree = f"({trees[i]},{trees[j]})"
        
        others = [k for k in active if k not in (i, j)]
        for k in others:
            dk = 0.5 * (d[i, k] + d[j, k])
            d[new_idx, k] = dk
            d[k, new_idx] = dk
        d[new_idx, new_idx] = 0.0
        
        active = others + [new_idx]
        trees.append(new_tree)
    
    a, b = active
    return f"({trees[a]},{trees[b]});"


def _bootstrap_tree(d, labels, n_bootstrap=50):
    """Rapid bootstrap: resample sites is not possible here (we only have distances).
    Instead, use distance perturbation to assess split support.
    Returns a consensus-like tree based on NJ on perturbed distances."""
    # Since we don't have raw sequences at this level, we'll just do NJ
    # This is a placeholder; in practice we'd need the sequences for bootstrapping
    return _nj_reconstruct(d, labels)


def _distance_matrix_from_sequences(sequences, labels):
    """Compute JC distance matrix directly from sequences using vectorized ops."""
    n = len(labels)
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arr = np.array([[char_map[c] for c in sequences[lab]] for lab in labels], dtype=np.int8)
    length = seq_arr.shape[1]
    
    d = np.zeros((n, n), dtype=np.float64)
    # Process in chunks to avoid memory issues
    chunk_size = max(1, min(n, 1024))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # seq_arr[start:end] has shape (chunk, length)
        # Compare with all: (chunk, n, length) might be large
        # Use row-by-row for safety
        for idx in range(start, end):
            diffs = (seq_arr[idx] != seq_arr).sum(axis=1)
            p = diffs / length
            p = np.clip(p, 0, 0.7499)
            d[idx] = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return d


def _nj_fast(d, labels):
    """Optimized NJ using NumPy for the inner loop."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    d = d.copy()
    trees = list(labels)
    active = list(range(n))
    n_active = n
    
    while n_active > 3:
        if _remaining() < 3.0:
            return _star_tree(trees, active)
        
        # Compute row sums for active nodes
        active_arr = np.array(active)
        sub_d = d[np.ix_(active_arr, active_arr)]
        row_sums = sub_d.sum(axis=1)
        
        # Compute NJ criterion matrix: (n-2)*d[i,j] - r[i] - r[j]
        m = n_active
        # Create matrix of (m-2)*d[i,j]
        criterion = (m - 2) * sub_d - row_sums[:, np.newaxis] - row_sums[np.newaxis, :]
        # Set diagonal to inf
        np.fill_diagonal(criterion, float('inf'))
        
        # Find minimum
        min_idx = np.unravel_index(np.argmin(criterion), criterion.shape)
        ii, jj = min_idx
        if ii > jj:
            ii, jj = jj, ii
        
        i, j = active[ii], active[jj]
        
        # Compute branch lengths
        r = (row_sums[ii] - row_sums[jj]) / (2 * (n_active - 2))
        li = 0.5 * d[i, j] + r
        lj = 0.5 * d[i, j] - r
        
        # Create new node
        new_idx = len(trees)
        new_tree = f"({trees[i]},{trees[j]})"
        
        # Compute distances from new node to all other active nodes
        others_idx = [k for k in range(n_active) if k not in (ii, jj)]
        for k_idx in others_idx:
            k = active[k_idx]
            dk = 0.5 * (d[i, k] + d[j, k] - d[i, j])
            d[new_idx, k] = dk
            d[k, new_idx] = dk
        d[new_idx, new_idx] = 0.0
        
        # Update active
        active = [active[k] for k in others_idx] + [new_idx]
        trees.append(new_tree)
        n_active = len(active)
    
    a, b, c = active
    return f"({trees[a]},({trees[b]},{trees[c]}));"


def _bionj_reconstruct(d, labels):
    """BioNJ: NJ with branch length correction (Felsenstein 1993).
    Iteratively adjusts distances using current tree estimates."""
    n = len(labels)
    if n <= 3:
        return _nj_reconstruct(d, labels)
    
    # Start with NJ tree
    tree_newick = _nj_fast(d, labels)
    
    # For BioNJ, we need to extract branch lengths and re-estimate distances
    # This is complex; for now, return NJ result
    # A proper BioNJ would iterate, but given time constraints, NJ is our base
    return tree_newick


def _parsimony_tree(sequences, labels):
    """Simple parsimony-based tree construction as an alternative signal.
    Uses Fitch parsimony to score splits, but for tree construction we use
    a greedy approach based on pairwise compatibility."""
    # This is a fallback; in practice NJ is usually better for JC model
    # Return a star tree as placeholder
    return _star_tree(labels, list(range(len(labels))))


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0, t1, ...) to DNA strings (ACGT)
    
    Returns:
        Newick string with the reconstructed tree
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return (labels[0] + ";") if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Compute distance matrix
    d = _distance_matrix_from_sequences(sequences, labels)
    
    # Primary method: NJ with NumPy optimization
    tree = _nj_fast(d, labels)
    
    # If we have time, we could try additional methods, but NJ is strong for JC model
    # Verify the tree has all taxa
    tree_labels = set()
    for lab in labels:
        if lab in tree:
            tree_labels.add(lab)
    
    # Fallback if something went wrong
    if len(tree_labels) != n:
        tree = _star_tree(labels, list(range(n)))
    
    return tree
# EVOLVE-BLOCK-END
