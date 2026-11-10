"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Vectorized Neighbor Joining on Jukes-Cantor corrected distances with robust deadline guard.'''
import time
import numpy as np
import re

_DEADLINE_START = None
_DEADLINE_LIMIT = 260.0  # Leave margin before 270s kill


def _elapsed():
    if _DEADLINE_START is None:
        return 0.0
    return time.time() - _DEADLINE_START


def _jc_distances(mat):
    """
    Compute Jukes-Cantor corrected distance matrix.
    mat: (n, L) int array with 0..3 encoding (A=0, C=1, G=2, T=3).
    Returns: (n, n) float array of JC distances.
    """
    n, L = mat.shape
    if L == 0:
        return np.zeros((n, n))
    
    # Compute pairwise Hamming distances using one-hot encoding matrix multiplication
    # matches[i,j] = sum over sites of (base_i == base_j)
    # We can compute this by summing over each base c: (mat==c) @ (mat==c).T
    matches = np.zeros((n, n), dtype=np.float32)
    for c in range(4):
        Mc = (mat == c).astype(np.float32)
        matches += Mc @ Mc.T
    
    # Mismatches = L - matches
    mismatches = L - matches
    p = mismatches / L
    np.fill_diagonal(p, 0.0)
    
    # JC correction: d = -0.75 * ln(1 - 4p/3)
    # Clip p to avoid log(0) or log(negative)
    p_clipped = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Handle edge cases where p might be exactly 0.75 due to floating point
    d = np.where(p >= 0.75, 3.0, d)
    d = np.where(np.isnan(d) | np.isinf(d), 3.0, d)
    
    np.fill_diagonal(d, 0.0)
    return d


def _nj_with_structure(dist, labels):
    """
    Neighbor Joining algorithm.
    dist: (n, n) distance matrix.
    labels: list of n label strings corresponding to rows/cols of dist.
    Returns: Newick string.
    """
    n0 = dist.shape[0]
    if n0 < 3:
        if n0 == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
        
    # Pre-allocate full (2n-1) x (2n-1) distance matrix
    max_nodes = 2 * n0 - 1
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n0, :n0] = dist
    
    # Track which nodes are "alive" (not yet joined)
    alive = list(range(n0))
    next_id = n0
    
    # Newick subtree string for each node index
    # Initialize with leaf labels
    subtree = [""] * max_nodes
    for i in range(n0):
        subtree[i] = labels[i]
        
    while len(alive) > 3:
        if _elapsed() > _DEADLINE_LIMIT:
            # Deadline exceeded: return star tree of remaining nodes
            return "(" + ",".join(subtree[x] for x in alive) + ");"
            
        k = len(alive)
        idx = np.array(alive, dtype=np.int64)
        
        # Extract sub-matrix of alive nodes
        sub = D[np.ix_(idx, idx)].copy()
        
        # Compute row sums (total distance to all other alive nodes)
        r = sub.sum(axis=1)
        
        # NJ criterion: Q[i,j] = (k-2)*D[i,j] - r[i] - r[j]
        Q = (k - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find minimum Q entry (i, j)
        min_idx = np.argmin(Q)
        i, j = np.unravel_index(min_idx, Q.shape)
        a = alive[i]
        b = alive[j]
        
        # Create new internal node
        new = next_id
        next_id += 1
        subtree[new] = "(" + subtree[a] + "," + subtree[b] + ")"
        
        # Update distances from new node to all other alive nodes
        others = [x for x in alive if x != a and x != b]
        if others:
            ox = np.array(others, dtype=np.int64)
            d_ab = D[a, b]
            vals = 0.5 * (D[a, ox] + D[b, ox] - d_ab)
            D[new, ox] = vals
            D[ox, new] = vals
            
        # Remove a and b from alive, add new
        alive = [x for x in alive if x != a and x != b]
        alive.append(new)
        
    # Final: 3 nodes remain, form a trifurcation
    if len(alive) == 3:
        a, b, c = alive
        return "(" + subtree[a] + "," + subtree[b] + "," + subtree[c] + ");"
    # Fallback (shouldn't happen if n>=3)
    return "(" + ",".join(subtree[x] for x in alive) + ");"


def _reconstruct_single(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 260.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n < 3:
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
        
    # Encode sequences: map A,C,G,T -> 0,1,2,3
    L = len(sequences[labels[0]])
    if L == 0:
        # No data, return star tree
        return "(" + ",".join(labels) + ");"
        
    enc = np.empty((n, L), dtype=np.int8)
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Vectorized encoding for speed
        enc[i] = np.vectorize(code.get)(s).astype(np.int8)
        
    D = _jc_distances(enc)
    
    # Check deadline before expensive NJ
    if _elapsed() > _DEADLINE_LIMIT:
        return "(" + ",".join(labels) + ");"
        
    newick = _nj_with_structure(D, labels)
    return newick


def reconstruct(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 260.0

    try:
        return _reconstruct_single(sequences)
    except Exception:
        # Fallback: star tree with all labels
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        if len(labels) <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
