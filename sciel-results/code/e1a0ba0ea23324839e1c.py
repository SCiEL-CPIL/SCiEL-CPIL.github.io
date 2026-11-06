"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math
import time
import numpy as np

_DEADLINE = 265.0  # safety margin before 270s kill
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _jc_distance_matrix(sequences, labels):
    """Build pairwise JC-corrected distance matrix using NumPy."""
    n = len(labels)
    # Encode each sequence as int array for fast comparison
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    # Convert to byte arrays for fast XOR comparison
    arr = np.array([list(s.encode()) for s in seqs], dtype=np.uint8)  # shape (n, L)
    # Pairwise Hamming distances: count positions where arr[i] != arr[j]
    # Use broadcasting: arr[:, None, :] vs arr[None, :, :] -> (n, n, L)
    # But for n=32, L=2000, that's 32*32*2000 = 2M elements, fine.
    diff = (arr[:, None, :] != arr[None, :, :]).sum(axis=2).astype(np.float64)
    p = diff / L
    # JC correction: -0.75 * ln(1 - 4p/3), cap at 3.0 for p >= 0.75
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    d = np.where(p >= 0.7499, 3.0, d)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_newick(dist, labels):
    """Run Neighbor Joining on a distance matrix (numpy array) and return Newick string."""
    n = len(labels)
    # Map label -> index
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    # Current nodes: list of (index, newick_string)
    # We'll work with a mutable list of node indices and a dict of newick strings
    active = list(range(n))
    newick = {i: labels[i] for i in range(n)}
    # dist is n x n numpy array; we'll maintain a dict for active nodes
    # For efficiency with small n, use a 2D list
    D = dist.tolist()
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums
        r = {a: 0.0 for a in active}
        for a in active:
            for b in active:
                if a != b:
                    r[a] += D[a][b]
        # Find pair with minimum criterion
        best_c = float('inf')
        best_a = best_b = None
        for i_idx, a in enumerate(active):
            for b in active[i_idx + 1:]:
                c = (m - 2) * D[a][b] - r[a] - r[b]
                if c < best_c:
                    best_c = c
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        # Create new node
        new_idx = max(active) + 1
        newick[new_idx] = f"({newick[a]},{newick[b]})"
        
        # Update distances to new node
        for x in active:
            if x != a and x != b:
                D[new_idx][x] = 0.5 * (D[a][x] + D[b][x] - D[a][b])
                D[x][new_idx] = D[new_idx][x]
        
        # Remove a and b from active, add new_idx
        active = [x for x in active if x != a and x != b] + [new_idx]
    
    # Build final newick from remaining 3 nodes
    parts = [newick[x] for x in active]
    return "(" + ",".join(parts) + ");"


def _fallback_newick(labels):
    """Star tree fallback."""
    return "(" + ",".join(labels) + ");"


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    
    if n == 3:
        return "(" + ",".join(labels) + ");"
    
    # Check deadline before heavy computation
    if _elapsed() > _DEADLINE - 5:
        return _fallback_newick(labels)
    
    try:
        dist = _jc_distance_matrix(sequences, labels)
        if _elapsed() > _DEADLINE - 3:
            return _fallback_newick(labels)
        newick = _nj_newick(dist, labels)
        return newick
    except Exception:
        return _fallback_newick(labels)
# EVOLVE-BLOCK-END
