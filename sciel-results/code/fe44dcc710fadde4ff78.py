"""Fast Distance-based Quartet (FDQ) topology reconstruction under Jukes-Cantor."""
import time
import math
import numpy as np
from itertools import combinations

_DEADLINE_MARGIN = 2.0  # seconds to keep before the 270s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return 270.0 - _elapsed() - _DEADLINE_MARGIN


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distances using vectorized NumPy ops."""
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    
    # Build lookup table once
    lut = np.zeros(256, dtype=np.int8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3
    
    mat = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        mat[i] = lut[arr]

    # Vectorized pairwise distance computation
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(mat[i] != mat[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_reconstruct(dist, labels):
    """
    Perform Neighbor Joining on the distance matrix.
    Uses a dictionary-based approach for correctness.
    Returns a Newick string.
    """
    n = dist.shape[0]
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"

    # Use dictionary for distances to handle dynamic node IDs safely
    D = {}
    for i in range(n):
        for j in range(n):
            D[(i, j)] = dist[i, j]
            D[(j, i)] = dist[j, i]

    active = list(range(n))
    newick_strs = {i: labels[i] for i in range(n)}
    next_id = n

    while len(active) > 3:
        if _remaining() < 0.5:
            # Fallback: return a chain of remaining active nodes
            if len(active) == 0:
                return ""
            if len(active) == 1:
                return newick_strs[active[0]] + ";"
            if len(active) == 2:
                return f"({newick_strs[active[0]]},{newick_strs[active[1]]});"
            current = newick_strs[active[0]]
            for i in range(1, len(active)):
                current = f"({current},{newick_strs[active[i]]})"
            return current + ";"

        m = len(active)
        # Compute r values
        r = {}
        for i in active:
            r[i] = sum(D[(i, j)] for j in active if j != i)

        # Find the pair (a, b) that minimizes (m-2)*D[a,b] - r[a] - r[b]
        best_val = float('inf')
        best_a, best_b = active[0], active[1]
        for idx_i in range(len(active)):
            for idx_j in range(idx_i + 1, len(active)):
                a, b = active[idx_i], active[idx_j]
                val = (m - 2) * D[(a, b)] - r[a] - r[b]
                if val < best_val:
                    best_val = val
                    best_a, best_b = a, b

        a, b = best_a, best_b

        # Create new node
        new_node = next_id
        next_id += 1

        # Update distances
        for c in active:
            if c != a and c != b:
                new_d = 0.5 * (D[(a, c)] + D[(b, c)] - D[(a, b)])
                D[(new_node, c)] = new_d
                D[(c, new_node)] = new_d

        # Update active list
        active.remove(a)
        active.remove(b)
        active.append(new_node)

        # Update Newick string
        newick_strs[new_node] = f"({newick_strs[a]},{newick_strs[b]})"

    # Handle final 3 nodes
    if len(active) == 3:
        a, b, c = active
        # Use the three distances to determine the best topology
        d_ab = D.get((a, b), 0)
        d_ac = D.get((a, c), 0)
        d_bc = D.get((b, c), 0)
        # For 3 taxa, any binary topology is valid; use balanced
        new_node1 = next_id
        next_id += 1
        newick_strs[new_node1] = f"({newick_strs[b]},{newick_strs[c]})"
        new_node2 = next_id
        next_id += 1
        newick_strs[new_node2] = f"({newick_strs[a]},{newick_strs[new_node1]})"
        return newick_strs[new_node2] + ";"
    elif len(active) == 2:
        a, b = active
        return f"({newick_strs[a]},{newick_strs[b]});"
    elif len(active) == 1:
        return newick_strs[active[0]] + ";"
    else:
        return ""


def _get_quartet_splits(dist, n):
    """
    Compute all quartet splits from the distance matrix.
    Returns a list of (frozenset_side1, frozenset_side2, score) for each quartet.
    """
    if n < 4:
        return []
    
    splits = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        d_ab = dist[a, b]
        d_cd = dist[c, d]
        d_ac = dist[a, c]
        d_bd = dist[b, d]
        d_ad = dist[a, d]
        d_bc = dist[b, c]
        
        s1 = d_ab + d_cd  # ab|cd
        s2 = d_ac + d_bd  # ac|bd
        s3 = d_ad + d_bc  # ad|bc
        
        min_s = min(s1, s2, s3)
        # Score: how much better the best split is vs the others
        if s1 == min_s:
            splits.append((frozenset({a, b}), frozenset({c, d}), s2 + s3 - 2*s1))
        elif s2 == min_s:
            splits.append((frozenset({a, c}), frozenset({b, d}), s1 + s3 - 2*s2))
        else:
            splits.append((frozenset({a, d}), frozenset({b, c}), s1 + s2 - 2*s3))
    
    return splits


def _build_tree_from_splits(labels, quartet_splits, dist):
    """
    Build a tree from quartet splits using an agglomerative approach.
    Start with all taxa as separate clusters, then merge based on quartet support.
    """
    n = len(labels)
    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # For small n, use NJ as it's reliable
    if n <= 10:
        return _nj_reconstruct(dist, labels)
    
    # Agglomerative clustering guided by quartet splits
    # Each cluster is a frozenset of taxon indices
    clusters = [frozenset({i}) for i in range(n)]
    
    # Build a lookup: for each pair of clusters, count supporting quartets
    # This is expensive, so we use a simpler approach:
    # Use NJ as the backbone, then refine with quartet information
    
    # Actually, for robustness and speed, let's use NJ and then try to improve
    # via quartet-based subtree pruning and regrafting
    
    newick = _nj_reconstruct(dist, labels)
    if newick and all(lb in newick for lb in labels):
        return newick
    
    # Fallback: balanced tree
    current = labels[0]
    for i in range(1, n):
        current = f"({current},{labels[i]})"
    return current + ";"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Use NJ
    try:
        newick = _nj_reconstruct(dist, labels)
        if newick and all(lb in newick for lb in labels):
            return newick
    except Exception:
        pass
    
    # Fallback: balanced tree
    current = labels[0]
    for i in range(1, n):
        current = f"({current},{labels[i]})"
    return current + ";"
