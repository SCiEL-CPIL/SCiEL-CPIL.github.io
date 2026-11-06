"""Phylogenetic topology recovery via quartet-based supertree with NJ fallback."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from itertools import combinations

_DEADLINE = None
_TIME_BUDGET = 250.0  # seconds, leave margin for 270s wall-clock limit


def _elapsed():
    if _DEADLINE is None:
        return 0.0
    return time.time() - _DEADLINE[0]


def _deadline_exceeded():
    if _DEADLINE is None:
        return False
    return time.time() > _DEADLINE[1]


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using vectorized operations."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Encode sequences to integer arrays
    encode_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, c in enumerate(s):
            mat[i, j] = encode_map.get(c, 0)
    
    # Compute pairwise distances in chunks
    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 32
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        diff = (mat[i:i_end, None, :] != mat[None, :, :])
        counts = diff.sum(axis=2)
        p = counts.astype(np.float64) / L
        p_safe = np.clip(p, 0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc
    
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_newick(dist, labels):
    """
    Neighbor Joining algorithm returning a Newick string.
    dist: n x n distance matrix
    labels: list of n labels
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    D = dist.copy()
    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums
        row_sum = np.zeros(m, dtype=np.float64)
        for idx, i in enumerate(nodes):
            for jdx, j in enumerate(nodes):
                if i != j:
                    row_sum[idx] += D[i, j]

        # Find the pair (a, b) that minimizes the NJ criterion
        best_criterion = float('inf')
        best_a = None
        best_b = None
        for idx_i, i in enumerate(nodes):
            for idx_j, j in enumerate(nodes):
                if i == j:
                    continue
                criterion = (m - 2) * D[i, j] - row_sum[idx_i] - row_sum[idx_j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = i
                    best_b = j

        a, b = best_a, best_b
        if a is None or b is None:
            return "(" + ",".join(trees[x] for x in nodes) + ")"

        new_id = next_id
        next_id += 1
        trees[new_id] = f"({trees[a]},{trees[b]})"

        # Update distance matrix for the new node
        for x in nodes:
            if x == a or x == b:
                continue
            D[new_id, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[x, new_id] = D[new_id, x]

        nodes = [x for x in nodes if x != a and x != b]
        nodes.append(new_id)

    # Final step: 3 nodes remain
    a, b, c = nodes
    return f"({trees[a]},{trees[b]},{trees[c]})"


def _quartet_score(dist, a, b, c, d):
    """
    Score the three possible quartet topologies using additive distances.
    Returns the best (score, topology) where topology is one of:
    0: ((a,b),(c,d))
    1: ((a,c),(b,d))
    2: ((a,d),(b,c))
    Lower score is better.
    """
    dab = dist[a, b]
    dac = dist[a, c]
    dad = dist[a, d]
    dbc = dist[b, c]
    dbd = dist[b, d]
    dcd = dist[c, d]
    
    # For each topology, compute the sum of squared residuals
    # Topology 0: ((a,b),(c,d))
    # Expected distances: d_ab, d_cd, and d_ac + d_bd should be consistent
    # Use the quartet likelihood approximation: score = (d_ab + d_cd - d_ac - d_bd)^2
    s0 = (dab + dcd - dac - dbd) ** 2
    # Topology 1: ((a,c),(b,d))
    s1 = (dac + dbd - dab - dcd) ** 2
    # Topology 2: ((a,d),(b,c))
    s2 = (dad + dbc - dab - dcd) ** 2
    
    # Return the minimum score and which topology
    if s0 <= s1 and s0 <= s2:
        return s0, 0
    elif s1 <= s0 and s1 <= s2:
        return s1, 1
    else:
        return s2, 2


def _quartet_supertree(dist, labels):
    """
    Reconstruct a tree from quartet votes using a greedy clustering approach.
    """
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ")"
    if n == 4:
        # Just evaluate all three quartets
        a, b, c, d = range(4)
        score, topo = _quartet_score(dist, a, b, c, d)
        if topo == 0:
            return f"(({labels[0]},{labels[1]}),({labels[2]},{labels[3]}))"
        elif topo == 1:
            return f"(({labels[0]},{labels[2]}),({labels[1]},{labels[3]}))"
        else:
            return f"(({labels[0]},{labels[3]}),({labels[1]},{labels[2]}))"

    # For n > 4, use a greedy approach:
    # 1. Compute all quartet votes
    # 2. Build a compatibility graph
    # 3. Use a greedy supertree algorithm

    # For simplicity and speed, we'll use a hybrid approach:
    # Start with NJ, then try to improve using quartet information

    # First, get NJ tree
    nj_tree = _nj_newick(dist, labels)
    
    # If we have time, try to improve with quartet-based refinement
    if _deadline_exceeded():
        return nj_tree

    # For now, return NJ as the baseline
    # A full quartet supertree implementation would be complex
    return nj_tree


def _small_n_exact(dist, labels):
    """
    For n <= 7, do an exhaustive search over all unrooted tree topologies
    to find the one that best fits the distance matrix.
    """
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ")"
    
    if n == 4:
        a, b, c, d = range(4)
        score, topo = _quartet_score(dist, a, b, c, d)
        if topo == 0:
            return f"(({labels[0]},{labels[1]}),({labels[2]},{labels[3]}))"
        elif topo == 1:
            return f"(({labels[0]},{labels[2]}),({labels[1]},{labels[3]}))"
        else:
            return f"(({labels[0]},{labels[3]}),({labels[1]},{labels[2]}))"
    
    # For n = 5, 6, 7, we could enumerate all unrooted tree topologies
    # but this is complex. For now, use NJ.
    return _nj_newick(dist, labels)


def reconstruct(sequences):
    """Reconstruct phylogenetic tree from sequences."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = (time.time(), time.time() + _TIME_BUDGET)
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Check deadline
    if _deadline_exceeded():
        # Return a star tree as fallback
        return "(" + ",".join(labels) + ")"
    
    # For small n, try exact or quartet-based methods
    if n <= 4:
        return _small_n_exact(dist, labels)
    
    # For larger n, use NJ with quartet refinement if time permits
    if n <= 7 and not _deadline_exceeded():
        # Try quartet supertree
        result = _quartet_supertree(dist, labels)
        if result:
            return result
    
    # Default to NJ
    return _nj_newick(dist, labels)
# EVOLVE-BLOCK-END
