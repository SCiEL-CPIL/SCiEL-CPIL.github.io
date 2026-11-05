"""Neighbour joining on Jukes-Cantor corrected distances with NumPy acceleration."""
import time
import math
import numpy as np

_DEADLINE = 260.0  # seconds, safety margin before 270s kill
_start_time = None


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    seqs = np.array([list(sequences[l]) for l in labels], dtype=np.uint8)
    # Map A,C,G,T to 0,1,2,3
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.vectorize(mapping.get)(seqs)
    
    # Compute pairwise Hamming distances
    # For each pair, count mismatches
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.count_nonzero(seqs[i] != seqs[j])
            p = mismatches / len(seqs[0])
            if p >= 0.7499:
                d = 3.0
            elif p > 0:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            else:
                d = 0.0
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def _nj_build(dist_matrix, labels):
    """Neighbor-Joining algorithm using NumPy for efficiency."""
    n = len(labels)
    if n <= 2:
        return ""
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Track which nodes are still active
    active = list(range(n))
    # Tree structure for each node
    tree = {i: labels[i] for i in range(n)}
    # Current distance matrix
    D = dist_matrix.copy()
    
    while len(active) > 3:
        m = len(active)
        # Compute r values: r[i] = sum of distances from i to all other active nodes
        r = np.array([np.sum([D[active[i], active[j]] for j in range(m) if j != i]) for i in range(m)])
        
        # Find the pair (a, b) that minimizes: (m-2)*D[a,b] - r[a] - r[b]
        best_val = float('inf')
        best_a, best_b = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * D[active[i], active[j]] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_a, best_b = i, j
        
        a, b = active[best_a], active[best_b]
        
        # Compute branch lengths
        # d_ab = D[a,b]
        # d_ax = (r[a] - r[b]) / (2*(m-2)) + D[a,b]/2
        # d_bx = D[a,b] - d_ax
        d_ab = D[a, b]
        if m > 2:
            d_ax = (r[best_a] - r[best_b]) / (2.0 * (m - 2)) + d_ab / 2.0
        else:
            d_ax = d_ab / 2.0
        d_bx = d_ab - d_ax
        
        # Create new node
        new_node = n
        n += 1
        tree[new_node] = f"({tree[a]},{tree[b]})"
        
        # Update distance matrix for the new node
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        for x in active:
            if x not in (a, b):
                D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
                D[x, new_node] = D[new_node, x]
        
        # Remove a and b from active, add new_node
        active.remove(a)
        active.remove(b)
        active.append(new_node)
    
    # Final tree with 3 nodes
    if len(active) == 3:
        a, b, c = active
        # Get distances
        d_ab = D[a, b]
        d_ac = D[a, c]
        d_bc = D[b, c]
        # Compute branch lengths for the 3-node tree
        # Using the standard formula
        l_a = (d_ab + d_ac - d_bc) / 2.0
        l_b = (d_ab + d_bc - d_ac) / 2.0
        l_c = (d_ac + d_bc - d_ab) / 2.0
        # Ensure non-negative
        l_a = max(0, l_a)
        l_b = max(0, l_b)
        l_c = max(0, l_c)
        return f"({tree[a]},{tree[b]},{tree[c]});"
    
    return ""


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a star tree as fallback
        return "(" + ",".join(labels) + ");"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    # Build tree using NJ
    result = _nj_build(D, labels)
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    return result
