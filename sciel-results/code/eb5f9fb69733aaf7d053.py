"""Neighbour joining on JC-corrected distances with NumPy acceleration and time guard."""
import math
import time
import numpy as np


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.array([[enc[c] for c in sequences[lab]] for lab in labels], dtype=np.int8)
    
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = int(np.count_nonzero(mat[i] != mat[j]))
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
                if d < 0:
                    d = 0.0
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_tree(dist, labels):
    """Run Neighbor-Joining, return Newick string."""
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    D = dist.copy()
    trees = {i: labels[i] for i in range(n)}
    active = list(range(n))
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums for active nodes
        row_sums = np.zeros(m)
        for ii in range(m):
            a = active[ii]
            s = 0.0
            for jj in range(m):
                if jj != ii:
                    s += D[a, active[jj]]
            row_sums[ii] = s
        
        best_val = float('inf')
        best_ii = 0
        best_jj = 1
        for ii in range(m):
            for jj in range(ii + 1, m):
                a = active[ii]
                b = active[jj]
                crit = (m - 2) * D[a, b] - row_sums[ii] - row_sums[jj]
                if crit < best_val:
                    best_val = crit
                    best_ii = ii
                    best_jj = jj
        
        a = active[best_ii]
        b = active[best_jj]
        
        joined = f"({trees[a]},{trees[b]})"
        new_node = -1 - len(active)
        trees[new_node] = joined
        
        rest = [x for x in active if x not in (a, b)]
        new_active = rest + [new_node]
        
        for x in rest:
            D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[x, new_node] = D[new_node, x]
        
        active = new_active
    
    return "(" + ",".join(trees[x] for x in active) + ");"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Check time
    if time.time() > deadline:
        # Fallback: star tree
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    newick = _nj_tree(dist, labels)
    
    return newick
