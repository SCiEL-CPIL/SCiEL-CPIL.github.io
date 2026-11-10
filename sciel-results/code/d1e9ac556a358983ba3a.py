"""
Reconstructs phylogenetic topology using Neighbor Joining (NJ) on Jukes-Cantor corrected distances.
Optimized with NumPy for fast distance computation and robust time management.
"""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance(a, b):
    """Calculate Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using vectorized NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Encode sequences as integer arrays for fast comparison
    encoded = []
    for label in labels:
        s = sequences[label]
        arr = np.array([ord(c) for c in s], dtype=np.uint8)
        # Map: A=65, C=67, G=71, T=84
        # Create differences with A as reference
        encoded.append(arr)
    
    dist = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Count positions where sequences differ
            diffs = np.sum(encoded[i] != encoded[j])
            p = diffs / L
            if p >= 0.75:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    
    return dist


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    start_time = time.time()
    deadline = start_time + 250.0
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    if time.time() > deadline:
        return "(" + ",".join(labels) + ");"
    
    # NJ reconstruction
    d = dist.copy()
    active = list(range(n))
    newick = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        if time.time() > deadline:
            break
        
        n_active = len(active)
        
        # Compute r[i] = sum of distances from i to all other active nodes
        r = np.zeros(n_active)
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if i != j:
                    r[idx_i] += d[i, j]
        
        # Find pair minimizing Q(i,j) = (n-2)*d(i,j) - r[i] - r[j]
        best_val = float('inf')
        best_i = None
        best_j = None
        
        for ui in range(n_active):
            i = active[ui]
            for uj in range(ui + 1, n_active):
                j = active[uj]
                criterion = (n_active - 2) * d[i, j] - r[ui] - r[uj]
                if criterion < best_val:
                    best_val = criterion
                    best_i = i
                    best_j = j
        
        if best_i is None:
            break
        
        i, j = best_i, best_j
        new_id = next_id
        next_id += 1
        
        newick[new_id] = f"({newick[i]},{newick[j]})"
        
        d_ij = d[i, j]
        for x in active:
            if x == i or x == j:
                continue
            d_nx = 0.5 * (d[i, x] + d[j, x] - d_ij)
            if d_nx < 0:
                d_nx = 0.0
            d[new_id, x] = d_nx
            d[x, new_id] = d_nx
        
        active.remove(i)
        active.remove(j)
        active.append(new_id)
    
    # Assemble final tree
    if len(active) == 3:
        a, b, c = active
        return f"({newick[a]},{newick[b]},{newick[c]});"
    else:
        # Fallback: complete with closest pairs
        while len(active) > 3:
            if time.time() > deadline:
                break
            best_dist = float('inf')
            best_i = active[0]
            best_j = active[1]
            for ui in range(len(active)):
                i = active[ui]
                for uj in range(ui + 1, len(active)):
                    j = active[uj]
                    if i < n and j < n:
                        dd = dist[i, j]
                    else:
                        dd = 0.0
                    if dd < best_dist:
                        best_dist = dd
                        best_i = i
                        best_j = j
            
            new_id = max(newick.keys()) + 1
            newick[new_id] = f"({newick[best_i]},{newick[best_j]})"
            active.remove(best_i)
            active.remove(best_j)
            active.append(new_id)
        
        a, b, c = active[:3]
        return f"({newick[a]},{newick[b]},{newick[c]});"

# EVOLVE-BLOCK-END
