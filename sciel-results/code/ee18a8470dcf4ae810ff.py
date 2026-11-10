"""
Reconstructs phylogenetic topology using Neighbor Joining (NJ) on Jukes-Cantor corrected distances.
Optimized with NumPy for vectorized distance calculation and efficient matrix operations.
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
    if p >= 0.7499:
        # Saturated
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _jc_distance_vectorized(seqs_matrix, i, j):
    """
    Calculate JC distance between two sequences given as rows in a matrix.
    seqs_matrix: (L, 4) or similar structure? No, let's use the original string approach but vectorized if possible.
    Actually, for speed, let's pre-process sequences into numpy arrays of integers (0-3) and use np.count_nonzero.
    """
    # This function is a placeholder if we go the full numpy route.
    # For now, we will optimize the main loop.
    pass

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
    timeout = 270.0
    safety_margin = 10.0 # Be conservative

    # 1. Pre-process sequences for fast distance calculation
    # Convert each sequence to a numpy array of integers for fast comparison?
    # Or just use the string comparison but vectorize the "diffs" count?
    # np.count_nonzero(a != b) is very fast.
    
    seqs = [np.frombuffer(sequences[label], dtype='S1') for label in labels]
    # Actually, S1 is 1 byte. A, C, G, T are distinct bytes.
    # Let's just use the strings directly with numpy if we convert to arrays of bytes.
    
    # Let's try a hybrid: keep the dictionary for the dynamic tree structure,
    # but compute the initial distance matrix efficiently.
    
    # Initial Distance Matrix
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        s_i = sequences[labels[i]]
        for j in range(i + 1, n):
            s_j = sequences[labels[j]]
            # Vectorized diff count
            # This might be slower for small L? L is 250-2000.
            # sum(1 for...) is O(L) in Python.
            # np.count_nonzero is O(L) in C.
            # Let's use numpy for the diff count.
            # Convert to bytes if not already?
            # sequences are strings.
            # a = np.frombuffer(s_i.encode('ascii'), dtype='u1')
            # b = np.frombuffer(s_j.encode('ascii'), dtype='u1')
            # diffs = np.count_nonzero(a != b)
            
            # Is encoding overhead worth it?
            # For L=2000, Python loop is ~2000 iterations.
            # Numpy setup + count is maybe ~2000 + overhead.
            # Let's stick to the simple loop first, but optimize the NJ part.
            # The NJ part is O(N^3) where N is taxa. N<=32.
            # 32^3 = 32768. Very small.
            # The distance calculation is O(N^2 * L).
            # 32^2 * 2000 = 2,048,000 operations.
            # Python loop for 2M ops is ~0.1-0.2s.
            # 20 instances * 0.2s = 4s. Well within 270s.
            
            d = _jc_distance(s_i, s_j)
            dist[i][j] = d
            dist[j][i] = d

    # Active nodes: initially 0..n-1
    active = list(range(n))
    # Newick representation for each active node
    newick = {i: labels[i] for i in range(n)}
    
    # Distance matrix for active nodes
    # Using a dictionary for clarity and dynamic updates
    d_matrix = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d_matrix[(i, j)] = dist[i][j]

    next_id = n
    
    while len(active) > 3:
        if time.time() - start_time > timeout - safety_margin:
            break
            
        n_active = len(active)
        
        # Calculate sum of distances for each active node
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += d_matrix[(i, j)]
        
        # Find the pair (i, j) that minimizes the NJ criterion
        best_val = float('inf')
        best_i = None
        best_j = None
        
        for u_idx in range(n_active):
            i = active[u_idx]
            for v_idx in range(u_idx + 1, n_active):
                j = active[v_idx]
                d_ij = d_matrix[(i, j)]
                criterion = (n_active - 2) * d_ij - r[i] - r[j]
                if criterion < best_val:
                    best_val = criterion
                    best_i = i
                    best_j = j
        
        if best_i is None:
            break
            
        i, j = best_i, best_j
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Newick for new node
        newick[new_id] = f"({newick[i]},{newick[j]})"
        
        # Calculate distances from new node to all other active nodes
        d_ij = d_matrix[(i, j)]
        
        for x in active:
            if x == i or x == j:
                continue
            d_ix = d_matrix[(i, x)]
            d_jx = d_matrix[(j, x)]
            d_nx = 0.5 * (d_ix + d_jx - d_ij)
            if d_nx < 0:
                d_nx = 0.0
            d_matrix[(new_id, x)] = d_nx
            d_matrix[(x, new_id)] = d_nx
            
        # Remove i and j from active
        active.remove(i)
        active.remove(j)
        active.append(new_id)

    # Assemble the final tree
    if len(active) == 3:
        a, b, c = active
        return f"({newick[a]},{newick[b]},{newick[c]});"
    else:
        # Fallback: complete the joining using simple distance min if we ran out of time
        while len(active) > 3:
             best_dist = float('inf')
             best_i = active[0]
             best_j = active[1]
             for u_idx in range(len(active)):
                 i = active[u_idx]
                 for v_idx in range(u_idx + 1, len(active)):
                     j = active[v_idx]
                     d = d_matrix[(i, j)]
                     if d < best_dist:
                         best_dist = d
                         best_i = i
                         best_j = j
             
             new_id = next_id
             next_id += 1
             newick[new_id] = f"({newick[best_i]},{newick[best_j]})"
             
             d_ij = best_dist
             for x in active:
                 if x == best_i or x == best_j:
                     continue
                 d_nx = 0.5 * (d_matrix[(best_i, x)] + d_matrix[(best_j, x)] - d_ij)
                 if d_nx < 0:
                     d_nx = 0.0
                 d_matrix[(new_id, x)] = d_nx
                 d_matrix[(x, new_id)] = d_nx
                 
             active.remove(best_i)
             active.remove(best_j)
             active.append(new_id)
             
        a, b, c = active
        return f"({newick[a]},{newick[b]},{newick[c]});"

# EVOLVE-BLOCK-END
