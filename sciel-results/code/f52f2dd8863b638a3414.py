import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_valid_fast(adj):
        """Fast validity check using numpy. Returns True if no K4 and no independent 6-set."""
        n = adj.shape[0]
        
        # Check K4: for each pair (i,j) with edge, check if they share 2+ common neighbors
        # Vectorized: for each edge (i,j), count common neighbors
        # adj is symmetric boolean matrix
        # For K4: need 4 vertices all mutually connected
        # For each pair i<j with adj[i,j], common = adj[i] & adj[j] & ~((1<<(j+1))-1) as mask
        # But let's use a different approach: count triangles, then check if any triangle has a 4th vertex connected to all 3
        
        # Simpler: iterate over edges, for each edge check common neighbors
        # This is O(n^3) but with numpy vectorization per edge it's fast enough for n<=40
        
        # K4 check: 
        # For each i<j<k with all three edges present, check if any l > k is connected to i, j, k
        # Alternative: for each triple (i,j,k) that forms a triangle, check if there's a 4th vertex
        # Even simpler: for each pair (i,j) that's an edge, let C = common neighbors of i and j with index > j
        # For each pair (k,l) in C with k<l, if adj[k,l] then K4 exists
        
        # Let's use bitmask approach but with Python integers (fast for n<=40)
        adj_bits = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            bits = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    bits |= (1 << j)
            adj_bits[i] = bits
        
        # K4 check
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]:
                    continue
                common = adj_bits[i] & adj_bits[j]
                # Only consider vertices > j
                common &= ~((1 << (j + 1)) - 1)
                # For each pair of common neighbors, check if they're connected
                # If any pair (k,l) in common with k<l has adj[k,l], we have K4
                # Iterate through common neighbors
                mask = common
                while mask:
                    lb = mask & (-mask)
                    k = lb.bit_length() - 1
                    mask ^= lb
                    # Check if k is connected to any other common neighbor > k
                    rest = common & ~((1 << (k + 1)) - 1)
                    if rest & adj_bits[k]:
                        return False
        
        # Independent set of 6 check: use complement
        comp = ~adj
        np.fill_diagonal(comp, False)
        
        comp_bits = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            bits = 0
            row = comp[i]
            for j in range(n):
                if row[j]:
                    bits |= (1 << j)
            comp_bits[i] = bits
        
        # Check for independent set of 6
        # For each 5-tuple that's an independent set, check if there's a 6th vertex
        # More efficiently: for each 4-tuple that's independent, check if 2 more vertices can be added
        # Or: for each pair (i,j) in comp, common = comp neighbors of i and j with index > j
        # For each triple (k,l,m) in common that's independent, check if 6th exists
        
        # Let's do: for each 5 independent vertices, check if 6th exists
        # But that's O(n^6). Better: for each 4 independent vertices, check if 2 more can extend
        # Even better: build up incrementally
        
        # For each pair (i,j) in comp:
        #   C = common comp-neighbors with index > j
        #   For each pair (k,l) in C that are connected in comp:
        #     D = common comp-neighbors of i,j,k,l with index > l
        #     For each pair (m,n) in D that are connected in comp:
        #       If m and n exist, we have independent 6-set
        
        for i in range(n):
            for j in range(i + 1, n):
                if not comp[i, j]:
                    continue
                common_ij = comp_bits[i] & comp_bits[j]
                common_ij &= ~((1 << (j + 1)) - 1)
                
                mask_k = common_ij
                while mask_k:
                    lb_k = mask_k & (-mask_k)
                    k = lb_k.bit_length() - 1
                    mask_k ^= lb_k
                    
                    common_ijk = comp_bits[i] & comp_bits[j] & comp_bits[k]
                    common_ijk &= ~((1 << (k + 1)) - 1)
                    
                    mask_l = common_ijk
                    while mask_l:
                        lb_l = mask_l & (-mask_l)
                        l = lb_l.bit_length() - 1
                        mask_l ^= lb_l
                        
                        # Check if k and l are connected in comp
                        if not comp[k, l]:
                            continue
                        
                        common_ijkl = comp_bits[i] & comp_bits[j] & comp_bits[k] & comp_bits[l]
                        common_ijkl &= ~((1 << (l + 1)) - 1)
                        
                        mask_m = common_ijkl
                        while mask_m:
                            lb_m = mask_m & (-mask_m)
                            m = lb_m.bit_length() - 1
                            mask_m ^= lb_m
                            
                            common_ijklm = comp_bits[i] & comp_bits[j] & comp_bits[k] & comp_bits[l] & comp_bits[m]
                            common_ijklm &= ~((1 << (m + 1)) - 1)
                            
                            if common_ijklm:
                                return False
        
        return True

    # First, try the known 35-vertex graph (complement of triangular graph T(7))
    vertices = list(combinations(range(7), 3))
    n35 = len(vertices)
    edges35 = []
    for i in range(n35):
        for j in range(i + 1, n35):
            u = vertices[i]
            v = vertices[j]
            inter_size = len(set(u).intersection(v))
            if inter_size != 1:
                edges35.append([i, j])
    
    adj35 = np.zeros((n35, n35), dtype=bool)
    for u, v in edges35:
        adj35[u, v] = True
        adj35[v, u] = True
    
    if check_valid_fast(adj35):
        # Try to extend to 36 vertices using local search
        if time.time() < deadline - 5:
            best_result = {'n': n35, 'edges': edges35}
            
            # Try to add a 36th vertex
            random.seed(42)
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                    
                # Start with 35-vertex graph, try to add vertex 35
                adj36 = np.zeros((36, 36), dtype=bool)
                adj36[:35, :35] = adj35
                
                # Randomly connect new vertex to some existing vertices
                # Need to avoid creating K4 and independent 6-sets
                # Strategy: connect to a random subset, then fix violations
                new_edges = set()
                for i in range(35):
                    if random.random() < 0.5:
                        new_edges.add(i)
                
                for i in new_edges:
                    adj36[35, i] = True
                    adj36[i, 35] = True
                
                # Try to fix violations with local search
                improved = True
                iterations = 0
                while improved and iterations < 100 and time.time() < deadline - 10:
                    improved = False
                    iterations += 1
                    
                    if check_valid_fast(adj36):
                        edges36 = []
                        for i in range(36):
                            for j in range(i + 1, 36):
                                if adj36[i, j]:
                                    edges36.append([i, j])
                        return {'n': 36, 'edges': edges36}
                    
                    # Find a violation and try to fix it
                    # Check for K4 involving vertex 35
                    # If there's a K4 with vertex 35, remove one edge
                    # If there's an independent 6-set with vertex 35, add one edge
                    
                    # Simple heuristic: randomly toggle an edge involving vertex 35
                    i = random.randint(0, 34)
                    if adj36[35, i]:
                        adj36[35, i] = False
                        adj36[i, 35] = False
                    else:
                        adj36[35, i] = True
                        adj36[i, 35] = True
                    improved = True
                
                # Also try modifying edges in the 35-vertex part
                for mod_attempt in range(20):
                    if time.time() > deadline - 10:
                        break
                    
                    adj_mod = adj35.copy()
                    # Randomly modify a few edges
                    for _ in range(5):
                        i = random.randint(0, 34)
                        j = random.randint(0, 34)
                        if i != j:
                            adj_mod[i, j] = not adj_mod[i, j]
                            adj_mod[j, i] = adj_mod[i, j]
                    
                    # Try to add vertex 35
                    adj_try = np.zeros((36, 36), dtype=bool)
                    adj_try[:35, :35] = adj_mod
                    for i in range(35):
                        if random.random() < 0.5:
                            adj_try[35, i] = True
                            adj_try[i, 35] = True
                    
                    if check_valid_fast(adj_try):
                        edges36 = []
                        for i in range(36):
                            for j in range(i + 1, 36):
                                if adj_try[i, j]:
                                    edges36.append([i, j])
                        return {'n': 36, 'edges': edges36}
            
            return best_result
        else:
            return {'n': n35, 'edges': edges35}
    
    # Fallback: try smaller graphs
    for n in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline - 5:
            break
        
        # Try random graph with local search
        random.seed(n)
        for attempt in range(20):
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            # Local search to fix violations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                if check_valid_fast(adj):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Randomly toggle an edge
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    adj[i, j] = not adj[i, j]
                    adj[j, i] = adj[i, j]
    
    return {'n': 1, 'edges': []}
