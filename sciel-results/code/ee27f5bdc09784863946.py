import time
import random

def construct():
    """
    Constructs a graph with no K5 and no I5, maximizing the number of vertices.
    Returns (n, edges).
    """
    start_time = time.time()
    DEADLINE = 280  # Safety margin before 300s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency as list of bitmasks (Python ints)
    # adj[i] is a bitmask where bit j is 1 if (i, j) is an edge
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Helper to get current edges list
    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only count edges where i < j
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                mask ^= lsb
        return edges

    # Precompute 4-cliques and 4-independent sets
    def find_4_cliques():
        cliques = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)): continue
                for k in range(j+1, n):
                    if not (adj[i] & (1 << k)): continue
                    if not (adj[j] & (1 << k)): continue
                    for l in range(k+1, n):
                        if not (adj[i] & (1 << l)): continue
                        if not (adj[j] & (1 << l)): continue
                        if not (adj[k] & (1 << l)): continue
                        # Found a K4: {i, j, k, l}
                        mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                        cliques.append(mask)
        return cliques

    def find_4_independent_sets():
        indep_sets = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j): continue
                for k in range(j+1, n):
                    if adj[i] & (1 << k): continue
                    if adj[j] & (1 << k): continue
                    for l in range(k+1, n):
                        if adj[i] & (1 << l): continue
                        if adj[j] & (1 << l): continue
                        if adj[k] & (1 << l): continue
                        # Found an I4: {i, j, k, l}
                        mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                        indep_sets.append(mask)
        return indep_sets

    k4_list = find_4_cliques()
    i4_list = find_4_independent_sets()
    
    # Index K4s and I4s by vertex for fast incremental updates
    def build_indices(k4s, i4s, current_n):
        k4_idx = [[] for _ in range(current_n)]
        i4_idx = [[] for _ in range(current_n)]
        for idx, mask in enumerate(k4s):
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                k4_idx[v].append(idx)
                m ^= lsb
        for idx, mask in enumerate(i4s):
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                i4_idx[v].append(idx)
                m ^= lsb
        return k4_idx, i4_idx

    k4_index, i4_index = build_indices(k4_list, i4_list, n)
    
    best_n = n
    best_edges = get_edges()
    
    # Local Search function to find a valid mask for a new vertex
    def find_valid_mask(current_n, k4s, i4s, k4_idx, i4_idx):
        max_iters = 8000
        best_cost = float('inf')
        best_mask = 0
        
        for restart in range(30): # 30 restarts
            # Initialize mask
            # Heuristic: Start with a mask that has density 0.5
            mask = 0
            for i in range(current_n):
                if random.random() < 0.5:
                    mask |= (1 << i)
            
            # Calculate initial cost
            cost = 0
            full_mask = (1 << current_n) - 1
            
            # Count violations
            for k4 in k4s:
                if (k4 & mask) == k4:
                    cost += 1
            for i4 in i4s:
                if (i4 & mask) == 0:
                    cost += 1
            
            if cost == 0:
                return mask
            
            current_cost = cost
            current_mask = mask
            
            for _ in range(max_iters):
                if time.time() - start_time > DEADLINE:
                    return None # Timeout
                
                # Pick a random vertex to flip
                v = random.randint(0, current_n - 1)
                bit = 1 << v
                
                # Calculate delta cost
                delta = 0
                old_v_bit = 1 if (current_mask & bit) else 0
                
                # Check K4s containing v
                for idx in k4_idx[v]:
                    k4 = k4s[idx]
                    others = k4 ^ bit
                    if (others & current_mask) == others:
                        if old_v_bit == 0:
                            delta += 1
                        else:
                            delta -= 1
                
                # Check I4s containing v
                for idx in i4_idx[v]:
                    i4 = i4s[idx]
                    others = i4 ^ bit
                    if (others & current_mask) == 0:
                        if old_v_bit == 1:
                            delta += 1
                        else:
                            delta -= 1
                
                new_cost = current_cost + delta
                
                # Accept if better, or if equal (to allow moving on plateaus)
                # Use simulated annealing-like acceptance for equal cost to escape local minima
                if new_cost < current_cost:
                    current_mask ^= bit
                    current_cost = new_cost
                    if current_cost == 0:
                        return current_mask
                elif new_cost == current_cost:
                    # Accept equal cost moves with high probability to explore
                    if random.random() < 0.8:
                        current_mask ^= bit
                else:
                    # Reject worse moves, but occasionally accept to escape deep local minima
                    # Temperature-like heuristic: accept worse if cost difference is small
                    if new_cost - current_cost <= 1 and random.random() < 0.1:
                        current_mask ^= bit
                        current_cost = new_cost
            
            # End of iterations for this restart
            if current_cost < best_cost:
                best_cost = current_cost
                best_mask = current_mask
        
        # If we didn't find 0 cost, return None
        return None

    # Extension loop
    target_n = 45
    max_restarts_per_vertex = 20
    
    while n < target_n and time.time() - start_time < DEADLINE:
        added = False
        
        # Try to find a valid mask for vertex n
        for attempt in range(max_restarts_per_vertex):
            if time.time() - start_time > DEADLINE:
                break
            
            mask = find_valid_mask(n, k4_list, i4_list, k4_index, i4_index)
            
            if mask is not None:
                # Accept new vertex
                adj.append(mask)
                # Update adjacency for existing vertices
                for i in range(n):
                    if mask & (1 << i):
                        adj[i] |= (1 << n)
                
                # Update lists of K4 and I4
                # New K4s must include the new vertex n.
                neigh_list = []
                for i in range(n):
                    if mask & (1 << i):
                        neigh_list.append(i)
                        
                new_k4s = []
                for a_idx in range(len(neigh_list)):
                    a = neigh_list[a_idx]
                    for b_idx in range(a_idx+1, len(neigh_list)):
                        b = neigh_list[b_idx]
                        if not (adj[a] & (1 << b)): continue
                        for c_idx in range(b_idx+1, len(neigh_list)):
                            c = neigh_list[c_idx]
                            if not (adj[a] & (1 << c)): continue
                            if not (adj[b] & (1 << c)): continue
                            new_k4s.append((1 << a) | (1 << b) | (1 << c) | (1 << n))
                
                # New I4s formed by n + I3 in non-neighborhood of n.
                non_neigh_list = []
                for i in range(n):
                    if not (mask & (1 << i)):
                        non_neigh_list.append(i)
                        
                new_i4s = []
                for a_idx in range(len(non_neigh_list)):
                    a = non_neigh_list[a_idx]
                    for b_idx in range(a_idx+1, len(non_neigh_list)):
                        b = non_neigh_list[b_idx]
                        if adj[a] & (1 << b): continue
                        for c_idx in range(b_idx+1, len(non_neigh_list)):
                            c = non_neigh_list[c_idx]
                            if adj[a] & (1 << c): continue
                            if adj[b] & (1 << c): continue
                            new_i4s.append((1 << a) | (1 << b) | (1 << c) | (1 << n))
                            
                k4_list.extend(new_k4s)
                i4_list.extend(new_i4s)
                
                # Rebuild indices for the new vertex and update existing indices
                k4_index.append([])
                i4_index.append([])
                
                # Add new K4s to the lists.
                start_idx = len(k4_list) - len(new_k4s)
                for i, mask_k4 in enumerate(new_k4s):
                    idx = start_idx + i
                    m = mask_k4
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        k4_index[v].append(idx)
                        m ^= lsb
                        
                # Add new I4s to the lists.
                start_idx = len(i4_list) - len(new_i4s)
                for i, mask_i4 in enumerate(new_i4s):
                    idx = start_idx + i
                    m = mask_i4
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        i4_index[v].append(idx)
                        m ^= lsb
                        
                n += 1
                added = True
                best_n = n
                best_edges = get_edges()
                break # Move to next vertex
                
        if not added:
            # Could not add vertex n
            # Try a few more times with different random seeds before giving up?
            # For now, break.
            break

    return best_n, best_edges
