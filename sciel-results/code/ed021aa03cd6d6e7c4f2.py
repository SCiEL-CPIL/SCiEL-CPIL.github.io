import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    N = 256
    
    # Identify bad nodes where 2*i is a perfect square
    bad_nodes = set()
    for i in range(1, N + 1):
        s = 2 * i
        r = int(math.isqrt(s))
        if r * r == s:
            bad_nodes.add(i)
            
    # Remaining nodes
    valid_nodes = [i for i in range(1, N + 1) if i not in bad_nodes]
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    n = len(valid_nodes)
    
    if n == 0:
        return [1] # Fallback, though 1 is not bad (2*1=2 not square)
        
    # Build adjacency bitsets
    adj = [0] * n
    
    # Precompute squares
    squares = set()
    k = 2
    while k*k <= 2*N:
        squares.add(k*k)
        k += 1
        
    for i in range(n):
        u = valid_nodes[i]
        mask = 0
        for j in range(i + 1, n):
            v = valid_nodes[j]
            if u + v in squares:
                mask |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = mask
        
    best_size = 0
    best_set = []
    
    def greedy_color_bound(candidates):
        if not candidates:
            return 0
            
        # Heuristic: if candidates is small, just return popcount
        if candidates.bit_count() < 10:
            return candidates.bit_count()
            
        # Get nodes
        nodes = []
        c = candidates
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            c ^= lsb
            
        # Sort by degree in subgraph (descending)
        nodes.sort(key=lambda i: (adj[i] & candidates).bit_count(), reverse=True)
        
        # color[i] stores the color assigned to node i in this subgraph coloring
        color = [0] * n 
        
        max_color = 0
        for node in nodes:
            used_colors = set()
            neighbors_mask = adj[node] & candidates
            nb_c = neighbors_mask
            while nb_c:
                lsb = nb_c & -nb_c
                nb_idx = lsb.bit_length() - 1
                if color[nb_idx] > 0:
                    used_colors.add(color[nb_idx])
                nb_c ^= lsb
                
            # Assign smallest available color
            c_val = 1
            while c_val in used_colors:
                c_val += 1
                
            color[node] = c_val
            if c_val > max_color:
                max_color = c_val
                
        return max_color

    def dfs(candidates, current_set, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Upper bound
        bound = greedy_color_bound(candidates)
        if current_size + bound <= best_size:
            return
            
        # Select vertex to branch on
        # Heuristic: pick vertex with max degree in subgraph
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = idx
            c ^= lsb
            
        if best_v == -1:
            return
            
        v_idx = best_v
        v = valid_nodes[v_idx]
        
        # Branch 1: Include v
        new_candidates_include = candidates & ~adj[v_idx] & ~(1 << v_idx)
        current_set.append(v)
        dfs(new_candidates_include, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: Exclude v
        new_candidates_exclude = candidates & ~(1 << v_idx)
        dfs(new_candidates_exclude, current_set, current_size)

    init_candidates = (1 << n) - 1
    dfs(init_candidates, [], 0)
    
    if not best_set:
        # Fallback: return any valid node
        return [valid_nodes[0]]
        
    return best_set

