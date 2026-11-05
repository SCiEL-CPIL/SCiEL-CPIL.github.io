import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 255
    # 255 = 3 * 5 * 17
    def shares_factor(x):
        return math.gcd(x, 255) > 1
    
    candidates = [x for x in range(1, n + 1) if shares_factor(x)]
    m = len(candidates)
    cand_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = cand_idx[255]
    
    # Build adjacency as bitmasks
    adj = [0] * m
    for i in range(m):
        vi = candidates[i]
        for j in range(i + 1, m):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Baseline: all multiples of 3 (size 85)
    multiples_3 = [x for x in range(3, 256, 3)]
    best_size = len(multiples_3)
    best_set = list(multiples_3)
    
    # Greedy clique: start with 255, add vertex with max degree in available subgraph
    def greedy_clique(adj, m, start, deadline):
        clique_mask = 1 << start
        available = adj[start] & ~(1 << start)
        
        while available and time.time() < deadline:
            # Find vertex in available with max degree within available
            best_v = -1
            best_deg = -1
            mask = available
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                deg = (adj[v] & available).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                mask ^= lb
            
            if best_v == -1:
                break
            clique_mask |= (1 << best_v)
            available &= adj[best_v]
            available &= ~(1 << best_v)
        
        return clique_mask
    
    greedy_mask = greedy_clique(adj, m, idx_255, deadline)
    greedy_set = [candidates[i] for i in range(m) if (greedy_mask >> i) & 1]
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set
    
    # Local search with bitmask operations for speed
    if time.time() < deadline - 5:
        # Start from the better of greedy or multiples-of-3
        if len(greedy_set) >= best_size:
            cur_mask = greedy_mask
        else:
            cur_mask = 0
            for x in best_set:
                cur_mask |= (1 << cand_idx[x])
        
        improved = True
        max_iters = 50 # Reduced to ensure speed, logic is fast
        for _ in range(max_iters):
            if not improved or time.time() > deadline - 5:
                break
            improved = False
            
            # For each vertex u in clique, try swapping with v not in clique
            clique_list = [i for i in range(m) if (cur_mask >> i) & 1]
            
            for u in clique_list:
                if time.time() > deadline - 5:
                    break
                # clique without u
                clique_wo_u = cur_mask & ~(1 << u)
                
                # Find best v not in cur_mask that connects to all in clique_wo_u
                best_v = -1
                best_score = -1
                not_in_clique = ((1 << m) - 1) & ~cur_mask
                mask = not_in_clique
                while mask:
                    lb = mask & (-mask)
                    v = lb.bit_length() - 1
                    if (adj[v] & clique_wo_u) == clique_wo_u:
                        # Score: prefer v that connects to many in original clique
                        score = (adj[v] & cur_mask).bit_count()
                        if score > best_score:
                            best_score = score
                            best_v = v
                    mask ^= lb
                
                if best_v != -1:
                    cur_mask = (cur_mask & ~(1 << u)) | (1 << best_v)
                    improved = True
        
        cur_set = [candidates[i] for i in range(m) if (cur_mask >> i) & 1]
        if len(cur_set) > best_size:
            best_size = len(cur_set)
            best_set = cur_set
    
    # Ensure 255 is in the result
    if 255 not in best_set:
        best_set.append(255)
    
    return best_set
