import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    # Target n=47 first, then 46
    # The problem asks to maximize n.
    # R(C4, K1,39) > n implies existence of C4-free graph on n vertices with min degree >= n-39.
    # Known: R(C4, K1,39) is 46 or 47.
    # If we find n=47, score is high. If n=46, score is slightly lower but still good.
    # Let's try to find n=47.
    
    best_result = None
    
    for target_n in [47, 46]:
        min_deg_needed = target_n - 39
        
        # Simulated Annealing
        # Initial state: Random graph with density p ~ 0.5?
        # Min degree needed is 8 for n=47.
        # Avg degree 0.5 * 46 = 23. Very dense. Many C4s.
        # Maybe start with p=0.3? Avg deg 13.8.
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            random.seed(restart * 12345 + target_n)
            
            # Initialize
            adj_mask = [0] * target_n
            deg = [0] * target_n
            
            p = 0.3
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
            
            # Calculate initial penalty
            c4_count = 0
            def count_c4s():
                total = 0
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if (adj_mask[u] >> v) & 1:
                            common = adj_mask[u] & adj_mask[v]
                            k = bin(common).count('1')
                            total += k * (k - 1) // 2
                return total
            
            # This is O(n^2) which is ~2000 ops. Fast enough.
            
            def get_penalty():
                c4s = count_c4s()
                deg_pen = 0
                for i in range(target_n):
                    if deg[i] < min_deg_needed:
                        diff = min_deg_needed - deg[i]
                        deg_pen += diff * diff
                return c4s * 10 + deg_pen # Weight C4s heavily? Or just sum?
                # Actually, we need BOTH to be 0.
                # Let's use lexicographic or just sum.
                # If we sum, we might accept a state with 1 C4 and 0 deg deficit over 0 C4 and 1 deg deficit?
                # Better to weight C4s higher if they are "hard" constraints?
                # Or just minimize c4s + deg_deficit.
                # Let's try c4s + 10 * deg_deficit.
                pass
            
            # Simpler penalty: c4s + sum(max(0, min_deg - deg[i]))
            # We want to reach 0.
            
            def calc_penalty():
                c4s = 0
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if (adj_mask[u] >> v) & 1:
                            common = adj_mask[u] & adj_mask[v]
                            k = bin(common).count('1')
                            if k >= 2:
                                c4s += k * (k - 1) // 2
                deg_def = 0
                for i in range(target_n):
                    if deg[i] < min_deg_needed:
                        deg_def += (min_deg_needed - deg[i])
                return c4s + deg_def
            
            current_penalty = calc_penalty()
            best_penalty = current_penalty
            best_adj = [m for m in adj_mask]
            best_deg = [d for d in deg]
            
            temp = 10.0
            min_temp = 0.01
            cooling = 0.995
            
            # Run SA
            max_steps = 2000
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Propose move: Add or Remove a random edge
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                is_edge = (adj_mask[u] >> v) & 1
                
                delta_penalty = 0
                
                if is_edge:
                    # Remove edge (u,v)
                    # Destroyed C4s: C(k, 2) where k = |N(u) \cap N(v)|
                    common = adj_mask[u] & adj_mask[v]
                    k = bin(common).count('1')
                    destroyed_c4s = k * (k - 1) // 2
                    
                    # Degree change
                    # deg[u] decreases by 1, deg[v] decreases by 1
                    deg_def_change = 0
                    if deg[u] == min_deg_needed:
                        deg_def_change += 1
                    if deg[v] == min_deg_needed:
                        deg_def_change += 1
                    
                    delta_penalty = -destroyed_c4s + deg_def_change
                    
                    # Apply
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    deg[u] -= 1
                    deg[v] -= 1
                    
                else:
                    # Add edge (u,v)
                    # Created C4s:
                    # Paths u-w-x-v.
                    # w in N(u), x in N(v), w!=x, (w,x) in E.
                    created_c4s = 0
                    mask_u = adj_mask[u]
                    m = mask_u
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        # Count neighbors of w in N(v)
                        common_w_v = adj_mask[w] & adj_mask[v]
                        count_w = bin(common_w_v).count('1')
                        # If w is in N(v), subtract 1 (w==x case)
                        if (adj_mask[v] >> w) & 1:
                            count_w -= 1
                        created_c4s += count_w
                        m ^= lsb
                    
                    # Degree change
                    deg_def_change = 0
                    if deg[u] < min_deg_needed:
                        deg_def_change -= 1
                    if deg[v] < min_deg_needed:
                        deg_def_change -= 1
                    
                    delta_penalty = created_c4s + deg_def_change
                    
                    # Apply
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                
                new_penalty = current_penalty + delta_penalty
                
                if new_penalty < current_penalty or random.random() < math.exp(-delta_penalty / temp):
                    current_penalty = new_penalty
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_adj = [m for m in adj_mask]
                        best_deg = [d for d in deg]
                else:
                    # Revert
                    if is_edge:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                    else:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
                
                temp *= cooling
                
                if best_penalty == 0:
                    break
            
            if best_penalty == 0:
                # Verify and return
                # Check min degree
                if min(best_deg) >= min_deg_needed:
                    # Check C4 free
                    is_c4_free = True
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if (best_adj[u] >> v) & 1:
                                common = best_adj[u] & best_adj[v]
                                if common & (common - 1): # > 1 bit
                                    is_c4_free = False
                                    break
                        if not is_c4_free:
                            break
                    if is_c4_free:
                        final_edges = []
                        for u in range(target_n):
                            for v in range(u + 1, target_n):
                                if (best_adj[u] >> v) & 1:
                                    final_edges.append([u, v])
                        return {'n': target_n, 'edges': final_edges}
    
    # Fallback
    return {'n': 39, 'edges': []}
