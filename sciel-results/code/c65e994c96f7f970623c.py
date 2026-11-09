import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def try_find_coloring(n):
        # Precompute 6-sets
        sets_6 = []
        set_to_idx = {}
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            for p in range(m+1, n):
                                s = (i, j, k, l, m, p)
                                set_to_idx[s] = len(sets_6)
                                sets_6.append(s)
                                
        num_sets = len(sets_6)
        if num_sets == 0:
            return None
            
        # Edge index function for 6 vertices
        def edge_idx(u, v):
            if u > v: u, v = v, u
            if u == 0: return v - 1
            elif u == 1: return 4 + (v - 2)
            elif u == 2: return 8 + (v - 3)
            elif u == 3: return 11 + (v - 4)
            else: return 14
            
        # Generate perfect matching masks
        perfect_matchings = set()
        for p in itertools.permutations([0, 1, 2, 3, 4, 5]):
            mask = 0
            mask |= 1 << edge_idx(p[0], p[1])
            mask |= 1 << edge_idx(p[2], p[3])
            mask |= 1 << edge_idx(p[4], p[5])
            perfect_matchings.add(mask)
            
        # Precompute edge_to_sets
        edge_to_sets = {}
        for s_idx, s in enumerate(sets_6):
            for u_idx in range(6):
                for v_idx in range(u_idx + 1, 6):
                    u = s[u_idx]
                    v = s[v_idx]
                    if u > v: u, v = v, u
                    key = (u, v)
                    if key not in edge_to_sets:
                        edge_to_sets[key] = []
                    edge_to_sets[key].append(s_idx)
                    
        # Pair index
        def pair_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
            
        # Initialize
        adj = [0] * n
        colors = [0] * (n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 1)
                idx = pair_idx(i, j)
                colors[idx] = c
                if c == 0:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
                    
        masks = [0] * num_sets
        violations = 0
        
        def compute_mask(s):
            mask = 0
            for u_idx in range(6):
                u = s[u_idx]
                for v_idx in range(u_idx + 1, 6):
                    v = s[v_idx]
                    if adj[u] & (1 << v):
                        mask |= 1 << edge_idx(u_idx, v_idx)
            return mask
            
        for idx, s in enumerate(sets_6):
            m = compute_mask(s)
            masks[idx] = m
            if m in perfect_matchings:
                violations += 1
            if ((1 << 15) - 1 - m) in perfect_matchings:
                violations += 1
                
        best_violations = violations
        best_colors = colors[:]
        
        # Local search
        while time.time() < deadline - 5 and violations > 0:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            idx = pair_idx(i, j)
            old_color = colors[idx]
            new_color = 1 - old_color
            colors[idx] = new_color
            
            if new_color == 0:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                
            sets_to_check = edge_to_sets.get((i, j), [])
            delta_violations = 0
            
            for s_idx in sets_to_check:
                old_mask = masks[s_idx]
                old_viol = (old_mask in perfect_matchings) or (((1 << 15) - 1 - old_mask) in perfect_matchings)
                
                s = sets_6[s_idx]
                # Find local indices
                # Optimization: precompute local indices for each edge in each set?
                # For now, use index
                local_i = s.index(i)
                local_j = s.index(j)
                if local_i > local_j:
                    local_i, local_j = local_j, local_i
                e_idx = edge_idx(local_i, local_j)
                
                if new_color == 0:
                    new_mask = old_mask | (1 << e_idx)
                else:
                    new_mask = old_mask & ~(1 << e_idx)
                    
                masks[s_idx] = new_mask
                new_viol = (new_mask in perfect_matchings) or (((1 << 15) - 1 - new_mask) in perfect_matchings)
                
                if old_viol and not new_viol:
                    delta_violations -= 1
                elif not old_viol and new_viol:
                    delta_violations += 1
                    
            violations += delta_violations
            
            if violations > best_violations:
                # Revert
                colors[idx] = old_color
                if old_color == 0:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
                    
                for s_idx in sets_to_check:
                    s = sets_6[s_idx]
                    local_i = s.index(i)
                    local_j = s.index(j)
                    if local_i > local_j:
                        local_i, local_j = local_j, local_i
                    e_idx = edge_idx(local_i, local_j)
                    if old_color == 0:
                        masks[s_idx] |= (1 << e_idx)
                    else:
                        masks[s_idx] &= ~(1 << e_idx)
                        
                violations -= delta_violations
            else:
                best_violations = violations
                best_colors = colors[:]
                
        if best_violations == 0:
            return best_colors
        return None

    # Try n=29
    res = try_find_coloring(29)
    if res:
        return {'n': 29, 'colors': res}
        
    # Fallback to n=17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
