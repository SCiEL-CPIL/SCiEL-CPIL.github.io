import time
import random
import math

def construct():
    deadline = time.time() + 145
    
    # Fallback
    best_n = 11
    # Simple valid coloring for n=11 (cyclic)
    # Actually, just use the search for n=11 as fallback if it fails?
    # Or hardcode a known small one?
    # The search for n=11 is very fast.
    
    targets = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    best_colors = []
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        if n == 26:
            t_limit = 100
        elif n == 25:
            t_limit = 30
        elif n <= 23:
            t_limit = 15
        else:
            t_limit = 10
            
        actual_limit = min(deadline - time.time() - 5, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break
            
    # If we didn't find anything (shouldn't happen for n=11), return n=1
    if not best_colors and best_n == 11:
        # Force find n=11
        success, colors = _find_coloring(11, time.time() + 5)
        if success:
            best_n = 11
            best_colors = colors
        else:
            best_n = 1
            best_colors = []

    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline):
    if n < 3:
        return True, []
        
    m = n * (n - 1) // 2
    
    # Precompute edge endpoints
    edge_u = [0] * m
    edge_v = [0] * m
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # adj_mask[c][v] = bitmask of neighbors of v in color c
    adj_mask = [[0] * n for _ in range(5)]
    
    def build_masks(colors):
        for c in range(5):
            for v in range(n):
                adj_mask[c][v] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
            
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    cnt = common.bit_count()
                    if cnt >= 2:
                        total += cnt - 1
        return total
        
    def find_violating_pairs():
        violations = []
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    cnt = common.bit_count()
                    if cnt >= 2:
                        violations.append((u, v, c, cnt))
        return violations
        
    def compute_delta(k, old_c, new_c):
        u = edge_u[k]
        v = edge_v[k]
        delta = 0
        
        # Removing (u,v) from old_c
        mask_w = adj_mask[old_c][v]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == u:
                continue
            cnt = (adj_mask[old_c][u] & adj_mask[old_c][w]).bit_count()
            if cnt >= 2:
                delta -= 1
        
        # Adding (u,v) to new_c
        mask_w = adj_mask[new_c][v]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == u:
                continue
            cnt = (adj_mask[new_c][u] & adj_mask[new_c][w]).bit_count()
            if cnt >= 1:
                delta += 1
        
        # Removing (v,u) from old_c
        mask_w = adj_mask[old_c][u]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == v:
                continue
            cnt = (adj_mask[old_c][v] & adj_mask[old_c][w]).bit_count()
            if cnt >= 2:
                delta -= 1
        
        # Adding (v,u) to new_c
        mask_w = adj_mask[new_c][u]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == v:
                continue
            cnt = (adj_mask[new_c][v] & adj_mask[new_c][w]).bit_count()
            if cnt >= 1:
                delta += 1
                
        return delta
        
    def apply_change(k, old_c, new_c, colors):
        u = edge_u[k]
        v = edge_v[k]
        adj_mask[old_c][u] &= ~(1 << v)
        adj_mask[old_c][v] &= ~(1 << u)
        adj_mask[new_c][u] |= (1 << v)
        adj_mask[new_c][v] |= (1 << u)
        colors[k] = new_c
        
    def targeted_fix(colors, current_violations, deadline):
        violations = find_violating_pairs()
        if not violations:
            return True, current_violations
            
        random.shuffle(violations)
        
        for (u, v, c, cnt) in violations:
            if time.time() > deadline - 1:
                break
            if current_violations == 0:
                return True, 0
                
            common = adj_mask[c][u] & adj_mask[c][v]
            common_list = []
            mask = common
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                mask -= lsb
                common_list.append(w)
                
            if len(common_list) < 2:
                continue
                
            w1, w2 = common_list[0], common_list[1]
            
            candidates = []
            for a, b in [(u, w1), (u, w2), (v, w1), (v, w2), (w1, w2)]:
                if a < b:
                    candidates.append((a, b))
                elif b < a:
                    candidates.append((b, a))
                    
            random.shuffle(candidates)
            
            for (a, b) in candidates:
                k = pair_to_idx.get((a, b), pair_to_idx.get((b, a), -1))
                if k < 0:
                    continue
                    
                old_c = colors[k]
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    delta = compute_delta(k, old_c, new_c)
                    if delta < 0:
                        apply_change(k, old_c, new_c, colors)
                        current_violations += delta
                        if current_violations == 0:
                            return True, 0
                        break
                        
        return False, current_violations

    def greedy_init():
        colors = [0] * m
        # Initialize adj_mask to 0
        for c in range(5):
            for v in range(n):
                adj_mask[c][v] = 0
        
        # Process edges in random order? Or sorted?
        # Random order is better to break symmetry
        edges_order = list(range(m))
        random.shuffle(edges_order)
        
        for k in edges_order:
            u = edge_u[k]
            v = edge_v[k]
            best_c = 0
            best_cost = float('inf')
            
            for c in range(5):
                # Tentatively add edge (u,v) with color c
                # Check impact on violations
                # This is expensive if done naively.
                # Approximation: just pick random?
                # Or: Check if adding this edge creates a C4.
                # A C4 is created if u and v have >= 2 common neighbors in color c.
                common = adj_mask[c][u] & adj_mask[c][v]
                cnt = common.bit_count()
                cost = max(0, cnt - 1)
                
                if cost < best_cost:
                    best_cost = cost
                    best_c = c
                    if cost == 0:
                        break
                elif cost == best_cost:
                    # Random tie break
                    if random.random() < 0.5:
                        best_c = c
                        
            colors[k] = best_c
            adj_mask[best_c][u] |= (1 << v)
            adj_mask[best_c][v] |= (1 << u)
            
        return colors

    # Multi-restart
    best_violations = float('inf')
    best_colors = None
    
    num_restarts = 20
    restart_budget = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 1:
            break
            
        # Greedy initial coloring
        colors = greedy_init()
        current_violations = count_violations()
        
        if current_violations == 0:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 1: Targeted fixing
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 2: Simulated Annealing
        T = 1.0
        min_T = 0.01
        phase_deadline = min(deadline - 1, time.time() + restart_budget * 0.7)
        stagnant_steps = 0
        
        while time.time() < phase_deadline:
            k = random.randint(0, m - 1)
            u = edge_u[k]
            v = edge_v[k]
            old_c = colors[k]
            new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
                
            delta = compute_delta(k, old_c, new_c)
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                apply_change(k, old_c, new_c, colors)
                current_violations += delta
                
                if current_violations == 0:
                    return True, colors
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1
                    
                if stagnant_steps > 5000:
                    T = min(2.0, T * 1.5)
                    stagnant_steps = 0
            else:
                stagnant_steps += 1
                
            T *= 0.999
            if T < min_T:
                T = min_T
                
        # Phase 3: Final targeted fix
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
    if best_violations == 0 and best_colors is not None:
        return True, best_colors
    return False, best_colors if best_colors else []
