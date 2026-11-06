import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Fallback: n=11 construction
    n_fb = 11
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    best_n = n_fb
    best_colors = colors_fb
    
    # Try to find valid colorings for larger n
    # Prioritize 27 (new result), then 26 (known Lazebnik-Woldar), then others
    targets = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 3:
            break
            
        # Allocate time budget
        if n == 27:
            t_limit = 40
        elif n == 26:
            t_limit = 40
        elif n == 25:
            t_limit = 20
        elif n <= 23:
            t_limit = 15
        else:
            t_limit = 10
            
        actual_limit = min(deadline - time.time() - 3, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break  # Found the largest n that works
            
    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline):
    """
    Find a valid 5-coloring of K_n with no monochromatic C4 using
    constraint-guided local search with targeted violation fixing.
    """
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
        """Count total C4 violations using bitmask intersections."""
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        total += cnt - 1
        return total
        
    def find_violating_pairs():
        """Find all (u,v,c) triples where u,v have >=2 common neighbors in color c."""
        violations = []
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        violations.append((u, v, c, cnt))
        return violations
        
    def compute_delta(k, old_c, new_c):
        """Compute exact delta in violation count when changing edge k from old_c to new_c."""
        u = edge_u[k]
        v = edge_v[k]
        delta = 0
        
        # Removing (u,v) from old_c: for each w in adj_mask[old_c][v] (w != u),
        # common neighbors of (u,w) in old_c decrease by 1
        mask_w = adj_mask[old_c][v]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == u:
                continue
            cnt = bin(adj_mask[old_c][u] & adj_mask[old_c][w]).count('1')
            if cnt >= 2:
                delta -= 1
        
        # Adding (u,v) to new_c: for each w in adj_mask[new_c][v] (w != u),
        # common neighbors of (u,w) in new_c increase by 1
        mask_w = adj_mask[new_c][v]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == u:
                continue
            cnt = bin(adj_mask[new_c][u] & adj_mask[new_c][w]).count('1')
            if cnt >= 1:
                delta += 1
        
        # Removing (v,u) from old_c: for each w in adj_mask[old_c][u] (w != v),
        # common neighbors of (v,w) in old_c decrease by 1
        mask_w = adj_mask[old_c][u]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == v:
                continue
            cnt = bin(adj_mask[old_c][v] & adj_mask[old_c][w]).count('1')
            if cnt >= 2:
                delta -= 1
        
        # Adding (v,u) to new_c: for each w in adj_mask[new_c][u] (w != v),
        # common neighbors of (v,w) in new_c increase by 1
        mask_w = adj_mask[new_c][u]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == v:
                continue
            cnt = bin(adj_mask[new_c][v] & adj_mask[new_c][w]).count('1')
            if cnt >= 1:
                delta += 1
                
        return delta
        
    def apply_change(k, old_c, new_c, colors):
        """Apply a color change to edge k."""
        u = edge_u[k]
        v = edge_v[k]
        adj_mask[old_c][u] &= ~(1 << v)
        adj_mask[old_c][v] &= ~(1 << u)
        adj_mask[new_c][u] |= (1 << v)
        adj_mask[new_c][v] |= (1 << u)
        colors[k] = new_c
        
    def targeted_fix(colors, current_violations, deadline):
        """Try to fix violations by targeting specific violating edges."""
        violations = find_violating_pairs()
        if not violations:
            return True, current_violations
            
        # For each violating pair (u,v,c), try changing one of the edges between
        # common neighbors to break the C4
        random.shuffle(violations)
        
        for (u, v, c, cnt) in violations:
            if time.time() > deadline - 1:
                break
            if current_violations == 0:
                return True, 0
                
            # Get common neighbors of u and v in color c
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
                
            # Pick two common neighbors w1, w2 that form a C4: u-w1-v-w2-u
            # Try changing edge (u,w1) or (v,w2) or (w1,w2) or (u,w2) or (v,w1)
            w1, w2 = common_list[0], common_list[1]
            
            # Candidate edges to change
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
                # Try each new color
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
        
    # Multi-restart with constraint-guided local search
    best_violations = float('inf')
    best_colors = None
    
    num_restarts = 15
    restart_budget = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 1:
            break
            
        # Random initial coloring
        colors = [random.randint(0, 4) for _ in range(m)]
        build_masks(colors)
        
        current_violations = count_violations()
        
        if current_violations == 0:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 1: Targeted violation fixing
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 2: Random local search with reheating
        T = 2.0
        min_T = 0.05
        phase_deadline = min(deadline - 1, time.time() + restart_budget * 0.6)
        stagnant_steps = 0
        max_stagnant = 20000
        
        while time.time() < phase_deadline and stagnant_steps < max_stagnant:
            k = random.randint(0, m - 1)
            u = edge_u[k]
            v = edge_v[k]
            old_c = colors[k]
            new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
                
            delta = compute_delta(k, old_c, new_c)
            
            # Acceptance: always accept improvements, sometimes accept worse
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
                    
                # Reheat if stuck
                if stagnant_steps > 10000:
                    T = min(2.0, T * 2)
                    stagnant_steps = 0
            else:
                stagnant_steps += 1
                
            T *= 0.99995
            if T < min_T:
                T = min_T
                
        # Phase 3: More targeted fixing after random search
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
    if best_violations == 0 and best_colors is not None:
        return True, best_colors
    return False, best_colors if best_colors else []
