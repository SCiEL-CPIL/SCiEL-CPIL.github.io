import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    # Precompute all nonconstant 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 <= n:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a3 - 1))
    aps = np.array(aps, dtype=np.int32)
    
    def count_violations_fast(colors):
        c = colors[aps]  # (num_aps, 4)
        # An AP violates if it uses <= 2 distinct colors
        # Sort each row
        cs = np.sort(c, axis=1)
        # Count distinct: 1 - (cs[:,1]==cs[:,0]) - (cs[:,2]==cs[:,1]) - (cs[:,3]==cs[:,2])... no
        # Better: distinct = 4 - (cs[:,1]==cs[:,0]).sum() - ... 
        # Actually: number of distinct = 1 + sum of (cs[i,j+1] != cs[i,j])
        distinct = 1 + (cs[:, 1] != cs[:, 0]).astype(int) + \
                     (cs[:, 2] != cs[:, 1]).astype(int) + \
                     (cs[:, 3] != cs[:, 2]).astype(int)
        violations = (distinct <= 2).sum()
        return violations
    
    def is_valid(colors):
        return count_violations_fast(colors) == 0
    
    rng = random.Random(12345)
    
    # Try to find valid 3-coloring
    best_colors = None
    best_k = 128
    
    # Generate diverse initial patterns
    def gen_pattern(seed):
        r = random.Random(seed)
        pattern_type = r.randint(0, 5)
        if pattern_type == 0:
            # mod 3 with offset
            off = r.randint(0, 2)
            return np.array([(i + off) % 3 for i in range(n)], dtype=np.int32)
        elif pattern_type == 1:
            # mod 4 mapped to 3
            perm = [0, 1, 2, 0]
            r.shuffle(perm)
            return np.array([perm[i % 4] for i in range(n)], dtype=np.int32)
        elif pattern_type == 2:
            # random
            return np.array([r.randint(0, 2) for _ in range(n)], dtype=np.int32)
        elif pattern_type == 3:
            # Thue-Morse variant
            c = np.zeros(n, dtype=np.int32)
            for i in range(n):
                v = i
                c[i] = (v ^ (v >> 1) ^ (v >> 2) ^ (v >> 3)) % 3
            return c
        elif pattern_type == 4:
            # Block-based
            block = r.randint(3, 8)
            c = np.zeros(n, dtype=np.int32)
            for i in range(n):
                c[i] = (i // block) % 3
            return c
        else:
            # Hybrid
            c = np.array([(i % 3) ^ (i // 4 % 2) for i in range(n)], dtype=np.int32)
            c = c % 3
            return c
    
    def local_search(colors, k, max_iters=80000, timeout=75):
        colors = colors.copy().astype(np.int32)
        current_violations = count_violations_fast(colors)
        best_colors = colors.copy()
        best_violations = current_violations
        best_k_used = len(set(colors.tolist()))
        
        if current_violations == 0:
            return colors, best_k_used
        
        temp = 3.0
        cooling = 0.99995
        no_improve = 0
        max_no_improve = 8000
        
        for it in range(max_iters):
            if time.time() > timeout:
                break
            if no_improve > max_no_improve and current_violations > 0:
                break
            
            pos = rng.randint(0, n - 1)
            old_color = colors[pos]
            new_color = rng.randint(0, k - 1)
            if new_color == old_color:
                continue
            
            colors[pos] = new_color
            new_violations = count_violations_fast(colors)
            delta = new_violations - current_violations
            
            if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 0.01)):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
                    best_k_used = len(set(colors.tolist()))
                    no_improve = 0
                    if new_violations == 0:
                        return colors, best_k_used
                else:
                    no_improve += 1
            else:
                colors[pos] = old_color
                no_improve += 1
            
            temp *= cooling
        
        return best_colors, best_k_used
    
    # Try many initial patterns
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        init = gen_pattern(seed)
        result, k_used = local_search(init, 3, max_iters=40000, timeout=75)
        if k_used < best_k:
            best_k = k_used
            best_colors = result
        if best_k <= 3:
            break
    
    # If still not 3, try 4 colors
    if best_k > 3:
        for k in range(4, 7):
            if time.time() > deadline - 5:
                break
            for seed in range(20):
                if time.time() > deadline - 5:
                    break
                init = gen_pattern(seed + 1000)
                # Expand to k colors
                if k > 3:
                    init = np.where(init == 2, rng.randint(3, k - 1), init)
                result, k_used = local_search(init, k, max_iters=30000, timeout=75)
                if k_used < best_k:
                    best_k = k_used
                    best_colors = result
                if best_k <= k:
                    break
    
    # Final: ensure we have a valid coloring
    if best_colors is None:
        # Fallback: use 4 colors in a structured way
        best_colors = np.array([(i % 4) for i in range(n)], dtype=np.int32)
    
    # Try to minimize colors used in the best valid coloring
    if is_valid(best_colors) and time.time() < deadline - 3:
        colors_used = sorted(set(best_colors.tolist()))
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            colors_used = sorted(set(best_colors.tolist()))
            for i in range(len(colors_used)):
                for j in range(i + 1, len(colors_used)):
                    if time.time() > deadline - 2:
                        break
                    trial = best_colors.copy()
                    trial[trial == colors_used[j]] = colors_used[i]
                    if is_valid(trial):
                        best_colors = trial
                        changed = True
                        break
                if changed:
                    break
    
    return list(best_colors.astype(int))
