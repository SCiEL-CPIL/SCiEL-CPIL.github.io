import time
import random
import math

def construct():
    n = 128
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    pos_to_aps = [[] for _ in range(n)]
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            idx = len(aps)
            aps.append(ap)
            for pos in ap:
                pos_to_aps[pos].append(idx)
    
    def is_ap_valid(colors, ap):
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        
        # Need at least 3 distinct colors
        # Invalid patterns: all same, 3 same, 2-2 (specific pairings)
        if c0 == c1 == c2 == c3:
            return False
        if c0 == c1 and c0 == c2: return False
        if c0 == c1 and c1 == c3: return False
        if c0 == c2 and c2 == c3: return False
        if c1 == c2 and c2 == c3: return False
        if c0 == c1 and c2 == c3 and c0 != c2:
            return False
        if c0 == c2 and c1 == c3 and c0 != c1:
            return False
        if c0 == c3 and c1 == c2 and c0 != c1:
            return False
            
        return True

    def count_violations(colors):
        v = 0
        for ap in aps:
            if not is_ap_valid(colors, ap):
                v += 1
        return v

    def greedy_construct(K, rng, max_retries=100):
        """Try to construct a valid K-coloring greedily."""
        for attempt in range(max_retries):
            colors = [-1] * n
            order = list(range(n))
            rng.shuffle(order)
            
            success = True
            for pos in order:
                # Find which color to assign
                # A color c is bad if any AP containing pos would violate with c
                bad = [False] * K
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    others = [p for p in ap if p != pos]
                    # Check if all others are assigned
                    all_assigned = True
                    other_colors = []
                    for p in others:
                        if colors[p] == -1:
                            all_assigned = False
                            break
                        other_colors.append(colors[p])
                    if not all_assigned:
                        continue
                    # For each candidate color, check if it causes violation
                    for c in range(K):
                        # Temporarily set and check
                        test_colors = colors[:]
                        test_colors[pos] = c
                        if not is_ap_valid(test_colors, ap):
                            bad[c] = True
                
                available = [c for c in range(K) if not bad[c]]
                if not available:
                    success = False
                    break
                colors[pos] = rng.choice(available)
            
            if success and count_violations(colors) == 0:
                return colors
        
        return None

    def solve_with_k(K, time_limit, initial_colors=None, rng=None):
        if rng is None:
            rng = random.Random()
        
        if initial_colors:
            colors = initial_colors[:]
        else:
            colors = [rng.randint(0, K - 1) for _ in range(n)]
            
        curr_violations = count_violations(colors)
        best_colors = colors[:]
        best_violations = curr_violations
        
        if best_violations == 0:
            return best_colors

        temp = 5.0
        cooling = 0.995
        min_temp = 0.01
        
        while time.time() < time_limit and temp > min_temp:
            pos = rng.randint(0, n - 1)
            old_color = colors[pos]
            new_color = rng.randint(0, K - 1)
            if new_color == old_color:
                continue
            
            # Calculate delta
            affected = pos_to_aps[pos]
            
            curr_invalid = 0
            for ap_idx in affected:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    curr_invalid += 1
            
            colors[pos] = new_color
            
            new_invalid = 0
            for ap_idx in affected:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    new_invalid += 1
            
            delta_v = new_invalid - curr_invalid
            
            if delta_v < 0:
                accept = True
            elif delta_v == 0:
                accept = True
            else:
                prob = math.exp(-delta_v / temp)
                accept = rng.random() < prob
            
            if accept:
                curr_violations += delta_v
                if curr_violations < best_violations:
                    best_violations = curr_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
            else:
                colors[pos] = old_color
                
            temp *= cooling
        
        return best_colors

    # Strategy: Try greedy first for small K, then SA as backup
    # K=3 is the target minimum
    
    best_result = None
    best_k = 128
    
    for K in [3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 10.0:
            break
        
        # Try greedy construction with multiple random seeds
        for seed in range(20):
            if time.time() > deadline - 10.0:
                break
            rng = random.Random(seed * 1000 + K)
            res = greedy_construct(K, rng, max_retries=50)
            if res is not None:
                if K < best_k:
                    best_k = K
                    best_result = res
                    return res
        
        # If greedy failed, try SA
        if time.time() > deadline - 15.0:
            break
        
        sa_limit = min(deadline - 15.0, time.time() + 10.0)
        rng = random.Random(K * 777 + 42)
        res = solve_with_k(K, sa_limit, rng=rng)
        if count_violations(res) == 0:
            if K < best_k:
                best_k = K
                best_result = res
                return res
    
    # Final fallback: greedy with larger K
    if best_result is None:
        for K in [12, 16, 20, 30]:
            if time.time() > deadline - 5.0:
                break
            rng = random.Random(K * 333)
            res = greedy_construct(K, rng, max_retries=20)
            if res is not None:
                if K < best_k:
                    best_k = K
                    best_result = res
                    return res
    
    # Last resort: return something valid (identity uses 128 colors)
    if best_result is not None:
        return best_result
    
    return list(range(n))
