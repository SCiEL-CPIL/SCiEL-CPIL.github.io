import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 128
    # Precompute APs
    APs = []
    pos_to_aps = [[] for _ in range(n)]
    idx = 0
    for d in range(1, 43):
        for a in range(1, 129 - 3*d):
            p0, p1, p2, p3 = a-1, a+d-1, a+2*d-1, a+3*d-1
            APs.append((p0, p1, p2, p3))
            pos_to_aps[p0].append(idx)
            pos_to_aps[p1].append(idx)
            pos_to_aps[p2].append(idx)
            pos_to_aps[p3].append(idx)
            idx += 1
            
    num_aps = len(APs)
    
    def count_bad_aps(colors):
        bad = 0
        for ap in APs:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                bad += 1
        return bad

    def try_find_valid(k, max_restarts=50):
        best_global_bad = float('inf')
        best_global_colors = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            colors = [random.randint(0, k-1) for _ in range(n)]
            current_bad = count_bad_aps(colors)
            
            if current_bad == 0:
                return colors
                
            T = 1.0
            T_min = 0.01
            cooling = 0.995
            steps = 0
            max_steps = 20000
            
            local_best_bad = current_bad
            local_best_colors = colors[:]
            
            while steps < max_steps and time.time() - start_time < time_limit - 5.0:
                steps += 1
                
                # Pick random position
                pos = random.randint(0, n-1)
                old_c = colors[pos]
                
                # Pick new color
                new_c = random.randint(0, k-1)
                if new_c == old_c:
                    continue
                    
                colors[pos] = new_c
                
                # Calculate delta in bad count
                delta_bad = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = APs[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    is_bad_now = len({c0, c1, c2, c3}) < 3
                    
                    # Restore old color to check if it was bad
                    colors[pos] = old_c
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    is_bad_before = len({c0, c1, c2, c3}) < 3
                    # Restore new color
                    colors[pos] = new_c
                    
                    if is_bad_now and not is_bad_before:
                        delta_bad += 1
                    elif not is_bad_now and is_bad_before:
                        delta_bad -= 1
                        
                new_bad = current_bad + delta_bad
                
                # Acceptance criterion (Simulated Annealing)
                if new_bad < current_bad or random.random() < np.exp(-(new_bad - current_bad) / T):
                    current_bad = new_bad
                    if current_bad < local_best_bad:
                        local_best_bad = current_bad
                        local_best_colors = colors[:]
                        if current_bad == 0:
                            return colors
                else:
                    # Reject move
                    colors[pos] = old_c
                    
                T *= cooling
                if T < T_min:
                    T = T_min
                    
            if local_best_bad < best_global_bad:
                best_global_bad = local_best_bad
                best_global_colors = local_best_colors[:]
                
        return best_global_colors

    # Try to find a valid coloring with minimal number of colors
    # Start with 3 colors. If we find a valid one, it's optimal (min 3).
    # If not, try 4, 5, etc.
    # However, SA might not find a valid one if it doesn't exist or is hard.
    # We must return a VALID coloring.
    # If we fail to find a valid one with K colors, we cannot return an invalid one.
    # So we must increase K until we find a valid one or run out of time.
    # If we run out of time, we have to return something.
    # The task says "invalid result... scores 0".
    # So we MUST return a valid coloring.
    
    # Heuristic: A random coloring with many colors is likely valid.
    # With K=128, probability of any AP having <3 colors is very low.
    # So we can always fall back to a high-K coloring.
    
    for k in range(3, 10):
        res = try_find_valid(k)
        if res is not None and count_bad_aps(res) == 0:
            return res
            
    # If we couldn't find a valid one with <=9 colors, use a fallback
    # Generate a random coloring with 128 colors (or just distinct enough)
    # Actually, just assigning random 0..127 is very likely valid.
    colors = [random.randint(0, 127) for _ in range(n)]
    # Verify and repair if necessary (unlikely to be needed with 128 colors)
    # But to be safe, let's do a quick check.
    # If invalid, we can try to fix specific violations by changing colors to unique ones.
    
    # Simple repair: If an AP is bad, change one of its positions to a new unique color (if available) or a random one.
    # With 128 colors and 128 positions, we can just assign unique colors to everyone?
    # If every position has a unique color, every AP has 4 distinct colors. Valid.
    # Number of labels used = 128.
    # This is a valid fallback. Score = 128.
    # It's better than 0.
    
    # Can we do better than 128?
    # We can try to merge colors in the unique assignment if it doesn't create violations.
    # But for now, let's just return the unique assignment if SA failed.
    # Wait, the loop above tries up to k=9.
    # If it fails, we return unique.
    
    # Let's add one more layer: If SA fails for k=3..9, try a greedy construction for k=10..20?
    # No, let's stick to the SA result if valid, else unique.
    
    # Actually, we can try to optimize the unique coloring to use fewer colors.
    # Start with unique, then try to merge colors using SA.
    # This is effectively the same as starting with high K and cooling.
    
    # Let's just return the unique coloring as a guaranteed valid fallback.
    # But we can try one more thing: The SA for k=3..9 might have found a *near* valid coloring.
    # We can't use it if it's invalid.
    
    # So:
    # 1. Try SA for k=3 to 8.
    # 2. If any returns valid, return it.
    # 3. Else, return a valid coloring with 128 colors (unique).
    
    # Optimization: The loop above returns `best_global_colors` which might be invalid if `count_bad_aps` wasn't 0.
    # My `try_find_valid` returns `best_global_colors` even if bad > 0.
    # I need to check validity before returning.
    
    # Let's restructure:
    best_valid = None
    best_k = 128
    
    for k in range(3, 10):
        res = try_find_valid(k)
        if res is not None:
            if count_bad_aps(res) == 0:
                return res # Found optimal or near-optimal valid coloring
            # If res is invalid, we ignore it for now, but maybe store it?
            # No, we need a valid one.
            
    # Fallback: Unique coloring
    return list(range(n))
