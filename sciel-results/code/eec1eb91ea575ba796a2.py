import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    
    # Precompute all non-constant 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with d > 0
    aps = []
    for d in range(1, N):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                aps.append((a, b, c, e))
                
    # For fast cost update, we need to know which APs contain a given index
    # index_to_aps[i] = list of indices in 'aps' list that contain i
    index_to_aps = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            index_to_aps[pos].append(idx)

    def count_distinct(colors, ap):
        c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c1, c2, c3, c4})

    def get_cost(colors):
        cost = 0
        for ap in aps:
            if count_distinct(colors, ap) < 3:
                cost += 1
        return cost

    def get_bad_aps(colors):
        bad = []
        for idx, ap in enumerate(aps):
            if count_distinct(colors, ap) < 3:
                bad.append(idx)
        return bad

    # Try to find a valid coloring with K colors
    for K in range(3, 11):
        # Time limit per K
        k_start = time.time()
        k_deadline = min(deadline, k_start + 5.0)
        
        found = False
        max_restarts = 100000 # Will be limited by time
        
        for r in range(max_restarts):
            if time.time() > k_deadline:
                break
                
            # Initialize random coloring
            colors = [0] * (N + 1)
            for i in range(1, N + 1):
                colors[i] = random.randint(0, K - 1)
                
            # Calculate initial cost
            current_cost = get_cost(colors)
            
            if current_cost == 0:
                # Extract used colors to ensure we minimize label count if K > used
                used = set(colors[1:N+1])
                # The problem says "Minimize the number of labels actually used"
                # If we found a solution with K colors, but only used 2 of them? 
                # No, if it's valid, it must use at least 3. 
                # So if K=3 and valid, it uses 3.
                return [colors[i] for i in range(1, N + 1)]
            
            # Local Search / Hill Climbing
            # We use a simple greedy repair: pick a random bad AP, try to fix it by changing one of its terms
            # Or better: pick a random position, try all K colors, pick best delta.
            
            stuck_count = 0
            while time.time() < k_deadline:
                if current_cost == 0:
                    return [colors[i] for i in range(1, N + 1)]
                
                # Pick a random position to change
                pos = random.randint(1, N)
                old_color = colors[pos]
                
                best_new_color = old_color
                best_cost = current_cost
                
                # Try all K colors
                for c in range(K):
                    if c == old_color:
                        continue
                    
                    # Calculate delta cost
                    # Only APs containing 'pos' are affected
                    delta = 0
                    for ap_idx in index_to_aps[pos]:
                        ap = aps[ap_idx]
                        
                        # Count distinct before
                        old_d = count_distinct(colors, ap)
                        
                        # Temporarily change
                        colors[pos] = c
                        new_d = count_distinct(colors, ap)
                        
                        # Restore
                        colors[pos] = old_color
                        
                        if old_d < 3 and new_d >= 3:
                            delta -= 1
                        elif old_d >= 3 and new_d < 3:
                            delta += 1
                        # If both < 3 or both >= 3, delta is 0 for this AP w.r.t validity
                        # Note: We only care about crossing the threshold of 3.
                        # The cost is defined as number of APs with < 3 colors.
                        
                    if delta < best_cost - current_cost:
                        best_cost = current_cost + delta
                        best_new_color = c
                
                # Apply best change
                if best_new_color != old_color:
                    colors[pos] = best_new_color
                    current_cost = best_cost
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 100:
                        break # Restart search for this K

        if found:
            break

    # Fallback: Use a pattern that is guaranteed valid with many colors
    # x % 128 ensures all distinct, so valid.
    # But we want to minimize colors. 
    # A pattern like (i // 10) % 5 ? 
    # If we just return the last tried solution even if invalid? No, invalid scores 0.
    # We must return a valid solution.
    # The fallback `i % 128` is valid.
    return [i % 128 for i in range(1, N + 1)]
