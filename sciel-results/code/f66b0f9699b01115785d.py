import time
import random
import math

def construct():
    N = 256
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Precompute 4-AP conflicts
    # conflicts[x] is a list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    conflicts = [[] for _ in range(N)]
    
    # Generate all 4-APs
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: (a, b, c, e)
            # For a: conflicts are (b, c, e)
            conflicts[a].append((b, c, e))
            # For b: conflicts are (a, c, e)
            conflicts[b].append((a, c, e))
            # For c: conflicts are (a, b, e)
            conflicts[c].append((a, b, e))
            # For e: conflicts are (a, b, c)
            conflicts[e].append((a, b, c))

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True

    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        # Iterate over all possible (start, diff)
        # A 3-AP is (i, i+d, i+2d)
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            max_d = (N - 1 - i) // 2
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                if ((mask >> j) & 1) and ((mask >> k) & 1):
                    count += 1
        return count

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        # Case 1: x is the start: x, x+d, x+2d
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # Case 2: x is the middle: x-d, x, x+d
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # Case 3: x is the end: x-2d, x-d, x
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        # This is exactly the same as delta_3aps_add for the current mask
        # because removing x breaks exactly the APs that adding it would create
        return delta_3aps_add(mask, x)

    # 2. Improved Greedy Initialization
    def greedy_max_3ap():
        mask = 0
        available = set(range(N))
        
        while available:
            best_x = None
            best_delta = -1
            candidates = list(available)
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_add(mask, x):
                    d = delta_3aps_add(mask, x)
                    if d > best_delta:
                        best_delta = d
                        best_x = x
            
            if best_x is not None and best_delta > 0:
                mask |= (1 << best_x)
                available.remove(best_x)
            else:
                # If no positive delta, stop adding to avoid blocking future gains
                # or pick a random valid one? 
                # Let's stop to keep the set "tight" and let SA expand if needed.
                break
                
        return mask

    def run_sa(initial_mask, time_limit):
        current_mask = initial_mask
        current_score = count_3aps(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        T = 2.0
        T_min = 0.01
        cooling = 0.9995
        steps_since_improve = 0
        
        while time.time() < time_limit:
            move_type = random.random()
            new_mask = current_mask
            new_score = current_score
            valid_move = False
            
            if move_type < 0.4:
                # Add
                x = random.randint(0, N - 1)
                if not ((current_mask >> x) & 1):
                    if is_valid_add(current_mask, x):
                        new_mask = current_mask | (1 << x)
                        new_score = current_score + delta_3aps_add(current_mask, x)
                        valid_move = True
            elif move_type < 0.7:
                # Remove
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    new_mask = current_mask & ~(1 << x)
                    new_score = current_score - delta_3aps_remove(current_mask, x)
                    valid_move = True
            else:
                # Swap
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    y = random.randint(0, N - 1)
                    if not ((current_mask >> y) & 1):
                        temp_mask = current_mask & ~(1 << x)
                        if is_valid_add(temp_mask, y):
                            d_rem = delta_3aps_remove(current_mask, x)
                            d_add = delta_3aps_add(temp_mask, y)
                            new_score = current_score - d_rem + d_add
                            new_mask = temp_mask | (1 << y)
                            valid_move = True

            if valid_move:
                if new_score >= current_score:
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        steps_since_improve = 0
                else:
                    delta = current_score - new_score
                    if delta < 1e-9:
                        current_mask = new_mask
                        current_score = new_score
                    else:
                        accept_prob = math.exp(-delta / T)
                        if random.random() < accept_prob:
                            current_mask = new_mask
                            current_score = new_score
                            steps_since_improve += 1
                        else:
                            steps_since_improve += 1
                T *= cooling
                if T < T_min:
                    T = T_min
            else:
                T *= cooling
                if T < T_min:
                    T = T_min
                
            # Reheating logic
            if steps_since_improve > 2000:
                T = 2.0
                steps_since_improve = 0

        return best_mask, best_score

    # Main execution
    best_mask_global = 0
    best_score_global = -1
    
    # Run 1: Greedy Max 3AP + SA
    init_mask = greedy_max_3ap()
    m, s = run_sa(init_mask, deadline - 5.0)
    if s > best_score_global:
        best_score_global = s
        best_mask_global = m
        
    # Run 2: If time remains, try another random greedy + SA
    if time.time() < deadline - 10.0:
        init_mask2 = greedy_max_3ap()
        m2, s2 = run_sa(init_mask2, deadline - 2.0)
        if s2 > best_score_global:
            best_score_global = s2
            best_mask_global = m2

    result = [i + 1 for i in range(N) if (best_mask_global >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
