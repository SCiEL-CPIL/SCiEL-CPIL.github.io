import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Precompute conflicts
    # conflict_pairs[x] is a list of (y, z) such that {x, y, z} is a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N or c <= b:
                continue
            # 3-AP (a, b, c)
            conflict_pairs[a].append((b, c))
            conflict_pairs[b].append((a, c))
            conflict_pairs[c].append((a, b))
            
    def can_add(x, S_mask):
        for y, z in conflict_pairs[x]:
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True

    def get_sum(S_mask):
        total = 0.0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                total += recip[x]
        return total

    def greedy_fill(S_mask, forbidden_mask=0):
        S = S_mask
        for x in range(1, N + 1):
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S

    def greedy_fill_descending(S_mask, forbidden_mask=0):
        S = S_mask
        for x in range(N, 0, -1):
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S

    def local_search(S_mask, max_time):
        S = S_mask
        current_sum = get_sum(S)
        S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Strategy 1: 1-Swap
            best_gain = 0
            best_x = -1
            best_y = -1
            
            # Iterate y (to add) in ascending order (prefer small y)
            # Iterate x (to remove) in descending order (prefer large x)
            for y in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if S & (1 << y):
                    continue
                
                for x in range(N, 0, -1):
                    if (time.time() - start_time) > max_time:
                        break
                    if not (S & (1 << x)):
                        continue
                    
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        continue
                    
                    S_new = S & ~(1 << x)
                    if can_add(y, S_new):
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
            
            if best_y != -1:
                S = S & ~(1 << best_x)
                S |= (1 << best_y)
                current_sum += best_gain
                S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                improved = True
                continue
            
            # Strategy 2: 1-Remove and Re-greedy
            best_remove_gain = -1
            best_remove_S = None
            
            # Try removing elements from S. 
            # Prioritize removing large elements (low weight) that might block many small ones.
            # Or just iterate all.
            for x in S_elements[:]:
                if (time.time() - start_time) > max_time:
                    break
                
                S_no_x = S & ~(1 << x)
                # Greedy fill ascending from S_no_x
                S_new = greedy_fill(S_no_x, 1 << x)
                sum_new = get_sum(S_new)
                gain = sum_new - current_sum
                
                if gain > 1e-15 and gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_S = S_new
            
            if best_remove_S is not None:
                S = best_remove_S
                current_sum += best_remove_gain
                S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                improved = True
                continue
                
            # Strategy 3: 2-Remove and Re-greedy (Limited)
            # Try removing 2 elements. Only try pairs involving large elements to save time.
            # If |S| is large, this is expensive. Let's limit to top 10 largest elements.
            large_elements = sorted(S_elements, reverse=True)[:10]
            best_2_remove_gain = -1
            best_2_remove_S = None
            
            for i in range(len(large_elements)):
                if (time.time() - start_time) > max_time:
                    break
                x1 = large_elements[i]
                for j in range(i + 1, len(large_elements)):
                    x2 = large_elements[j]
                    S_no_2 = S & ~(1 << x1) & ~(1 << x2)
                    S_new = greedy_fill(S_no_2, (1 << x1) | (1 << x2))
                    sum_new = get_sum(S_new)
                    gain = sum_new - current_sum
                    if gain > 1e-15 and gain > best_2_remove_gain:
                        best_2_remove_gain = gain
                        best_2_remove_S = S_new
            
            if best_2_remove_S is not None:
                S = best_2_remove_S
                current_sum += best_2_remove_gain
                S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                improved = True
                continue

            # Strategy 4: Random Perturbation
            if (time.time() - start_time) > max_time:
                break
            if not S_elements:
                break
            
            # Remove 1 or 2 random elements and re-greedy
            k = random.choice([1, 2])
            removed = random.sample(S_elements, k)
            forbidden = 0
            for x in removed:
                forbidden |= (1 << x)
            S_no_k = S
            for x in removed:
                S_no_k &= ~(1 << x)
                
            S_new = greedy_fill(S_no_k, forbidden)
            sum_new = get_sum(S_new)
            
            # Accept if better
            if sum_new > current_sum + 1e-15:
                S = S_new
                current_sum = sum_new
                S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                improved = True
                continue
            
            break
            
        return S, current_sum

    # Step 1: Initializations
    initializations = []
    
    # Greedy ascending
    S_asc = greedy_fill(0)
    initializations.append(S_asc)
    
    # Greedy descending
    S_desc = greedy_fill_descending(0)
    initializations.append(S_desc)
    
    # Cantor Set Initialization (Base 3, no 1s)
    S_cantor = 0
    for x in range(1, N + 1):
        num = x
        valid = True
        while num > 0:
            if num % 3 == 1:
                valid = False
                break
            num //= 3
        if valid:
            S_cantor |= (1 << x)
    initializations.append(S_cantor)
    
    # Randomized Greedy
    for trial in range(20):
        if (time.time() - start_time) > deadline - 20:
            break
        random.seed(trial * 12345 + 67890)
        S_rand = 0
        for x in all_nums:
            # Skip small numbers with higher probability to create diversity?
            # Or just skip with fixed probability.
            if random.random() < 0.1:
                continue
            if can_add(x, S_rand):
                S_rand |= (1 << x)
        initializations.append(S_rand)
        
    # Step 2: Local Search on all initializations
    best_S = 0
    best_sum = 0.0
    
    # Allocate time evenly
    time_per_ls = max(2.0, (deadline - 15 - (time.time() - start_time)) / max(1, len(initializations)))
    
    for i, S_init in enumerate(initializations):
        if (time.time() - start_time) > deadline - 5:
            break
        
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 2:
            break
        
        S_ls, sum_ls = local_search(S_init, min(time_per_ls, remaining_time))
        
        if sum_ls > best_sum:
            best_sum = sum_ls
            best_S = S_ls
    
    # Step 3: Final Polish on the best result
    if best_S != 0:
        remaining = deadline - 2 - (time.time() - start_time)
        if remaining > 5:
            S_final, sum_final = local_search(best_S, remaining)
            if sum_final > best_sum:
                best_S = S_final
                best_sum = sum_final
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if best_S & (1 << x)]
    
    # Validation
    S_set = set(result)
    valid = True
    for a in S_set:
        for b in S_set:
            if a >= b:
                continue
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if mid in S_set and mid != a and mid != b:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid set (e.g., greedy ascending)
        result = [x for x in range(1, N + 1) if S_asc & (1 << x)]
        
    return result
