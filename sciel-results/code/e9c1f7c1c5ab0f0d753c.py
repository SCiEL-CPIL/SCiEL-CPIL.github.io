import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s kill

    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]

    # Precompute conflicts
    # For each x, conflict_pairs[x] contains (y, z) such that {x, y, z} forms a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N or c <= b:
                continue
            # a, b, c is a 3-AP
            conflict_pairs[a].append((b, c))
            conflict_pairs[b].append((a, c))
            conflict_pairs[c].append((a, b))

    def can_add(x, S_mask):
        """Check if x can be added to set S_mask."""
        for y, z in conflict_pairs[x]:
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True

    def get_float_sum(S_mask):
        """Compute floating-point reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                total += recip[x]
        return total

    def greedy_fill(forbidden_mask, S_mask, order=None):
        """Fill S by adding elements in a given order."""
        S = S_mask
        if order is None:
            order = range(1, N + 1)
        for x in order:
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S

    def local_search(S_mask, max_time):
        """Run local search starting from S_mask, return best mask found."""
        S = S_mask
        sum_float = get_float_sum(S)
        
        # Precompute list of elements in S for faster iteration
        S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Try 1-Swap: remove x, add y
            best_gain = 0
            best_x = -1
            best_y = -1
            
            # Optimization: only consider y < x for gain > 0
            for y in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if S & (1 << y):
                    continue
                
                for x in range(y + 1, N + 1):
                    if not (S & (1 << x)):
                        continue
                    
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        continue
                    
                    # Check validity
                    S_new = S & ~(1 << x)
                    if can_add(y, S_new):
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
            
            if best_y != -1:
                S = S & ~(1 << best_x)
                S |= (1 << best_y)
                sum_float += best_gain
                improved = True
                continue
            
            # Try 2-Swap: remove x1, x2, add y1, y2
            best_2swap_gain = 0
            best_2swap_x1 = -1
            best_2swap_x2 = -1
            best_2swap_y1 = -1
            best_2swap_y2 = -1
            
            # Get candidates
            # y: small numbers not in S (high recip)
            y_candidates = [y for y in range(1, N + 1) if not (S & (1 << y))]
            # x: large numbers in S (low recip)
            x_candidates = [x for x in range(1, N + 1) if S & (1 << x)]
            
            # Limit candidates to keep O(K^4) manageable
            # Top 10 y, Bottom 10 x
            y_top = y_candidates[:10]
            x_top = x_candidates[-10:]
            
            if len(y_top) >= 2 and len(x_top) >= 2:
                for i in range(len(y_top)):
                    for j in range(i + 1, len(y_top)):
                        y1, y2 = y_top[i], y_top[j]
                        
                        for k in range(len(x_top)):
                            for l in range(k + 1, len(x_top)):
                                x1, x2 = x_top[k], x_top[l]
                                
                                gain = recip[y1] + recip[y2] - recip[x1] - recip[x2]
                                if gain <= 1e-15:
                                    continue
                                
                                # Check validity
                                S_new = S
                                S_new &= ~(1 << x1)
                                S_new &= ~(1 << x2)
                                S_new |= (1 << y1)
                                S_new |= (1 << y2)
                                
                                # Check if y1, y2 can be added to S_new
                                # We check can_add for each new element against the set without itself
                                S_no_y1 = S_new & ~(1 << y1)
                                S_no_y2 = S_new & ~(1 << y2)
                                
                                if can_add(y1, S_no_y1) and can_add(y2, S_no_y2):
                                    if gain > best_2swap_gain:
                                        best_2swap_gain = gain
                                        best_2swap_x1, best_2swap_x2 = x1, x2
                                        best_2swap_y1, best_2swap_y2 = y1, y2
            
            if best_2swap_y1 != -1:
                S = S & ~(1 << best_2swap_x1)
                S = S & ~(1 << best_2swap_x2)
                S |= (1 << best_2swap_y1)
                S |= (1 << best_2swap_y2)
                sum_float += best_2swap_gain
                S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                improved = True
                continue

            # If no 1-swap or 2-swap, try 1-Remove + Greedy
            # Remove x, then greedily fill
            best_remove_gain = -1
            best_remove_x = -1
            best_remove_S = 0
            
            for x in S_elements:
                if (time.time() - start_time) > max_time:
                    break
                if not (S & (1 << x)):
                    continue
                
                S_no_x = S & ~(1 << x)
                S_new = greedy_fill(1 << x, S_no_x)
                sum_new = get_float_sum(S_new)
                gain = sum_new - sum_float
                
                if gain > 1e-15 and gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_x = x
                    best_remove_S = S_new
            
            if best_remove_x != -1:
                S = best_remove_S
                sum_float += best_remove_gain
                # Update S_elements
                S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                improved = True
                continue
            
            # Try adding a single element if possible
            added = False
            for y in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if S & (1 << y):
                    continue
                if can_add(y, S):
                    S |= (1 << y)
                    sum_float += recip[y]
                    S_elements.append(y)
                    improved = True
                    added = True
                    break
            
            if not added:
                # Random perturbation: remove a random element and re-greedy
                if not S_elements:
                    break
                x = random.choice(S_elements)
                S_no_x = S & ~(1 << x)
                S_new = greedy_fill(1 << x, S_no_x)
                sum_new = get_float_sum(S_new)
                if sum_new > sum_float:
                    S = S_new
                    sum_float = sum_new
                    S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                    improved = True
        
        return S, sum_float

    # Step 1: Initializations
    initializations = []
    
    # Greedy ascending
    S_asc = greedy_fill(0, 0, order=range(1, N + 1))
    initializations.append(S_asc)
    
    # Greedy descending
    S_desc = greedy_fill(0, 0, order=range(N, 0, -1))
    initializations.append(S_desc)
    
    # Greedy with random shuffles
    for trial in range(15):
        if (time.time() - start_time) > deadline - 20:
            break
        random.seed(trial * 12345 + 67890)
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = greedy_fill(0, 0, order=order)
        initializations.append(S_rand)
    
    # Step 2: Local Search
    best_S = 0
    best_sum = 0.0
    
    time_per_ls = max(5.0, (deadline - 10 - (time.time() - start_time)) / max(1, len(initializations)))
    
    for i, S_init in enumerate(initializations):
        if (time.time() - start_time) > deadline - 5:
            break
        
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 3:
            break
        
        S_ls, sum_ls = local_search(S_init, min(time_per_ls, remaining_time))
        
        if sum_ls > best_sum:
            best_sum = sum_ls
            best_S = S_ls
    
    # Step 3: Final Polish
    if best_S != 0:
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 5:
            S_final, sum_final = local_search(best_S, remaining)
            if sum_final > best_sum:
                best_S = S_final
                best_sum = sum_final
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if best_S & (1 << x)]
    
    # Validate
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
        # Fallback to a known valid set if validation fails (should not happen)
        # Use greedy result
        result = [x for x in range(1, N + 1) if S_asc & (1 << x)]
        
    return result
