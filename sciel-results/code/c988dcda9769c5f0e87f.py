import time
import random
import math

def construct():
    deadline = time.time() + 55  # Safe margin
    N = 256
    
    # Precomputation
    # We use 0-indexed internally for bitmasks, so element i is bit i.
    # Elements are 0..255 corresponding to 1..256.
    
    three_aps = []
    four_aps = []
    
    for d in range(1, N // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))
            
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # For each element, store the bitmask of the OTHER elements in each 3-AP and 4-AP it belongs to.
    # This allows O(1) check for "are the other elements present?" using (S & mask) == mask
    others_3ap = [[] for _ in range(N)]
    others_4ap = [[] for _ in range(N)]
    
    for a, b, c in three_aps:
        # For a, others are b, c
        others_3ap[a].append((1 << b) | (1 << c))
        # For b, others are a, c
        others_3ap[b].append((1 << a) | (1 << c))
        # For c, others are a, b
        others_3ap[c].append((1 << a) | (1 << b))
        
    for a, b, c, d in four_aps:
        # For a, others are b, c, d
        others_4ap[a].append((1 << b) | (1 << c) | (1 << d))
        # For b, others are a, c, d
        others_4ap[b].append((1 << a) | (1 << c) | (1 << d))
        # For c, others are a, b, d
        others_4ap[c].append((1 << a) | (1 << b) | (1 << d))
        # For d, others are a, b, c
        others_4ap[d].append((1 << a) | (1 << b) | (1 << c))
        
    def count_3aps(S):
        """Count 3-APs in bitset S."""
        count = 0
        for a, b, c in three_aps:
            # Check if a, b, c are in S
            if (S & (1 << a)) and (S & (1 << b)) and (S & (1 << c)):
                count += 1
        return count

    def is_valid_add(S, x):
        """Check if adding x to S creates a 4-AP."""
        for mask in others_4ap[x]:
            if (S & mask) == mask:
                return False
        return True

    def is_valid_swap(S, x_out, x_in):
        """Check if S - {x_out} + {x_in} is valid."""
        # Temporarily remove x_out
        S_temp = S & ~(1 << x_out)
        return is_valid_add(S_temp, x_in)

    def delta_add(S, x):
        """Change in 3-AP count if we add x to S."""
        delta = 0
        for mask in others_3ap[x]:
            if (S & mask) == mask:
                delta += 1
        return delta

    def delta_remove(S, x):
        """Change in 3-AP count if we remove x from S. (Negative of delta_add if x not in S, but logic is same for neighbors)"""
        # The APs lost are those where the other 2 are in S.
        delta = 0
        for mask in others_3ap[x]:
            if (S & mask) == mask:
                delta += 1
        return -delta

    # Greedy Initialization
    # Sort elements by number of 3-APs they participate in, descending
    # This prioritizes elements that create more 3-APs
    elem_popularity = [len(others_3ap[i]) for i in range(N)]
    sorted_elems = sorted(range(N), key=lambda x: elem_popularity[x], reverse=True)
    
    S = 0
    for x in sorted_elems:
        if is_valid_add(S, x):
            S |= (1 << x)
            
    best_S = S
    best_score = count_3aps(best_S)
    current_score = best_score
    
    # Local Search (Best Improvement)
    def local_search(S, current_score, max_iters=100):
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            best_delta = 0
            best_x_out = -1
            best_x_in = -1
            
            # Try all swaps
            # Optimization: Only try x_in that are "good"
            for x_out in range(N):
                if not (S & (1 << x_out)):
                    continue
                S_without_out = S & ~(1 << x_out)
                loss = -delta_remove(S, x_out) # delta_remove returns negative, so loss is positive count of APs lost
                # Actually delta_remove(S, x) returns -count. So loss = -(-count) = count.
                # Let's just compute delta for the swap directly.
                
                for x_in in range(N):
                    if S & (1 << x_in):
                        continue
                    if not is_valid_add(S_without_out, x_in):
                        continue
                    
                    gain = delta_add(S_without_out, x_in)
                    delta = gain + delta_remove(S, x_out) # delta_remove is negative
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_x_out = x_out
                        best_x_in = x_in
            
            if best_delta > 0:
                S &= ~(1 << best_x_out)
                S |= (1 << best_x_in)
                current_score += best_delta
                improved = True
                
        return S, current_score

    S, current_score = local_search(S, current_score, max_iters=5)
    if current_score > best_score:
        best_score = current_score
        best_S = S

    # Simulated Annealing
    random.seed(42)
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    # Precompute list of elements for faster random choice
    all_elems = list(range(N))
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        move_type = random.random()
        accepted = False
        
        if move_type < 0.4: # Add
            # Pick random element not in S
            x = random.choice(all_elems)
            if not (S & (1 << x)):
                if is_valid_add(S, x):
                    delta = delta_add(S, x)
                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)) or (delta == 0 and random.random() < 0.5):
                        S |= (1 << x)
                        current_score += delta
                        accepted = True
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S
                            
        elif move_type < 0.6: # Remove
            if S:
                # Pick random element in S
                # Find set bits
                # Simple way: random choice from all, check if in S
                x = random.choice(all_elems)
                if S & (1 << x):
                    delta = delta_remove(S, x)
                    if random.random() < math.exp(delta / T):
                        S &= ~(1 << x)
                        current_score += delta
                        accepted = True
        else: # Swap
            if S:
                x_out = random.choice(all_elems)
                if S & (1 << x_out):
                    x_in = random.choice(all_elems)
                    if not (S & (1 << x_in)):
                        if is_valid_swap(S, x_out, x_in):
                            S_without_out = S & ~(1 << x_out)
                            gain = delta_add(S_without_out, x_in)
                            loss = -delta_remove(S, x_out)
                            delta = gain - loss
                            if delta > 0 or random.random() < math.exp(delta / T):
                                S &= ~(1 << x_out)
                                S |= (1 << x_in)
                                current_score += delta
                                accepted = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_S = S
                                
        T *= cooling
        
        # Periodically run local search to polish
        if accepted and random.random() < 0.05:
            S, current_score = local_search(S, current_score, max_iters=2)
            if current_score > best_score:
                best_score = current_score
                best_S = S

    # Convert best_S (0-indexed bitset) to list of 1-indexed integers
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
