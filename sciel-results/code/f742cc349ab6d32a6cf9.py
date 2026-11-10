import time
import random
import math

N = 256
DEADLINE = 85.0  # Leave 5s safety margin for 90s limit

def _precompute():
    """
    Precomputes 3-AP and 4-AP partner lists.
    For each element i, stores tuples of other elements that form APs with it.
    """
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    # For a 3-AP (a, b, c), b is the middle.
    # We want to count distinct 3-APs.
    # A 3-AP is uniquely identified by its middle term and difference, or by its start and difference.
    # The current code counts by iterating elements and checking pairs.
    # Let's stick to the partner list approach but ensure we don't double count.
    # The current _count_3aps divides by 3. Why 3?
    # If we have a 3-AP (a, b, c):
    # a's partners include (b, c)
    # b's partners include (a, c)
    # c's partners include (a, b)
    # So each 3-AP is counted 3 times (once for each element).
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    # For a 4-AP (a, b, c, e):
    # a's partners include (b, c, e)
    # b's partners include (a, c, e)
    # c's partners include (a, b, e)
    # e's partners include (a, b, c)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

def _count_3aps(mask, ap3_partners):
    """
    Counts distinct 3-APs in the set represented by mask.
    mask is an integer where bit i-1 is 1 if i is in the set.
    """
    total = 0
    # Iterate over set bits
    m = mask
    while m:
        # Get lowest set bit index
        # i is 1-based in our data structures, bit index is 0-based
        lsb = m & (-m)
        i = lsb.bit_length()  # 1-based index
        m ^= lsb
        
        for j, k in ap3_partners[i]:
            # Check if j and k are in mask
            if (mask & (1 << (j - 1))) and (mask & (1 << (k - 1))):
                total += 1
    return total // 3

def _is_valid_addition(y, mask, ap4_partners):
    """
    Checks if adding y to the set represented by mask creates a 4-AP.
    """
    bit_y = 1 << (y - 1)
    for j, k, l in ap4_partners[y]:
        # If j, k, l are all in mask, then adding y creates a 4-AP
        if (mask & (1 << (j - 1))) and (mask & (1 << (k - 1))) and (mask & (1 << (l - 1))):
            return False
    return True

def _is_valid_swap(x, y, mask, ap4_partners):
    """
    Checks if swapping x (out) and y (in) maintains 4-AP free property.
    x is in mask, y is not.
    """
    mask_no_x = mask & ~(1 << (x - 1))
    # Check if y can be added to mask_no_x
    for j, k, l in ap4_partners[y]:
        if (mask_no_x & (1 << (j - 1))) and (mask_no_x & (1 << (k - 1))) and (mask_no_x & (1 << (l - 1))):
            return False
    return True

def _calculate_gain(y, mask, ap3_partners, exclude_x=None):
    """
    Calculates the increase in 3-AP count if y is added to the set.
    If exclude_x is provided, x is considered not in the set (for swaps).
    """
    count = 0
    for j, k in ap3_partners[y]:
        j_ok = (mask & (1 << (j - 1))) and (j != exclude_x)
        k_ok = (mask & (1 << (k - 1))) and (k != exclude_x)
        if j_ok and k_ok:
            count += 1
    return count

def _calculate_loss(x, mask, ap3_partners):
    """
    Calculates the decrease in 3-AP count if x is removed from the set.
    """
    count = 0
    for j, k in ap3_partners[x]:
        if (mask & (1 << (j - 1))) and (mask & (1 << (k - 1))):
            count += 1
    return count

def _greedy_init(ap3_partners, ap4_partners):
    """
    Greedy initialization with randomness.
    """
    mask = 0
    current_candidates = list(range(1, N + 1))
    random.shuffle(current_candidates)
    
    while current_candidates:
        valid_candidates = []
        gains = {}
        
        for y in current_candidates:
            if _is_valid_addition(y, mask, ap4_partners):
                gain = _calculate_gain(y, mask, ap3_partners)
                valid_candidates.append(y)
                gains[y] = gain
        
        if not valid_candidates:
            break
        
        # Select from top K with some randomness
        K = min(8, len(valid_candidates))
        valid_candidates.sort(key=lambda x: gains[x], reverse=True)
        top_k = valid_candidates[:K]
        
        # Weighted random selection favoring higher gains
        weights = [gains[y] + 1 for y in top_k]
        total_w = sum(weights)
        r = random.random() * total_w
        cum = 0
        chosen = top_k[0]
        for i, y in enumerate(top_k):
            cum += weights[i]
            if r <= cum:
                chosen = y
                break
        
        mask |= (1 << (chosen - 1))
        current_candidates.remove(chosen)
        
    return mask

def _simulated_annealing(mask, ap3_partners, ap4_partners, time_limit=5.0):
    """
    Simulated Annealing to maximize 3-APs while maintaining 4-AP free.
    """
    start_time = time.time()
    current_mask = mask
    current_score = _count_3aps(current_mask, ap3_partners)
    best_mask = current_mask
    best_score = current_score
    
    present = [i for i in range(1, N + 1) if (current_mask & (1 << (i - 1)))]
    absent = [i for i in range(1, N + 1) if not (current_mask & (1 << (i - 1)))]
    
    T = 50.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    steps = 0
    max_steps = 100000
    
    while time.time() - start_time < time_limit and T > T_min and steps < max_steps:
        # Pick a random move type: Add, Remove, Swap
        move_type = random.random()
        
        best_delta = -10**9
        best_move = None
        
        # We don't want to evaluate ALL moves, just a random sample to keep it fast
        # But SA benefits from exploring the neighborhood.
        # Let's pick a random candidate for each move type.
        
        candidate_move = None
        delta = 0
        
        if move_type < 0.3 and absent:
            # Add
            y = random.choice(absent)
            if _is_valid_addition(y, current_mask, ap4_partners):
                delta = _calculate_gain(y, current_mask, ap3_partners)
                candidate_move = (0, y)
        elif move_type < 0.6 and present:
            # Remove
            x = random.choice(present)
            delta = -_calculate_loss(x, current_mask, ap3_partners)
            candidate_move = (1, x)
        else:
            # Swap
            if present and absent:
                x = random.choice(present)
                y = random.choice(absent)
                if _is_valid_swap(x, y, current_mask, ap4_partners):
                    gain_y = _calculate_gain(y, current_mask, ap3_partners, exclude_x=x)
                    loss_x = _calculate_loss(x, current_mask, ap3_partners)
                    delta = gain_y - loss_x
                    candidate_move = (2, x, y)
        
        if candidate_move is None:
            # No valid move of chosen type, try another or break
            continue
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            # Apply move
            if candidate_move[0] == 0: # Add
                y = candidate_move[1]
                current_mask |= (1 << (y - 1))
                present.append(y)
                absent.remove(y)
            elif candidate_move[0] == 1: # Remove
                x = candidate_move[1]
                current_mask &= ~(1 << (x - 1))
                present.remove(x)
                absent.append(x)
            else: # Swap
                x, y = candidate_move[1], candidate_move[2]
                current_mask &= ~(1 << (x - 1))
                current_mask |= (1 << (y - 1))
                present.remove(x)
                present.append(y)
                absent.remove(y)
                absent.append(x)
            
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling_rate
        steps += 1
        
    return best_score, best_mask

def construct():
    start_time = time.time()
    ap3_partners, ap4_partners = _precompute()
    
    best_score = -1
    best_mask = 0
    
    # Multiple restarts
    while time.time() - start_time < DEADLINE - 2.0:
        # Greedy initialization
        mask = _greedy_init(ap3_partners, ap4_partners)
        
        # Simulated Annealing
        score, refined_mask = _simulated_annealing(mask, ap3_partners, ap4_partners, time_limit=4.0)
        
        if score > best_score:
            best_score = score
            best_mask = refined_mask
            
        # Check time
        if time.time() - start_time > DEADLINE - 5.0:
            # Final polish on best found
            if best_score > 0:
                score, refined_mask = _simulated_annealing(best_mask, ap3_partners, ap4_partners, time_limit=3.0)
                if score > best_score:
                    best_score = score
                    best_mask = refined_mask
            break
    
    # Convert mask to list
    result = [i for i in range(1, N + 1) if (best_mask & (1 << (i - 1)))]
    
    # If result is empty (shouldn't happen), return a valid small set
    if not result:
        result = [1]
        
    return result
