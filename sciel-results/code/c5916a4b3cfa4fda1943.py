import time
import random
import sys

N = 256
DEADLINE = 55.0

def _precompute():
    """
    Precompute data structures for fast 3-AP counting and 4-AP validation.
    """
    # 3-APs: (a, a+d, a+2d)
    three_aps = []
    three_ap_metas = [] # (a, b, c)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
            three_ap_metas.append((a-1, b-1, c-1))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    four_aps = []
    four_ap_metas = [] # (a, b, c, d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps.append(mask)
            four_ap_metas.append((a-1, b-1, c-1, e-1))
    
    # Map each element index to list of 4-AP indices containing it
    # And for each such 4-AP, store the indices of the OTHER 3 elements
    elem_to_4ap_others = [[] for _ in range(N)]
    for idx, (a, b, c, e) in enumerate(four_ap_metas):
        # For element a, others are b, c, e
        elem_to_4ap_others[a].append((b, c, e))
        # For element b, others are a, c, e
        elem_to_4ap_others[b].append((a, c, e))
        # For element c, others are a, b, e
        elem_to_4ap_others[c].append((a, b, e))
        # For element e, others are a, b, c
        elem_to_4ap_others[e].append((a, b, c))
        
    # Map each element index to list of 3-AP indices containing it
    # And for each such 3-AP, store the indices of the OTHER 2 elements
    elem_to_3ap_others = [[] for _ in range(N)]
    for idx, (a, b, c) in enumerate(three_ap_metas):
        elem_to_3ap_others[a].append((b, c))
        elem_to_3ap_others[b].append((a, c))
        elem_to_3ap_others[c].append((a, b))
        
    return elem_to_4ap_others, elem_to_3ap_others

def _count_3aps(mask, elem_to_3ap_others):
    """
    Count total 3-APs in the set represented by mask.
    This is O(N * K) where K is avg 3-APs per element.
    Used for initialization and occasional full recounts.
    """
    count = 0
    for i in range(N):
        if mask & (1 << i):
            for b, c in elem_to_3ap_others[i]:
                # Check if b and c are in mask
                if (mask & (1 << b)) and (mask & (1 << c)):
                    count += 1
    # Each 3-AP is counted 3 times (once for each element)
    return count // 3

def _is_valid_addition(x, mask, elem_to_4ap_others):
    """
    Check if adding element x to set 'mask' creates a 4-AP.
    """
    # If x is already in mask, it's not an addition
    if mask & (1 << x):
        return False
        
    for b, c, d in elem_to_4ap_others[x]:
        # Check if b, c, d are all in mask
        if (mask & (1 << b)) and (mask & (1 << c)) and (mask & (1 << d)):
            return False
    return True

def _calculate_gain_add(x, mask, elem_to_3ap_others):
    """
    Calculate the increase in 3-AP count if x is added to 'mask'.
    """
    gain = 0
    for b, c in elem_to_3ap_others[x]:
        if (mask & (1 << b)) and (mask & (1 << c)):
            gain += 1
    return gain

def _calculate_loss_remove(x, mask, elem_to_3ap_others):
    """
    Calculate the decrease in 3-AP count if x is removed from 'mask'.
    """
    loss = 0
    for b, c in elem_to_3ap_others[x]:
        if (mask & (1 << b)) and (mask & (1 << c)):
            loss += 1
    return loss

def _calculate_gain_swap(x_out, x_in, mask, elem_to_3ap_others):
    """
    Calculate net change in 3-AP count if x_out is removed and x_in is added.
    """
    # Loss from removing x_out
    # Note: The 3-APs lost are those containing x_out and two other elements in (mask - {x_out})
    # Since x_in is not in mask yet, we just check against current mask.
    # Any 3-AP containing x_out and two elements from mask will be lost.
    loss = 0
    for b, c in elem_to_3ap_others[x_out]:
        if (mask & (1 << b)) and (mask & (1 << c)):
            loss += 1
            
    # Gain from adding x_in
    # The 3-APs gained are those containing x_in and two other elements in (mask - {x_out})
    # We must exclude x_out from the check for gain, because x_out is being removed.
    gain = 0
    mask_without_out = mask & ~(1 << x_out)
    for b, c in elem_to_3ap_others[x_in]:
        if (mask_without_out & (1 << b)) and (mask_without_out & (1 << c)):
            gain += 1
            
    return gain - loss

def construct():
    start_time = time.time()
    elem_to_4ap_others, elem_to_3ap_others = _precompute()
    
    # Initialize with a random valid set or greedy
    # Greedy: add elements in random order if valid
    random.seed(42)
    
    # Strategy: Start with a sparse random set, then optimize.
    # Or start with all elements and remove those causing 4-APs? No, that's hard.
    # Let's start with an empty set and add random elements greedily to get a valid base.
    mask = 0
    current_score = 0
    
    # Greedy initialization
    # Shuffle elements
    elements = list(range(N))
    random.shuffle(elements)
    
    for x in elements:
        if _is_valid_addition(x, mask, elem_to_4ap_others):
            mask |= (1 << x)
            current_score += _calculate_gain_add(x, mask ^ (1 << x), elem_to_3ap_others)
            
    best_mask = mask
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    # Precompute bit masks for speed
    bits = [1 << i for i in range(N)]
    
    while time.time() - start_time < DEADLINE:
        T *= cooling
        if T < T_min:
            T = T_min
            
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            # Find candidates that are not in mask
            # To speed up, pick a random index and check
            x = random.randint(0, N - 1)
            if not (mask & bits[x]):
                if _is_valid_addition(x, mask, elem_to_4ap_others):
                    delta = _calculate_gain_add(x, mask, elem_to_3ap_others)
                    if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                        mask |= bits[x]
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
                            
        elif move_type < 0.5:
            # Remove
            # Pick a random element from mask
            # To pick efficiently, we can just pick a random index and check if it's in mask
            # Or maintain a list of present elements. Maintaining a list is O(N) update.
            # Given N=256, picking random index is fine.
            x = random.randint(0, N - 1)
            if mask & bits[x]:
                delta = -_calculate_loss_remove(x, mask, elem_to_3ap_others)
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    mask &= ~bits[x]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                        
        else:
            # Swap
            # Pick x_out from mask, x_in from not mask
            x_out = random.randint(0, N - 1)
            x_in = random.randint(0, N - 1)
            
            if (mask & bits[x_out]) and not (mask & bits[x_in]):
                # Check validity of adding x_in to (mask - x_out)
                # We can check _is_valid_addition(x_in, mask ^ bits[x_out], ...)
                # But _is_valid_addition checks if x_in is in the set. We pass mask without x_out.
                # Wait, _is_valid_addition(x, mask, ...) checks if adding x to mask creates 4-AP.
                # Here the new set is (mask - {x_out}) U {x_in}.
                # So we check if adding x_in to (mask - {x_out}) is valid.
                
                mask_temp = mask & ~bits[x_out]
                
                if _is_valid_addition(x_in, mask_temp, elem_to_4ap_others):
                    delta = _calculate_gain_swap(x_out, x_in, mask, elem_to_3ap_others)
                    if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                        mask = mask_temp | bits[x_in]
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask

    # Convert best_mask back to sorted list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)
