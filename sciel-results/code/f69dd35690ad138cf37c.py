import time
import random
import sys

N = 256
TIME_LIMIT = 55.0  # Safety margin from 60s

def _precompute():
    # Precompute 3-AP and 4-AP masks
    # mask[i] corresponds to element i+1 (0-indexed bit i)
    
    # 3-APs: a, a+d, a+2d
    m3 = []
    for d in range(1, N):
        for a in range(1, N - 2*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1))
            m3.append(mask)
            
    # 4-APs: a, a+d, a+2d, a+3d
    m4 = []
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1)) | (1 << (a+3*d-1))
            m4.append(mask)
            
    # For each element x (0-indexed), compute:
    # 1. conflict_mask[x]: Bits that cannot be in A if x is in A (to prevent 4-APs)
    #    If x is in a 4-AP, the other 3 bits cannot all be in A.
    #    Actually, if we add x, we must ensure that for every 4-AP containing x,
    #    not all other 3 bits are in A.
    #    A simpler constraint for "Add" move: 
    #    If we want to add x, we need to check if adding x completes any 4-AP.
    #    A 4-AP is completed if the other 3 bits are already in A.
    #    So, for each x, let 'forbidden_pairs' or similar? 
    #    Better: For each x, precompute the set of 4-APs it belongs to.
    #    But checking "does adding x create a 4-AP" is:
    #    For each 4-AP mask M containing x, if (A & (M ^ (1<<x))) == (M ^ (1<<x)), then adding x creates a 4-AP.
    
    # Let's precompute for each x:
    # x_4ap_masks: list of masks of the other 3 bits in each 4-AP containing x.
    x_4ap_others = [[] for _ in range(N)]
    for m in m4:
        # Find bits set in m
        bits = []
        for i in range(N):
            if m & (1 << i):
                bits.append(i)
        # For each bit in this 4-AP, the other 3 bits are the conflict set for adding that bit
        for b in bits:
            others = m ^ (1 << b)
            x_4ap_others[b].append(others)
            
    # x_3ap_gain: For each x, the mask of bits that, if present in A, 
    # form a 3-AP with x.
    # If x is the first term: x, x+d, x+2d -> gain bits are x+d, x+2d
    # If x is the middle: x-d, x, x+d -> gain bits are x-d, x+d
    # If x is the last: x-2d, x-d, x -> gain bits are x-2d, x-d
    # Note: A 3-AP is counted once.
    # If we add x, we form a 3-AP if the other two elements are in A.
    # So gain_mask[x] is the union of pairs? No, it's a mask of single bits.
    # Wait, to count the number of 3-APs formed by adding x:
    # We need to count how many pairs (u, v) are in A such that {u, v, x} is a 3-AP.
    # This is equivalent to: For each 3-AP containing x, check if the other 2 bits are in A.
    # We can precompute for each x a list of "other 2 bits masks" for all 3-APs containing x.
    
    x_3ap_others = [[] for _ in range(N)]
    for m in m3:
        bits = []
        for i in range(N):
            if m & (1 << i):
                bits.append(i)
        for b in bits:
            others = m ^ (1 << b)
            x_3ap_others[b].append(others)
            
    return x_4ap_others, x_3ap_others

X_4AP_OTHERS, X_3AP_OTHERS = _precompute()

def _is_valid_addition(x, current_mask):
    """Check if adding bit x to current_mask creates a 4-AP."""
    if current_mask & (1 << x):
        return True # Already there, no change
    # Check all 4-APs containing x
    for others_mask in X_4AP_OTHERS[x]:
        if (current_mask & others_mask) == others_mask:
            return False
    return True

def _count_new_3aps(x, current_mask):
    """Count how many 3-APs are formed by adding bit x to current_mask."""
    count = 0
    for others_mask in X_3AP_OTHERS[x]:
        if (current_mask & others_mask) == others_mask:
            count += 1
    return count

def _count_3aps(mask):
    """Count total 3-APs in mask."""
    # This is O(N^3) if naive, but we can use the precomputed lists?
    # Actually, summing _count_new_3aps for all x in mask / 3 works?
    # No, each 3-AP is counted 3 times (once for each element).
    total = 0
    for x in range(N):
        if mask & (1 << x):
            total += _count_new_3aps(x, mask)
    return total // 3

def _greedy_construct(random_order=None):
    """Construct a 4-AP-free set using greedy method."""
    if random_order is None:
        random_order = list(range(N))
        random.shuffle(random_order)
        
    mask = 0
    # We prioritize elements that form many 3-APs with already selected elements.
    # Simple greedy: iterate through order, add if valid.
    # To improve, we can pick the best valid element at each step.
    # But that is O(N^2) per step. With N=256, 256*256 = 65k ops per step, 256 steps -> 16M ops. Fast enough.
    
    remaining = set(random_order)
    while remaining:
        best_x = None
        best_score = -1
        for x in remaining:
            if _is_valid_addition(x, mask):
                score = _count_new_3aps(x, mask)
                if score > best_score:
                    best_score = score
                    best_x = x
        if best_x is None:
            break
        mask |= (1 << best_x)
        remaining.remove(best_x)
        
    return mask

def _local_search(mask, time_limit):
    """
    Hill Climbing with Add, Remove, and Swap moves.
    """
    current_mask = mask
    current_score = _count_3aps(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try Add moves
        # Iterate through bits not in mask
        # Optimization: Only try bits that are "close" to existing bits?
        # For now, try all.
        for x in range(N):
            if not (current_mask & (1 << x)):
                if _is_valid_addition(x, current_mask):
                    delta = _count_new_3aps(x, current_mask)
                    if delta > 0:
                        # Apply
                        current_mask |= (1 << x)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                        improved = True
                        no_improve_count = 0
                        break # Restart search with new mask
        
        if improved:
            continue
            
        # 2. Try Swap moves (Remove x, Add y)
        # This is O(N^2). 256*256 = 65k checks.
        # Check if valid and delta > 0.
        # To speed up, we can iterate y first, then x?
        # Or just nested loop.
        
        found_swap = False
        # We need to be careful: removing x might allow adding y which was previously invalid.
        # Or removing x might break 3-APs, but adding y makes more.
        
        for y in range(N):
            if current_mask & (1 << y):
                continue # y is already in mask, can't add it (unless we remove it first, which is a remove move, handled separately or in swap)
            
            # Check if y is valid to add to (current_mask - x) for some x?
            # This depends on x.
            # Let's iterate x (to remove) and y (to add).
            # To avoid O(N^3) or heavy O(N^2) with expensive checks, we can be smart.
            
        # Efficient Swap Search:
        # For each x in mask:
        #   temp_mask = current_mask ^ (1 << x)
        #   For each y not in temp_mask:
        #      if _is_valid_addition(y, temp_mask):
        #          delta_remove = -_count_lost_3aps(x, current_mask) 
        #          delta_add = _count_new_3aps(y, temp_mask)
        #          # Note: _count_new_3aps(y, temp_mask) counts 3-APs formed with y in temp_mask.
        #          # The total change is delta_add - delta_remove.
        #          # However, delta_remove calculation is expensive if done naively.
        #          # We can precompute "loss" for x?
        #          # Loss of x is the number of 3-APs containing x that are present in current_mask.
        #          # This is _count_new_3aps(x, current_mask).
        
        loss_x_cache = {}
        for x in range(N):
            if current_mask & (1 << x):
                loss_x_cache[x] = _count_new_3aps(x, current_mask)
        
        for x in range(N):
            if not (current_mask & (1 << x)):
                continue
            temp_mask = current_mask ^ (1 << x)
            loss = loss_x_cache[x]
            
            for y in range(N):
                if temp_mask & (1 << y):
                    continue
                # y is not in temp_mask (so y != x and y not in current_mask)
                if _is_valid_addition(y, temp_mask):
                    gain = _count_new_3aps(y, temp_mask)
                    if gain > loss:
                        # Apply Swap
                        current_mask = temp_mask | (1 << y)
                        current_score += (gain - loss)
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                        improved = True
                        no_improve_count = 0
                        found_swap = True
                        break
            if found_swap:
                break
                
        if improved:
            continue
            
        # 3. Try Remove moves (if stuck)
        # Removing an element reduces score, so only useful if it enables better swaps later?
        # Hill climbing usually doesn't do pure removals unless part of a swap.
        # However, sometimes removing a "bad" element allows adding two "good" ones?
        # That would be a 2-swap.
        
        no_improve_count += 1
        if no_improve_count > 20:
            # Perturb
            # Randomly remove 2-5 elements
            num_remove = random.randint(2, 5)
            bits_in_mask = [i for i in range(N) if current_mask & (1 << i)]
            if len(bits_in_mask) > num_remove:
                to_remove = random.sample(bits_in_mask, num_remove)
                for x in to_remove:
                    current_mask &= ~(1 << x)
                current_score = _count_3aps(current_mask)
                no_improve_count = 0
            else:
                break # Can't remove more, stuck
                
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_mask = 0
    best_score = -1
    
    # Multi-start
    num_restarts = 10
    time_per_restart = (TIME_LIMIT - 5.0) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Generate initial mask
        if i == 0:
            # Deterministic greedy for first run? Or random?
            # Random greedy provides diversity.
            mask = _greedy_construct()
        else:
            # Perturb best found
            mask = best_mask
            # Flip 5-15 bits
            num_flips = random.randint(5, 15)
            for _ in range(num_flips):
                bit = random.randint(0, N-1)
                mask ^= (1 << bit)
            
            # Repair: ensure no 4-APs
            # If a 4-AP is present, remove one random bit from it.
            changed = True
            while changed:
                changed = False
                # Check all 4-APs
                # To speed up, we can check only 4-APs involving bits that were flipped?
                # But simple full check is O(N^3) ~ 16M, which is okay for a few iterations.
                # Actually, _is_4ap_free is not precomputed as a single mask check.
                # Let's just iterate through 4-AP masks.
                # We don't have M4 list directly in scope here, but we can reconstruct or use a helper.
                # Let's use a helper that checks validity.
                # Better: Precompute M4 list globally.
                
                # For now, simple repair:
                # Check if mask is valid. If not, find a 4-AP and break it.
                # We need access to M4. Let's assume we have it.
                
                # I'll add a helper _is_4ap_free and _find_4ap
                # Actually, _local_search assumes input is valid?
                # My _greedy_construct produces valid mask.
                # My perturbation might create 4-APs.
                # So repair is needed.
                
                # Let's implement a quick repair
                # Check all 4-APs
                # This is expensive. 
                # Alternative: Just run local search which handles validity?
                # No, local search assumes current state is valid.
                # So we MUST repair.
                
                # Efficient repair:
                # For each bit x in mask, check if adding x to (mask ^ (1<<x)) was valid?
                # No, that's not right.
                
                # Let's just use the M4 list.
                # I will define M4 globally.
                
                for m in M4_LIST:
                    if (mask & m) == m:
                        # Found a 4-AP. Remove a random bit from it.
                        bits_in_ap = [b for b in range(N) if m & (1 << b)]
                        b_to_remove = random.choice(bits_in_ap)
                        mask &= ~(1 << b_to_remove)
                        changed = True
                        break
        
        score = _count_3aps(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask
            
        # Run local search
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            break
            
        ls_mask, ls_score = _local_search(mask, remaining_time - 1.0)
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
            
    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)

# Global M4 list for repair
def _get_m4_list():
    m4 = []
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1)) | (1 << (a+3*d-1))
            m4.append(mask)
    return m4

M4_LIST = _get_m4_list()

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    mask = 0
    for x in res:
        mask |= (1 << (x-1))
    print(f"Fitness: {_count_3aps(mask)}")
    
    # Validate
    valid = True
    for m in M4_LIST:
        if (mask & m) == m:
            valid = False
            break
    print(f"Valid: {valid}")
