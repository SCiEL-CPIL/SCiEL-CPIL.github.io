import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    ap_list = []
    elem_to_aps = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                break
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
            elem_to_aps[a].append(ap_idx)
            elem_to_aps[b].append(ap_idx)
            elem_to_aps[c].append(ap_idx)
            elem_to_aps[e].append(ap_idx)
            
    num_aps = len(ap_list)
    
    # Precompute "others" masks for each AP and each element position
    # others_mask[ap_idx][pos] is a bitmask of the 3 elements in ap_list[ap_idx] excluding the one at pos
    others_mask = [[0]*4 for _ in range(num_aps)]
    for idx, (a, b, c, e) in enumerate(ap_list):
        elems = [a, b, c, e]
        for p in range(4):
            m = 0
            for q in range(4):
                if q != p:
                    m |= (1 << (elems[q] - 1))
            others_mask[idx][p] = m
            
    # To speed up finding position of x in AP, we can just check equality or precompute pos
    # Since APs are small (4), linear search is fine, but let's precompute pos for safety/speed
    # pos_in_ap[ap_idx] = dict {elem: pos} or just list of elems and check
    # Actually, we can just iterate elem_to_aps[x] and check which elem of ap matches x.
    
    def can_add(x, mask, counts):
        """
        Check if x can be added to the set represented by mask.
        counts is used for O(1) check? No, counts tells us how many are present.
        If counts[ap_idx] == 3, then adding x (which is in that AP) makes it 4.
        Wait, if x is in the AP, and counts[ap_idx] is 3, that means the OTHER 3 are in the set.
        So yes, if counts[ap_idx] == 3, x CANNOT be added.
        This is much faster than bitmask checking!
        """
        for ap_idx in elem_to_aps[x]:
            if counts[ap_idx] == 3:
                return False
        return True

    def do_add(x, mask, counts, size):
        mask |= (1 << (x - 1))
        size += 1
        for ap_idx in elem_to_aps[x]:
            counts[ap_idx] += 1
        return mask, size

    def do_remove(x, mask, counts, size):
        mask &= ~(1 << (x - 1))
        size -= 1
        for ap_idx in elem_to_aps[x]:
            counts[ap_idx] -= 1
        return mask, size

    best_size = 0
    best_mask = 0
    
    # List of all elements 1..n
    all_elems = list(range(1, n + 1))
    
    num_restarts = 1000 # High number, limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        mask = 0
        counts = [0] * num_aps
        size = 0
        
        # Greedy Phase
        candidates = all_elems[:]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, mask, counts):
                mask, size = do_add(x, mask, counts, size)
                
        # Local Search / Simulated Annealing Phase
        # We will use a simple SA-like approach with neutral swaps and occasional perturbations
        T = 1.0
        cooling = 0.99
        min_T = 0.1
        
        # We need a list of elements in the set for random selection
        # Reconstructing from mask is O(n). We can maintain a list, but updates are tricky.
        # For n=512, iterating all elems to find in/out is O(n) = 512.
        # Doing this every step is too slow if steps are many.
        # Let's limit local search steps per restart.
        
        max_steps = 2000
        steps = 0
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            # Generate in_set and out_set lists occasionally?
            # Or just pick random x in 1..n and check if in mask.
            
            action = random.random()
            
            if action < 0.7:
                # Try to add a random element
                x = random.randint(1, n)
                if not (mask & (1 << (x - 1))): # x not in set
                    if can_add(x, mask, counts):
                        mask, size = do_add(x, mask, counts, size)
                        # Always accept improvement
                    else:
                        # Failed to add. Do nothing or try swap?
                        pass
            else:
                # Try a swap
                # Pick x not in set, y in set
                # To pick y in set efficiently, we could keep a list, but let's just sample
                # Since size is ~100-200, probability of picking y in set is low if we pick random 1..n
                # Better: Pick y from a precomputed list of current set elements?
                # Let's reconstruct in_set/out_set every 100 steps or if needed?
                # For now, let's just try to pick y by sampling.
                # If we pick y not in set, we can't remove it.
                
                # Optimized Swap:
                # Pick random x (1..n). If x in set, swap roles? No, x must be OUT.
                # Pick random y (1..n). If y NOT in set, swap roles? No, y must be IN.
                
                # Let's just iterate a few times to find valid x, y
                found = False
                for _ in range(10): # Max 10 attempts to find valid pair
                    x = random.randint(1, n)
                    y = random.randint(1, n)
                    if x == y: continue
                    
                    x_in = bool(mask & (1 << (x - 1)))
                    y_in = bool(mask & (1 << (y - 1)))
                    
                    if x_in and not y_in:
                        # Swap: remove x, add y
                        # Remove x
                        mask, size = do_remove(x, mask, counts, size)
                        # Try add y
                        if can_add(y, mask, counts):
                            mask, size = do_add(y, mask, counts, size)
                            # Size unchanged. Accept with prob 0.5 to explore
                            if random.random() < 0.5:
                                pass # Accepted
                            else:
                                # Revert
                                mask, size = do_remove(y, mask, counts, size)
                                mask, size = do_add(x, mask, counts, size)
                        else:
                            # Revert
                            mask, size = do_add(x, mask, counts, size)
                        found = True
                        break
                    elif not x_in and y_in:
                        # Swap: remove y, add x
                        # Remove y
                        mask, size = do_remove(y, mask, counts, size)
                        # Try add x
                        if can_add(x, mask, counts):
                            mask, size = do_add(x, mask, counts, size)
                            # Size unchanged. Accept with prob 0.5
                            if random.random() < 0.5:
                                pass
                            else:
                                # Revert
                                mask, size = do_remove(x, mask, counts, size)
                                mask, size = do_add(y, mask, counts, size)
                        else:
                            # Revert
                            mask, size = do_add(y, mask, counts, size)
                        found = True
                        break
                    # else: both in or both out, try again
                
                if not found:
                    # Couldn't find a swap candidate, maybe set is full or empty?
                    # If full, break local search
                    if size == n:
                        break
                    # Else, just skip step

            # Cool down
            T *= cooling
            if T < min_T:
                T = min_T
                
        # Update best
        if size > best_size:
            best_size = size
            best_mask = mask
            # Optional: try to polish best solution with more local search?
            # For now, just record.

    # Convert best_mask to list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return result
