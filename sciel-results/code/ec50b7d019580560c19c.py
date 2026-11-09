import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    # Bit indices 0 to 255 correspond to numbers 1 to 256
    # bit i represents number i+1
    
    # Precomputation
    # 1. Find all 4-APs and 3-APs
    # 4-AP: a, a+d, a+2d, a+3d
    # 3-AP: a, a+d, a+2d
    
    # For each x (1-based index 1..256, 0-based 0..255), 
    # store masks of other elements for 4-APs and 3-APs containing x.
    
    # 4-AP conflicts: 
    # If we add x, we create a 4-AP if there are 3 other elements in A that form a 4-AP with x.
    # Let M be the set of 4 other elements? No, 4-AP has 4 elements.
    # If x is part of a 4-AP {x, u, v, w}, then adding x creates a 4-AP iff {u, v, w} subset of A.
    
    # Precompute for each x (0-indexed):
    # four_conflicts[x] = list of bitmasks representing the set of the OTHER 3 elements in 4-APs containing x.
    # three_contributions[x] = list of bitmasks representing the set of the OTHER 2 elements in 3-APs containing x.
    
    four_conflicts = [[] for _ in range(N)]
    three_contributions = [[] for _ in range(N)]
    
    # Generate all 4-APs
    for d in range(1, (N-3)//3 + 1):
        for a in range(1, N - 3*d + 1):
            # a, a+d, a+2d, a+3d
            idxs = [a-1, a+d-1, a+2*d-1, a+3*d-1]
            # For each element in this 4-AP, the other 3 form a conflict mask
            mask_full = 0
            for idx in idxs:
                mask_full |= (1 << idx)
            
            for i, idx in enumerate(idxs):
                # Mask of the other 3
                mask_other = mask_full ^ (1 << idx)
                four_conflicts[idx].append(mask_other)
                
    # Generate all 3-APs
    for d in range(1, (N-2)//2 + 1):
        for a in range(1, N - 2*d + 1):
            # a, a+d, a+2d
            idxs = [a-1, a+d-1, a+2*d-1]
            mask_full = 0
            for idx in idxs:
                mask_full |= (1 << idx)
            
            for i, idx in enumerate(idxs):
                # Mask of the other 2
                mask_other = mask_full ^ (1 << idx)
                three_contributions[idx].append(mask_other)
                
    # Helper: Check if adding x is valid (no 4-AP formed)
    def is_valid_add(x, A_mask):
        # x is 0-indexed
        bit_x = 1 << x
        if A_mask & bit_x:
            return False # Already in set
            
        for mask_other in four_conflicts[x]:
            # If the other 3 elements are in A, adding x creates a 4-AP
            if (A_mask & mask_other) == mask_other:
                return False
        return True
        
    # Helper: Calculate gain in 3-APs if x is added
    def get_3ap_gain(x, A_mask):
        # x is 0-indexed
        gain = 0
        for mask_other in three_contributions[x]:
            # If the other 2 elements are in A, adding x forms a 3-AP
            if (A_mask & mask_other) == mask_other:
                gain += 1
        return gain
        
    # Helper: Calculate total 3-APs in A_mask
    def count_3aps(A_mask):
        count = 0
        # Iterate through all elements in A
        # To avoid double counting, we can just sum contributions of each x in A 
        # and divide by something? 
        # Each 3-AP (a,b,c) is counted once for a, once for b, once for c.
        # So summing get_3ap_gain(x, A_mask \ {x}) for all x in A will count each 3-AP 3 times.
        # Alternatively, just iterate all 3-APs and check if subset of A.
        # Number of 3-APs is ~ 256^2/4 = 16000.
        # Checking 16000 masks is fast.
        
        # Let's precompute all 3-AP masks for faster counting?
        # Actually, we can just use the three_contributions logic but carefully.
        # If we sum gain(x, A) for x in A, we count each 3-AP 3 times.
        # So count = sum(gain(x, A) for x in A) // 3?
        # Wait, gain(x, A) checks if OTHER 2 are in A.
        # If x is in A, and other 2 are in A, then the 3-AP is in A.
        # So yes, summing gain(x, A) for all x in A counts each 3-AP exactly 3 times.
        
        total = 0
        temp_mask = A_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            x = lsb.bit_length() - 1
            total += get_3ap_gain(x, A_mask)
            temp_mask ^= lsb
            
        return total // 3

    # Initial Solution: Best-First Greedy
    A_mask = 0
    available = set(range(N))
    
    # We will greedily add elements that maximize 3-AP gain while maintaining no 4-AP.
    # To speed up, we can pre-sort or just scan.
    # Scanning 256 elements * 256 steps * (check cost)
    # Check cost is len(four_conflicts[x]) bitwise ANDs. Avg len ~ 10000/256 ~ 40.
    # 256 * 256 * 40 = 2.6M ops. Very fast.
    
    while available and time.time() < deadline - 10:
        best_x = -1
        best_gain = -1
        for x in available:
            if is_valid_add(x, A_mask):
                gain = get_3ap_gain(x, A_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
        if best_x == -1:
            break
            
        A_mask |= (1 << best_x)
        available.remove(best_x)
        
    # Local Search / Hill Climbing
    # We will try to improve A_mask by swapping elements or adding/removing.
    # Since we want to maximize 3-APs, and we are constrained by 4-APs,
    # we can try:
    # 1. Add a random valid element.
    # 2. Remove an element and try to replace it with a different one.
    
    best_A_mask = A_mask
    best_score = count_3aps(A_mask)
    
    # Precompute list of 3-AP masks for faster global count if needed?
    # count_3aps is O(N * K) where K is avg 3-APs per element.
    # N=256, K~100. 25600 ops. Fast enough for frequent calls?
    # Maybe. But we can be smarter.
    
    # Let's use a simple Hill Climbing with random restarts if stuck.
    current_A_mask = A_mask
    current_score = best_score
    
    # To make it faster, we can update score incrementally.
    # But for simplicity and robustness, we'll rely on the fast count_3aps.
    
    # We'll do a loop of random moves.
    # Move types:
    # 1. Add x (if valid)
    # 2. Remove x
    # 3. Swap x, y (remove x, add y if valid)
    
    # To avoid getting stuck in local optima, we can use Simulated Annealing.
    # But Hill Climbing with restarts is often sufficient for this size.
    
    # Let's stick to Hill Climbing first. If no improvement for many steps, restart with random perturbation.
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 2:
        # Pick a random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_A_mask = current_A_mask
        valid_move = True
        new_score = -1
        
        if move_type == 'add':
            # Pick random x not in A
            not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
            if not not_in_A:
                continue
            x = random.choice(not_in_A)
            if is_valid_add(x, current_A_mask):
                new_A_mask = current_A_mask | (1 << x)
                new_score = count_3aps(new_A_mask)
            else:
                valid_move = False
                
        elif move_type == 'remove':
            in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
            if not in_A:
                continue
            x = random.choice(in_A)
            new_A_mask = current_A_mask & ~(1 << x)
            new_score = count_3aps(new_A_mask)
            
        elif move_type == 'swap':
            in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
            not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
            if not in_A or not not_in_A:
                continue
            x = random.choice(in_A)
            y = random.choice(not_in_A)
            
            # Remove x, add y
            temp_mask = current_A_mask & ~(1 << x)
            if is_valid_add(y, temp_mask):
                new_A_mask = temp_mask | (1 << y)
                new_score = count_3aps(new_A_mask)
            else:
                valid_move = False
                
        if valid_move:
            if new_score > current_score:
                current_A_mask = new_A_mask
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_A_mask = current_A_mask
            else:
                no_improve_count += 1
                
                # If stuck, restart with a random valid set or perturb current
                if no_improve_count > max_no_improve:
                    # Perturb: remove 5 random elements, add 5 random valid ones
                    # Or just restart greedy from scratch?
                    # Restarting greedy is safe but slow.
                    # Let's just randomize a bit.
                    for _ in range(5):
                        in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
                        if in_A:
                            x = random.choice(in_A)
                            current_A_mask &= ~(1 << x)
                            
                    # Try to fill back up
                    for _ in range(5):
                        not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
                        if not not_in_A: break
                        # Find best valid add
                        best_x = -1
                        best_gain = -1
                        for x in not_in_A:
                            if is_valid_add(x, current_A_mask):
                                gain = get_3ap_gain(x, current_A_mask)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_x = x
                        if best_x != -1:
                            current_A_mask |= (1 << best_x)
                            
                    current_score = count_3aps(current_A_mask)
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_A_mask = current_A_mask
                        
    # Convert best_A_mask to sorted list
    result = []
    for i in range(N):
        if best_A_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

# EVOLVE-BLOCK-START
def construct():
    # Call the internal logic
    return _construct_impl()

def _construct_impl():
    # (Copy of the code above, indented)
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    
    four_conflicts = [[] for _ in range(N)]
    three_contributions = [[] for _ in range(N)]
    
    for d in range(1, (N-3)//3 + 1):
        for a in range(1, N - 3*d + 1):
            idxs = [a-1, a+d-1, a+2*d-1, a+3*d-1]
            mask_full = 0
            for idx in idxs:
                mask_full |= (1 << idx)
            for i, idx in enumerate(idxs):
                mask_other = mask_full ^ (1 << idx)
                four_conflicts[idx].append(mask_other)
                
    for d in range(1, (N-2)//2 + 1):
        for a in range(1, N - 2*d + 1):
            idxs = [a-1, a+d-1, a+2*d-1]
            mask_full = 0
            for idx in idxs:
                mask_full |= (1 << idx)
            for i, idx in enumerate(idxs):
                mask_other = mask_full ^ (1 << idx)
                three_contributions[idx].append(mask_other)
                
    def is_valid_add(x, A_mask):
        bit_x = 1 << x
        if A_mask & bit_x:
            return False
        for mask_other in four_conflicts[x]:
            if (A_mask & mask_other) == mask_other:
                return False
        return True
        
    def get_3ap_gain(x, A_mask):
        gain = 0
        for mask_other in three_contributions[x]:
            if (A_mask & mask_other) == mask_other:
                gain += 1
        return gain
        
    def count_3aps(A_mask):
        total = 0
        temp_mask = A_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            x = lsb.bit_length() - 1
            total += get_3ap_gain(x, A_mask)
            temp_mask ^= lsb
        return total // 3

    A_mask = 0
    available = set(range(N))
    
    while available and time.time() < deadline - 10:
        best_x = -1
        best_gain = -1
        for x in available:
            if is_valid_add(x, A_mask):
                gain = get_3ap_gain(x, A_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
        if best_x == -1:
            break
            
        A_mask |= (1 << best_x)
        available.remove(best_x)
        
    best_A_mask = A_mask
    best_score = count_3aps(A_mask)
    
    current_A_mask = A_mask
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 2:
        move_type = random.choice(['add', 'remove', 'swap'])
        new_A_mask = current_A_mask
        valid_move = True
        new_score = -1
        
        if move_type == 'add':
            not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
            if not not_in_A:
                continue
            x = random.choice(not_in_A)
            if is_valid_add(x, current_A_mask):
                new_A_mask = current_A_mask | (1 << x)
                new_score = count_3aps(new_A_mask)
            else:
                valid_move = False
                
        elif move_type == 'remove':
            in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
            if not in_A:
                continue
            x = random.choice(in_A)
            new_A_mask = current_A_mask & ~(1 << x)
            new_score = count_3aps(new_A_mask)
            
        elif move_type == 'swap':
            in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
            not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
            if not in_A or not not_in_A:
                continue
            x = random.choice(in_A)
            y = random.choice(not_in_A)
            temp_mask = current_A_mask & ~(1 << x)
            if is_valid_add(y, temp_mask):
                new_A_mask = temp_mask | (1 << y)
                new_score = count_3aps(new_A_mask)
            else:
                valid_move = False
                
        if valid_move:
            if new_score > current_score:
                current_A_mask = new_A_mask
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_A_mask = current_A_mask
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    for _ in range(5):
                        in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
                        if in_A:
                            x = random.choice(in_A)
                            current_A_mask &= ~(1 << x)
                    for _ in range(5):
                        not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
                        if not not_in_A: break
                        best_x = -1
                        best_gain = -1
                        for x in not_in_A:
                            if is_valid_add(x, current_A_mask):
                                gain = get_3ap_gain(x, current_A_mask)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_x = x
                        if best_x != -1:
                            current_A_mask |= (1 << best_x)
                    current_score = count_3aps(current_A_mask)
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_A_mask = current_A_mask
                        
    result = []
    for i in range(N):
        if best_A_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)
# EVOLVE-BLOCK-END
