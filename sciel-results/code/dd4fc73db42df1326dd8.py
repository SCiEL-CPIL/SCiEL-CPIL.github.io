import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 192
    
    # Precompute residue masks
    # ResidueMasks[a][r] is a bitmask of integers v in [1, n] such that v % a == r
    ResidueMasks = [[0] * a for a in range(n + 1)]
    for a in range(1, n + 1):
        for v in range(1, n + 1):
            r = v % a
            ResidueMasks[a][r] |= (1 << (v - 1))
    
    def can_add(S, S_mask, x):
        """
        Checks if x can be added to set S (represented by list S and bitmask S_mask).
        Returns True if valid.
        """
        # 1. Check if x divides the sum of any 3 elements in S
        if len(S) >= 3:
            # For each a in S, check if there are b, c in S\{a} such that b+c = -a mod x
            for a in S:
                r_a = a % x
                target_bc = (-r_a) % x
                # Mask of S excluding a
                S_excl_a_mask = S_mask & ~(1 << (a - 1))
                
                # Iterate b in S excluding a
                # We can iterate over bits in S_excl_a_mask
                m = S_excl_a_mask
                while m:
                    # Get lowest set bit
                    lsb = m & -m
                    b = lsb.bit_length()  # This gives index+1 if bit 0 is 1? 
                    # bit_length of 1 is 1. 1<<0 is 1. So bit index is 0. Value is 1.
                    # bit_length of 2 is 2. 1<<1 is 2. Bit index 1. Value 2.
                    # So value = bit_length.
                    b_val = b
                    m ^= lsb
                    
                    r_b = b_val % x
                    r_c = (target_bc - r_b) % x
                    
                    # Check if there exists c in S \ {a, b} such that c % x == r_c
                    # Mask for c: ResidueMasks[x][r_c]
                    # Exclude a and b from S_mask
                    S_excl_ab_mask = S_excl_a_mask & ~(1 << (b_val - 1))
                    
                    if ResidueMasks[x][r_c] & S_excl_ab_mask:
                        return False
                        
        # 2. Check if any a in S divides x + b + c for b, c in S \ {a}
        for a in S:
            if a == 1:
                # 1 divides any sum. If there are >= 3 other elements, it's invalid.
                # Other elements are S \ {1} U {x}.
                # Size of others = len(S) - (1 if 1 in S else 0) + 1
                # If 1 in S, others = len(S). If len(S) >= 3, invalid.
                # If 1 not in S, others = len(S) + 1. If len(S)+1 >= 3 -> len(S)>=2, invalid.
                # Wait, the condition is "No selected integer may divide...".
                # If we add x, the new set is S U {x}.
                # If a=1 is in S, it is in the new set.
                # It must not divide sum of any 3 distinct OTHER elements in new set.
                # Others = (S U {x}) \ {1}.
                # Size of Others = len(S U {x}) - 1 = len(S) + 1 - 1 = len(S).
                # So if len(S) >= 3, there are 3 others, and 1 divides their sum.
                # So if 1 in S and len(S) >= 3, adding x is invalid.
                if len(S) >= 3:
                    return False
                # If len(S) < 3, we don't have 3 others yet?
                # If len(S)=2, others size = 2. No 3 others. Safe.
                # If len(S)=1, others size = 1. Safe.
                # So continue to next a.
                continue
            
            target = (-x) % a
            # We need b, c in S \ {a} such that b+c = target mod a
            S_excl_a_mask = S_mask & ~(1 << (a - 1))
            
            m = S_excl_a_mask
            while m:
                lsb = m & -m
                b_val = lsb.bit_length()
                m ^= lsb
                
                r_b = b_val % a
                r_c = (target - r_b) % a
                
                # Check if c exists in S \ {a, b}
                S_excl_ab_mask = S_excl_a_mask & ~(1 << (b_val - 1))
                
                if ResidueMasks[a][r_c] & S_excl_ab_mask:
                    return False
                    
        return True

    best_set = []
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # 1. Greedy Descending
    S1 = []
    mask1 = 0
    for x in range(n, 0, -1):
        if can_add(S1, mask1, x):
            S1.append(x)
            mask1 |= (1 << (x - 1))
    update_best(S1)

    # 2. Greedy Ascending
    S2 = []
    mask2 = 0
    for x in range(1, n + 1):
        if can_add(S2, mask2, x):
            S2.append(x)
            mask2 |= (1 << (x - 1))
    update_best(S2)

    # 3. Random Greedy
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S3 = []
        mask3 = 0
        for x in order:
            if can_add(S3, mask3, x):
                S3.append(x)
                mask3 |= (1 << (x - 1))
        update_best(S3)

    # 4. Local Search on Best Set
    if best_set:
        current_set = best_set[:]
        current_mask = 0
        for x in current_set:
            current_mask |= (1 << (x - 1))
        unselected = set(range(1, n + 1)) - set(current_set)
        
        attempts = 0
        while time.time() < deadline and attempts < 20000:
            attempts += 1
            if not current_set or not unselected:
                break
            
            # Random Swap
            idx = random.randint(0, len(current_set) - 1)
            a = current_set[idx]
            b = random.choice(list(unselected))
            
            # Check if swap is valid
            # Remove a, add b
            S_prime = [x for x in current_set if x != a]
            mask_prime = current_mask & ~(1 << (a - 1))
            
            if can_add(S_prime, mask_prime, b):
                # Perform swap
                current_set[idx] = b
                current_mask = mask_prime | (1 << (b - 1))
                unselected.remove(b)
                unselected.add(a)
                
                # Try to add more after swap
                # Check if any unselected can be added now
                added = False
                # Shuffle unselected to avoid bias
                unselected_list = list(unselected)
                random.shuffle(unselected_list)
                for cand in unselected_list:
                    if can_add(current_set, current_mask, cand):
                        current_set.append(cand)
                        current_mask |= (1 << (cand - 1))
                        unselected.remove(cand)
                        update_best(current_set)
                        added = True
                        # Try to add one more?
                        unselected_list2 = list(unselected)
                        random.shuffle(unselected_list2)
                        for cand2 in unselected_list2:
                            if can_add(current_set, current_mask, cand2):
                                current_set.append(cand2)
                                current_mask |= (1 << (cand2 - 1))
                                unselected.remove(cand2)
                                update_best(current_set)
                                break
                        break

    return best_set
