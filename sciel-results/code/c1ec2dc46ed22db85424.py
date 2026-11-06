import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We aim to finish by 85s.
    deadline = 85.0
    
    n = 512
    k = 4
    
    # 1. Precomputation
    # We represent the set S as a Python integer bitmask.
    # Bit i is 1 if i is in S.
    
    # Precompute conflict masks for each element x.
    # conflict_masks[x] is a list of bitmasks M.
    # If (S & M) == M, then x cannot be added to S without creating a 4-AP.
    conflict_masks = [[] for _ in range(n + 1)]
    
    # Generate all 4-term APs (a, a+d, a+2d, a+3d)
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # For each element in the AP, the "conflict mask" is the bitmask of the other 3.
            # If those 3 are present, the current element is blocked.
            
            # Element a is blocked if b, c, e are in S
            conflict_masks[a].append((1 << b) | (1 << c) | (1 << e))
            
            # Element b is blocked if a, c, e are in S
            conflict_masks[b].append((1 << a) | (1 << c) | (1 << e))
            
            # Element c is blocked if a, b, e are in S
            conflict_masks[c].append((1 << a) | (1 << b) | (1 << e))
            
            # Element e is blocked if a, b, c are in S
            conflict_masks[e].append((1 << a) | (1 << b) | (1 << c))
    
    def is_valid_add(x, S_mask):
        """Check if adding x to S_mask creates a 4-AP."""
        for m in conflict_masks[x]:
            if (S_mask & m) == m:
                return False
        return True

    def popcount(x):
        return bin(x).count('1')
        
    # 2. Initial Construction: Ruzsa's Construction
    # Ruzsa's construction for 4-AP-free sets uses a base-3 representation.
    # Specifically, let S be the set of numbers in [1, n] whose base-3 representation
    # contains only digits 0 and 1.
    # This set is 3-AP free? No, it's 2-AP free in some contexts, but let's check.
    # Actually, the standard Ruzsa construction for k-AP free sets involves sumsets.
    # A simpler, well-known construction for 4-AP free sets is based on the Cantor set (digits 0,1 in base 3).
    # Let's verify if the Cantor set (digits 0,1 in base 3) is 4-AP free.
    # If x, x+d, x+2d, x+3d are all in the Cantor set, their base-3 digits are 0 or 1.
    # This is a known property: The set of numbers with no digit 2 in base 3 is 3-AP free? 
    # Actually, the set of numbers with digits 0,1 in base 3 is NOT 4-AP free necessarily?
    # Wait, the classic result is that the set of numbers with base-3 digits 0 and 1 is 3-AP free?
    # No, 0, 1, 3, 4... 1 (1), 3 (10), 4 (11). 1, 3, 4 is not an AP.
    # Let's just use a greedy approach on the "Ruzsa-like" structure or just a strong greedy.
    # Actually, a very strong deterministic construction is to take numbers with base-4 digits 0,1?
    # Let's stick to a robust Greedy + Local Search, but optimize the Greedy part.
    
    # Strategy:
    # 1. Start with a "Ruzsa-like" set: Numbers in [1, n] that have no digit 2 in base 3.
    #    This set is known to be 3-AP free? 
    #    Reference: The set of integers with no digit 2 in base 3 is 3-AP free.
    #    Is it 4-AP free? If it's 3-AP free, it is automatically 4-AP free.
    #    Let's verify: If a set has no 3-term AP, it has no 4-term AP.
    #    So this is a valid starting set.
    #    Size: Approximately n * (2/3)^log_3(n) ? No, the density is 0.
    #    Wait, the Cantor set has measure 0. The number of such elements in [1, n] is roughly n^log_3(2) ~ n^0.63.
    #    For n=512, 512^0.63 ~ 512^(0.63). 
    #    512 = 2^9 = 3^4.26.
    #    Count is approx 2^9 = 512? No.
    #    Let's just generate it.
    
    S_mask = 0
    for i in range(1, n + 1):
        valid = True
        temp = i
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            S_mask |= (1 << i)
            
    # This set is 3-AP free, hence 4-AP free.
    # Let's check its size.
    # If it's too small, we can try a different base or a hybrid.
    # For n=512, the count of numbers with digits 0,1 in base 3:
    # Max base-3 number is 200000 (511).
    # It's a small set. Let's say ~100 elements.
    # We want ~200+.
    
    # Alternative: Use a greedy algorithm but with a smarter order.
    # Order elements by "constraint degree" or just random/greedy.
    # The current program uses random greedy. Let's try a deterministic greedy 
    # that prioritizes elements with fewer conflicts?
    
    # Actually, the most effective strategy for 4-AP free sets in this range is often 
    # to use a "greedy + local search" but with a better initial seed.
    # Let's try a "Base-4" digit restriction? 
    # Numbers with digits 0,1 in base 4?
    # 4-AP free?
    
    # Let's stick to the proven method: 
    # 1. Generate a set using the "Cantor" (base 3, digits 0,1). 
    # 2. Use local search to add more elements.
    
    # However, the Cantor set might be too sparse.
    # Let's try a different deterministic construction:
    # Take all numbers i such that i % 3 != 0? No, that has 3-APs (1, 4, 7).
    
    # Let's use the "Greedy" approach but optimized.
    # We will run multiple random greedy passes and pick the best, then local search.
    
    best_mask = 0
    best_size = 0
    
    # Phase 1: Multiple Greedy Restarts
    # We have ~80 seconds. 
    # A single greedy pass is very fast.
    # Let's do 50 greedy passes.
    
    for _ in range(50):
        S_mask = 0
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_add(x, S_mask):
                S_mask |= (1 << x)
        
        size = popcount(S_mask)
        if size > best_size:
            best_size = size
            best_mask = S_mask
            
        if time.time() > deadline - 10:
            break
            
    # Phase 2: Local Search from the best greedy result
    cur_mask = best_mask
    cur_size = best_size
    
    # Helper to get unselected elements
    def get_unselected(S_mask):
        return [i for i in range(1, n + 1) if not (S_mask & (1 << i))]
        
    def get_selected(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]
    
    # Local Search Loop
    # We will try to add elements. If stuck, try swaps.
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try to add any valid element
        unselected = get_unselected(cur_mask)
        random.shuffle(unselected)
        
        for x in unselected:
            if is_valid_add(x, cur_mask):
                cur_mask |= (1 << x)
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
                improved = True
                break
        
        if improved:
            continue
            
        # If no additions, try 1-for-2 swaps
        selected = get_selected(cur_mask)
        if not selected:
            break
            
        # Try a limited number of random swaps
        # We pick a random y to remove, and try to add 2 random x's.
        # To be efficient, we only try a few times per iteration.
        
        found_swap = False
        # Limit the number of swap attempts per iteration to avoid spending too much time
        # if we are stuck.
        swap_attempts = 0
        while swap_attempts < 20 and time.time() < deadline - 2:
            swap_attempts += 1
            y = random.choice(selected)
            temp_mask = cur_mask & ~(1 << y)
            
            # Get unselected elements in temp_mask
            # Note: y is now unselected, but we want to add elements NOT in cur_mask (excluding y? No, we removed y).
            # We want to add elements from the original unselected set.
            unselected_in_temp = [i for i in range(1, n + 1) if not (temp_mask & (1 << i)) and i != y]
            
            if len(unselected_in_temp) < 2:
                continue
                
            # Pick 2 random elements
            x1, x2 = random.sample(unselected_in_temp, 2)
            
            # Check if x1 can be added
            if not is_valid_add(x1, temp_mask):
                continue
                
            temp_mask1 = temp_mask | (1 << x1)
            
            # Check if x2 can be added
            if not is_valid_add(x2, temp_mask1):
                continue
                
            # Success
            cur_mask = temp_mask1 | (1 << x2)
            cur_size += 1
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
            improved = True
            found_swap = True
            break
            
        if improved:
            continue
            
        # If stuck, try a "kick" (remove a few random elements) to escape local optimum
        # Or try 1-for-1 swaps (neutral) to explore.
        
        # Let's try 1-for-1 swaps first (neutral moves)
        if selected:
            y = random.choice(selected)
            temp_mask = cur_mask & ~(1 << y)
            unselected_in_temp = [i for i in range(1, n + 1) if not (temp_mask & (1 << i)) and i != y]
            
            if unselected_in_temp:
                # Try a few random neutral swaps
                for _ in range(5):
                    x = random.choice(unselected_in_temp)
                    if is_valid_add(x, temp_mask):
                        cur_mask = temp_mask | (1 << x)
                        # Size unchanged
                        break
        
        # If still stuck, kick
        if selected:
            kick_size = min(3, len(selected))
            kicks = random.sample(selected, kick_size)
            for k in kicks:
                cur_mask &= ~(1 << k)
            cur_size -= kick_size

    # Final Verification
    result = [i for i in range(1, n + 1) if (best_mask & (1 << i))]
    
    # Double check validity
    S_set = set(result)
    valid = True
    for a in S_set:
        for d in range(1, (n - a) // 3 + 1):
            if (a+d) in S_set and (a+2*d) in S_set and (a+3*d) in S_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Greedy
        S_mask = 0
        for x in range(1, n+1):
            if is_valid_add(x, S_mask):
                S_mask |= (1 << x)
        result = [i for i in range(1, n + 1) if (S_mask & (1 << i))]
        
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
