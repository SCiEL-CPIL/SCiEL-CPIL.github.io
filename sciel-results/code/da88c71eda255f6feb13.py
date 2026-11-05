import time
import random
import numpy as np

N = 512

def _precompute_masks():
    """
    Precompute for each element x (0-indexed in [0, 511] representing 1..512):
    1. mask[x]: a 512-bit integer where bit i is set if i is in a 4-AP with x.
       Actually, we need to check if adding x creates a 4-AP.
       A 4-AP is (a, a+d, a+2d, a+3d).
       If we add x, we create a 4-AP if there exists a 4-AP containing x
       such that the other 3 elements are already in the set.
       
       Let's define conflict_mask[x] as a bitset of all elements that are in
       a 4-AP with x? No, that's not enough. We need to know if the *triple*
       of other elements is present.
       
       Better approach for bit-parallelism:
       For each x, precompute a list of "forbidden triples" T_x = { (y, z, w) }.
       But checking if a triple is in the set is not O(1) with a single bitset.
       
       Alternative: 
       We can reformulate the check.
       x conflicts with current set S if there exists a 4-AP (x, a, b, c) such that {a,b,c} subset S.
       This is equivalent to: for every 4-AP containing x, if the other 3 are in S, conflict.
       
       To make this fast, we can precompute for each x a list of masks M_x where each mask
       represents a specific 4-AP containing x, and we check if (S & mask) == mask.
       But storing lists of masks and iterating is slower than a single bit operation.
       
       Let's try a different bit-parallel trick.
       Is there a way to encode the "3-out-of-4" condition?
       
       Actually, the number of 4-APs is relatively small.
       Total 4-APs in [1, 512]:
       Sum_{d=1}^{170} (512 - 3d) approx 512*170 - 3*170^2/2 ~ 87000 - 43000 ~ 44000.
       Each element is in ~44000*4/512 ~ 340 4-APs.
       
       Iterating 340 masks per check is fast in Python if the masks are integers.
       (S & mask) == mask is very fast.
       
       So:
       1. Precompute all 4-APs.
       2. For each x, store a list of bitmasks, where each bitmask has exactly 3 bits set
          (the other 3 elements of the 4-AP).
       3. can_add(x, S_mask): return all((S_mask & m) != m for m in masks[x])
          Wait, if (S_mask & m) == m, it means all 3 are present. So conflict.
          So can_add is True if for all m in masks[x], (S_mask & m) != m.
    """
    masks = [[] for _ in range(N)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d <= N
    # 1-indexed in logic, 0-indexed in bits
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                # 0-indexed
                idx_a = a - 1
                idx_b = b - 1
                idx_c = c - 1
                idx_e = e - 1
                
                # For each element in the AP, store the mask of the other 3
                # Element a: others are b, c, e
                mask_a = (1 << idx_b) | (1 << idx_c) | (1 << idx_e)
                masks[idx_a].append(mask_a)
                
                # Element b: others are a, c, e
                mask_b = (1 << idx_a) | (1 << idx_c) | (1 << idx_e)
                masks[idx_b].append(mask_b)
                
                # Element c: others are a, b, e
                mask_c = (1 << idx_a) | (1 << idx_b) | (1 << idx_e)
                masks[idx_c].append(mask_c)
                
                # Element e: others are a, b, c
                mask_e = (1 << idx_a) | (1 << idx_b) | (1 << idx_c)
                masks[idx_e].append(mask_e)
    
    return masks

_MASKS = _precompute_masks()

def _can_add(x, S_mask):
    """
    Check if x can be added to set represented by S_mask.
    x is 0-indexed.
    """
    for m in _MASKS[x]:
        if (S_mask & m) == m:
            return False
    return True

def _construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # 1. Initial Greedy Construction
    # We want a large initial set.
    # Greedy order: elements with fewest conflicts first? 
    # Or just random?
    # Let's try a few greedy passes.
    
    best_mask = 0
    best_size = 0
    
    def greedy(order):
        s_mask = 0
        for x in order:
            if _can_add(x, s_mask):
                s_mask |= (1 << x)
        return s_mask
    
    # Pass 1: Natural order
    order = list(range(N))
    s = greedy(order)
    best_mask = s
    best_size = bin(s).count('1')
    
    # Pass 2: Reverse
    if time.time() < deadline:
        order = list(range(N-1, -1, -1))
        s = greedy(order)
        if bin(s).count('1') > best_size:
            best_mask = s
            best_size = bin(s).count('1')
            
    # Pass 3: Random
    for _ in range(10):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        s = greedy(order)
        if bin(s).count('1') > best_size:
            best_mask = s
            best_size = bin(s).count('1')
            
    # 2. Local Search / Hill Climbing with Bit-Parallel Checks
    # Try to add elements.
    # We can also try to remove one and add two (swap).
    
    def local_search(s_mask, max_iter=20000):
        s_mask = s_mask
        size = bin(s_mask).count('1')
        no_improve = 0
        max_no_improve = 1000
        
        # Precompute list of missing elements for faster access?
        # Or just iterate.
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Try to add a random missing element
            missing = [x for x in range(N) if not (s_mask & (1 << x))]
            if not missing:
                break
                
            x = random.choice(missing)
            if _can_add(x, s_mask):
                s_mask |= (1 << x)
                size += 1
                no_improve = 0
            else:
                no_improve += 1
                # Try a swap: remove y, add x
                # Only if no_improve is high
                if no_improve % 10 == 0:
                    # Try to find a y to remove such that x can be added
                    # We need to check if removing y resolves the conflict.
                    # The conflict is caused by some mask m in _MASKS[x] such that (s_mask & m) == m.
                    # If we remove y from m, then (s_mask & m) != m.
                    # So we just need to pick a y that is in one of the conflicting masks.
                    
                    conflicting_masks = []
                    for m in _MASKS[x]:
                        if (s_mask & m) == m:
                            conflicting_masks.append(m)
                            
                    if conflicting_masks:
                        # Pick a random conflicting mask
                        m = random.choice(conflicting_masks)
                        # Pick a random bit from m to remove
                        # m has exactly 3 bits set.
                        bits = []
                        mm = m
                        while mm:
                            lsb = mm & (-mm)
                            bits.append(lsb)
                            mm ^= lsb
                        y_bit = random.choice(bits)
                        
                        # Try removing y_bit and adding x
                        new_mask = (s_mask & ~y_bit) | (1 << x)
                        # We must ensure that removing y_bit doesn't break other constraints?
                        # No, removing an element never creates a new 4-AP.
                        # So new_mask is valid if x can be added to (s_mask - y_bit).
                        # Which is what we are doing.
                        # But we must also ensure that x is not already in s_mask (it isn't, it's missing).
                        # And y_bit is in s_mask (it is, because m subset s_mask).
                        
                        # So this move is always valid!
                        # Size remains the same.
                        # But it might allow adding *more* elements later.
                        # So we accept it.
                        s_mask = new_mask
                        no_improve = 0
                        # Note: size doesn't change, but we reset no_improve to allow exploration.
        
        return s_mask, bin(s_mask).count('1')
    
    if time.time() < deadline:
        s, size = local_search(best_mask)
        if size > best_size:
            best_mask = s
            best_size = size
            
    # 3. Branch and Bound for Exact/High-Quality Solution
    # We can try to improve the best set found so far using B&B.
    # However, B&B on 512 nodes is hard.
    # Instead, let's do a "repair" step: 
    # Identify elements that are "blocking" many additions.
    
    # Alternative: Multi-start Local Search
    for i in range(5):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        s = greedy(order)
        s, size = local_search(s, max_iter=10000)
        if size > best_size:
            best_mask = s
            best_size = size

    # Convert mask to list of 1-indexed integers
    result = []
    for x in range(N):
        if best_mask & (1 << x):
            result.append(x + 1)
            
    return result

def construct():
    return _construct()
