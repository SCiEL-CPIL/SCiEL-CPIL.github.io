"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s kill (evaluator might have overhead)
    
    N = 256
    # Precompute reciprocal weights as floats for fast comparison
    # and Fractions for final exact calculation
    recip_float = [0.0] * (N + 1)
    recip_frac = [Fraction(0, 1)] * (N + 1)
    for i in range(1, N + 1):
        recip_float[i] = 1.0 / i
        recip_frac[i] = Fraction(1, i)
        
    # Bitmask helpers
    def bit_set(mask, i):
        return (mask >> (i - 1)) & 1
    
    def set_bit(mask, i):
        return mask | (1 << (i - 1))
        
    def clear_bit(mask, i):
        return mask & ~(1 << (i - 1))

    # Precompute conflict pairs: for each x, store all y such that x and y cannot coexist
    # x and y conflict if there exists z such that {x, y, z} forms a 3-AP
    # This means: z = 2x - y (x is middle), z = 2y - x (y is middle), or x+y is even and z=(x+y)/2 (z is middle)
    # Actually, three distinct numbers a, b, c form a 3-AP iff one is the average of the other two.
    # So x and y conflict if:
    #   - 2x - y is in [1,N] and 2x-y != x and 2x-y != y (i.e., x is middle of AP containing y)
    #   - 2y - x is in [1,N] and 2y-x != x and 2y-x != y (i.e., y is middle of AP containing x)
    #   - (x+y) is even and (x+y)/2 is in [1,N] and (x+y)/2 != x and (x+y)/2 != y (i.e., some z is middle)
    
    # For efficient O(1) validity check: precompute for each x a bitmask of all y that conflict with x
    conflict_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x == y:
                continue
            # Check if x and y can be in the same 3-AP with some third element z
            # Case 1: x is middle: z = 2x - y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                conflict_mask[x] |= (1 << (y - 1))
                continue
            # Case 2: y is middle: z = 2y - x
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                conflict_mask[x] |= (1 << (y - 1))
                continue
            # Case 3: z is middle: z = (x+y)/2, requires x+y even
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= N and z != x and z != y:
                    conflict_mask[x] |= (1 << (y - 1))
    
    # Symmetrize: if x conflicts with y, then y conflicts with x
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if bit_set(conflict_mask[x], y) and not bit_set(conflict_mask[y], x):
                conflict_mask[y] |= (1 << (x - 1))
    
    def is_valid_addition_fast(mask, x):
        """
        Check if x can be added to the set represented by mask using precomputed conflicts.
        x is 1-indexed.
        """
        if bit_set(mask, x):
            return False
        
        # Check if any element in mask conflicts with x
        # conflict_mask[x] has bit y set if x and y conflict
        # We need to check if mask & conflict_mask[x] == 0
        return (mask & conflict_mask[x]) == 0

    def get_weight(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length()
            total += recip_float[x]
            m &= m - 1
        return total

    def exact_weight(mask):
        total = Fraction(0, 1)
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length()
            total += recip_frac[x]
            m &= m - 1
        return total

    # 1. Greedy Initialization
    # Sort by weight descending (smaller numbers first)
    order = list(range(1, N + 1))
    order.sort(key=lambda x: -recip_float[x])
    
    best_mask = 0
    best_weight = 0.0
    
    # Try a few different greedy orders to find a good starting point
    # Order 1: Descending weight
    mask = 0
    for x in order:
        if is_valid_addition_fast(mask, x):
            mask = set_bit(mask, x)
    w = get_weight(mask)
    if w > best_weight:
        best_weight = w
        best_mask = mask

    # 2. Local Search (Swaps)
    # We will perform iterations of:
    # - Try adding any valid element
    # - Try swapping any element u in S with any element v not in S
    # - Try 2-for-1 swaps (remove u1, u2, add v)
    # - Try 1-for-2 swaps (remove u, add v1, v2) -- usually less critical but possible
    
    max_iterations = 50000
    current_mask = best_mask
    current_weight = best_weight
    
    for _ in range(max_iterations):
        if time.time() > DEADLINE:
            break
            
        improved = False
        
        # A. Try Additions
        # Iterate through candidates not in current_mask
        # To speed up, we can just iterate 1..N
        for x in range(1, N + 1):
            if bit_set(current_mask, x):
                continue
            if is_valid_addition_fast(current_mask, x):
                new_mask = set_bit(current_mask, x)
                new_w = current_weight + recip_float[x]
                if new_w > current_weight + 1e-12:
                    current_mask = new_mask
                    current_weight = new_w
                    if current_weight > best_weight:
                        best_weight = current_weight
                        best_mask = current_mask
                    improved = True
                    # Restart additions to find more
                    break 
        
        if improved:
            continue # Restart the loop to re-evaluate additions
            
        # B. Try Swaps (1-for-1)
        # For each u in S, for each v not in S:
        # Check if S \ {u} U {v} is valid.
        # Optimization: Only try v where recip_float[v] > recip_float[u] ?
        # Not strictly, because removing u might allow adding v AND keep other elements,
        # but usually we want to increase weight.
        # However, sometimes removing a 'heavy' u (small number) allows adding 'lighter' v's?
        # No, weight is 1/x. Small x is heavy.
        # We want to maximize sum.
        # So we generally want to swap u for v if 1/v > 1/u, i.e., v < u.
        # But we must check validity.
        
        found_swap = False
        # Get list of elements in current set
        elements_in_set = []
        m = current_mask
        while m:
            lsb = m & -m
            elements_in_set.append(lsb.bit_length())
            m &= m - 1
            
        for u in elements_in_set:
            # Try replacing u with v
            # v should ideally be smaller than u to gain weight, 
            # but let's check all v < u first for efficiency, then maybe others if time permits?
            # Actually, if v > u, weight decreases. Unless it allows adding MORE elements?
            # Local search 1-for-1 only replaces one. So weight must increase.
            # So we only care about v < u.
            for v in range(1, u):
                if bit_set(current_mask, v):
                    continue
                
                # Check validity of S' = S \ {u} U {v}
                # We can create the new mask and check
                temp_mask = clear_bit(current_mask, u)
                if is_valid_addition_fast(temp_mask, v):
                    new_mask = set_bit(temp_mask, v)
                    new_w = current_weight - recip_float[u] + recip_float[v]
                    if new_w > current_weight + 1e-12:
                        current_mask = new_mask
                        current_weight = new_w
                        if current_weight > best_weight:
                            best_weight = current_weight
                            best_mask = current_mask
                        found_swap = True
                        break # Restart swap search
            if found_swap:
                break
        
        if found_swap:
            continue
            
        # C. Try 2-for-1 Swaps (Remove u1, u2, Add v)
        # This is more expensive. O(N^3) roughly.
        # We can limit this to a few iterations or only if time is left.
        # Given N=256, N^3 is ~16M. Might be too slow in Python if done naively.
        # Let's try a limited version:
        # Iterate u1, u2 in S.
        # For each pair, calculate temp_mask = S \ {u1, u2}.
        # Calculate weight_loss = w[u1] + w[u2].
        # Find best v not in S such that is_valid_addition(temp_mask, v) and w[v] > weight_loss.
        
        if time.time() < DEADLINE - 5: # Only do this if we have time
            elements_in_set = []
            m = current_mask
            while m:
                lsb = m & -m
                elements_in_set.append(lsb.bit_length())
                m &= m - 1
                
            found_2_1 = False
            # To avoid O(N^3), we can pre-filter or just break early.
            # Let's iterate u1, u2.
            for i in range(len(elements_in_set)):
                if time.time() > DEADLINE - 2:
                    break
                u1 = elements_in_set[i]
                for j in range(i + 1, len(elements_in_set)):
                    u2 = elements_in_set[j]
                    
                    temp_mask = clear_bit(clear_bit(current_mask, u1), u2)
                    weight_loss = recip_float[u1] + recip_float[u2]
                    
                    # Look for best v
                    best_v = -1
                    best_v_w = 0.0
                    for v in range(1, N + 1):
                        if bit_set(current_mask, v):
                            continue
                        # Optimization: if w[v] <= weight_loss, no point
                        if recip_float[v] <= weight_loss:
                            continue
                        if is_valid_addition_fast(temp_mask, v):
                            if recip_float[v] > best_v_w:
                                best_v_w = recip_float[v]
                                best_v = v
                                
                    if best_v != -1:
                        new_mask = set_bit(temp_mask, best_v)
                        new_w = current_weight - weight_loss + best_v_w
                        if new_w > current_weight + 1e-12:
                            current_mask = new_mask
                            current_weight = new_w
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_mask = current_mask
                            found_2_1 = True
                            break
                if found_2_1:
                    break
                    
            if found_2_1:
                continue

        # D. Try 1-for-2 Swaps (Remove u, Add v1, v2)
        # Similar logic.
        # If no improvement found in A, B, C, we are at a local optimum for these moves.
        # We could break, or try random restarts.
        # Given the time budget, breaking and returning the best is safer than random restarts 
        # which might lose time. However, a simple random restart of the greedy phase 
        # with perturbed weights might find a better basin.
        
        # Let's do one random restart if we have > 10s left.
        if time.time() < DEADLINE - 10:
             # Perturb weights slightly
             perturbed_weights = [recip_float[i] * (1 + random.uniform(-0.1, 0.1)) for i in range(N+1)]
             order2 = list(range(1, N+1))
             order2.sort(key=lambda x: -perturbed_weights[x])
             
             mask2 = 0
             for x in order2:
                 if is_valid_addition_fast(mask2, x):
                     mask2 = set_bit(mask2, x)
             w2 = get_weight(mask2)
             if w2 > best_weight:
                 best_weight = w2
                 best_mask = mask2
                 current_mask = mask2
                 current_weight = w2
             # Continue local search from new best if improved
             # The loop will restart additions/swaps
        else:
            # No time for restarts, break if no improvement
            # We've tried Add, Swap, 2-for-1. 
            # If none worked, we are likely stuck.
            break

    # Final Result
    # Extract numbers from best_mask
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        result.append(lsb.bit_length())
        m &= m - 1
        
    # Sort to be safe, though order doesn't matter for the set
    result.sort()
    
    return result

# EVOLVE-BLOCK-END
