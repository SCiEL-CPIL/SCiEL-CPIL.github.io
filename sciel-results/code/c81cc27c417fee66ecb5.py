import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute for each divisor d, the set of residues mod period
    # Actually, since period is the LCM, each divisor d divides period.
    # The multiples of d in any interval of length period cover all residues
    # that are multiples of d mod period.
    # There are period/d such residues.
    
    # We want to find a window [L, L+W] of minimal width W
    # such that we can assign distinct multiples.
    
    # Strategy:
    # Iterate W from 9 upwards.
    # For a fixed W, we need to check if there exists an L such that
    # the system of constraints is satisfiable.
    
    # Instead of iterating L from 1 to period (which is ~30k),
    # we can exploit the periodicity.
    # The pattern of available multiples for a given window width W
    # shifts as L changes.
    
    # However, checking all L is O(period * n * (W/d_avg)) which might be slow
    # if done naively for large W. But W is expected to be small.
    
    # Let's estimate the cost.
    # period = 30030.
    # For each L, we build the list of available multiples.
    # Total work per L is sum(W/d_i) for i in 1..n.
    # If W is ~20, sum(20/d_i) is roughly 20 * sum(1/d_i).
    # sum(1/d_i) = 1/6+1/10+...+1/39 ~ 0.16+0.1+0.07+0.066+0.047+0.045+0.038+0.03+0.028+0.025 ~ 0.6
    # So ~12 operations per L for collection.
    # Then backtracking.
    # Total operations ~ 30030 * 12 * (backtrack cost).
    # Backtrack cost depends on the number of choices.
    
    # Let's try to optimize the inner loop.
    # We can precompute the "offset" of the first multiple for each divisor.
    
    # Actually, a better approach for fixed W:
    # We are looking for a matching in a bipartite graph.
    # Left side: divisors. Right side: integers in [L, L+W].
    # Edge (d, x) exists if d|x.
    
    # We can iterate L. To make it faster, we can skip L values that are "equivalent".
    # But since the constraints are "divisible by d", the structure changes
    # whenever L crosses a multiple of any d.
    # The "events" happen at multiples of d_i.
    # There are sum(period/d_i) such events.
    # sum(30030/d_i) = 5005 + 3003 + 2145 + 2002 + 1430 + 1365 + 1155 + 910 + 858 + 770
    # = 20643 events.
    # Between events, the set of available multiples for each divisor is constant?
    # No, the values change, but the *count* and *relative positions* might be simpler.
    # Actually, if no new multiple enters or leaves the window, the *set* of available
    # values for each divisor changes by shifting? No, the values are absolute.
    
    # Let's stick to iterating L but optimize the backtracking.
    # Use bitmasks for used values if W is small enough (W < 64).
    # If W is small, the number of distinct values in the window is W+1.
    # We can map each value x in [L, L+W] to an index x-L in [0, W].
    # Then we use a bitmask of size W+1.
    
    # This will be much faster than a set.
    
    best_result = None
    
    # We'll search W from 9 up to a reasonable limit, say 100.
    # If we find a solution for W, we return it immediately since we want min W.
    
    for W in range(9, 200):
        if time.time() > deadline:
            break
            
        # We only need to check L in a reduced range?
        # The pattern repeats every period.
        # But maybe we can restrict L further?
        # No, any L in [1, period] is a candidate.
        
        # Optimization: Precompute for each divisor d, the list of multiples
        # in a large range, or just compute on the fly.
        
        # Let's try iterating L.
        # To speed up, we can break early if a divisor has no multiples in [L, L+W].
        
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
            
            # Check feasibility and solve
            # Map values in [L, L+W] to indices 0..W
            # available[i] is a bitmask of indices available for divisor i
            
            available_masks = []
            feasible = True
            
            for d in divisors:
                mask = 0
                # First multiple of d >= L
                k = (L + d - 1) // d
                m = k * d
                while m <= L + W:
                    idx = m - L
                    mask |= (1 << idx)
                    m += d
                
                if mask == 0:
                    feasible = False
                    break
                
                available_masks.append(mask)
            
            if not feasible:
                continue
            
            # Solve matching using bitmask DP or backtracking
            # n=10, W is small (likely < 30).
            # Backtracking with MRV and bitmask.
            
            # Order divisors by number of available bits (popcount)
            # popcount of mask
            def popcount(x):
                c = 0
                while x:
                    c += 1
                    x &= x - 1
                return c
            
            order = sorted(range(n), key=lambda i: popcount(available_masks[i]))
            
            used_mask = 0
            result = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = order[idx]
                mask = available_masks[i]
                
                # Iterate through set bits in mask
                # We want to try bits that are not in used_mask
                # available = mask & ~used_mask
                
                avail = mask & ~used_mask
                while avail:
                    # Get lowest set bit
                    bit = avail & -avail
                    avail -= bit
                    
                    # Assign this bit
                    # The value is L + index
                    index = bit.bit_length() - 1
                    result[i] = L + index
                    
                    used_mask_new = used_mask | bit
                    if backtrack(idx + 1):
                        return True
                    
                    avail &= avail - 1 # This is redundant if we just do avail -= bit, but good for clarity if we were iterating differently. Actually avail -= bit is enough.
                
                return False
            
            # Correcting the loop:
            def backtrack2(idx):
                if idx == n:
                    return True
                
                i = order[idx]
                mask = available_masks[i]
                avail = mask & ~used_mask
                
                while avail:
                    bit = avail & -avail
                    avail -= bit
                    
                    index = bit.bit_length() - 1
                    result[i] = L + index
                    
                    # Recurse
                    # We need to modify used_mask. Since it's an integer, we pass it or use nonlocal.
                    # Using nonlocal is faster than passing in Python recursion?
                    # Let's use a list for used_mask to avoid global/nonlocal overhead?
                    # Or just pass it as argument.
                    
                    if backtrack2_inner(idx + 1, used_mask | bit):
                        return True
                
                return False
            
            def backtrack2_inner(idx, current_used):
                if idx == n:
                    return True
                
                i = order[idx]
                mask = available_masks[i]
                avail = mask & ~current_used
                
                while avail:
                    bit = avail & -avail
                    avail -= bit
                    
                    index = bit.bit_length() - 1
                    result[i] = L + index
                    
                    if backtrack2_inner(idx + 1, current_used | bit):
                        return True
                
                return False
            
            if backtrack2_inner(0, 0):
                # Found a solution
                # Verify
                if len(set(result)) != n:
                    continue
                if any(x < 1 or x > 1000000 for x in result):
                    continue
                valid = True
                for i in range(n):
                    if result[i] % divisors[i] != 0:
                        valid = False
                        break
                if valid:
                    return [int(x) for x in result]
        
    # Fallback
    return divisors[:]

# To make it faster, we should avoid re-defining functions inside the loop if possible,
# but Python closures are okay.
# The main bottleneck is the L loop.
# Let's see if we can prune L.
# If L is such that for some d, there are no multiples in [L, L+W], skip.
# This is already done.

# One more optimization:
# We can precompute the "next multiple" pointers.
# But the bitmask approach is quite fast.

# Let's try to run this mentally for W=9.
# period = 30030.
# 30030 iterations.
# Each iteration:
# 1. Compute masks (10 divisors, small loops).
# 2. Backtracking.
# If W=9 is not feasible, it will quickly fail for most L?
# Probably.

# What if W is large?
# If W=100, the bitmask is 101 bits wide, which fits in Python int easily.
# The number of bits set in the mask for divisor d is ~ W/d + 1.
# For d=6, ~17 bits.
# Backtracking depth 10.
# Branching factor is the number of available bits.
# Worst case is high, but MRV helps.

# I will add a small optimization:
# Sort divisors by d ascending? No, MRV is dynamic.
# But we can pre-sort the divisors list itself?
# No, the result must be in the original order.
# The 'order' variable handles the search order.

# Let's ensure the fallback is robust.
# The current fallback returns divisors, which is valid.

# One potential issue: recursion depth.
# n=10, so depth is 10. No issue.

# Let's refine the bitmask iteration.
# bit = avail & -avail
# index = bit.bit_length() - 1
# This is efficient.

# Final check on constraints:
# "ten DISTINCT positive Python integers <=1000000"
# "aligned with the pinned input"
# "Minimize max-min"

# The code seems correct.

# To be safe against time limits, I'll add a check every 1000 L iterations.
# The current check is every L iteration, which is fine (time.time() is fast enough?
# Actually, calling time.time() 30000 times is negligible).

# Let's produce the final code.

import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)
    
    start_time = time.time()
    deadline = start_time + 85
    
    # Pre-compute popcount helper?
    # Python 3.10+ has int.bit_count().
    # Let's assume standard env supports it or use fallback.
    # The prompt says "Python standard library", usually implies recent versions.
    # I'll use a helper to be safe.
    
    def popcount(x):
        return bin(x).count("1")
    
    # Search for minimal W
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
            
            available_masks = []
            feasible = True
            
            for d in divisors:
                mask = 0
                k = (L + d - 1) // d
                m = k * d
                while m <= L + W:
                    idx = m - L
                    mask |= (1 << idx)
                    m += d
                
                if mask == 0:
                    feasible = False
                    break
                
                available_masks.append(mask)
            
            if not feasible:
                continue
            
            # MRV ordering
            order = sorted(range(n), key=lambda i: popcount(available_masks[i]))
            
            result = [0] * n
            
            def backtrack(idx, used_mask):
                if idx == n:
                    return True
                
                i = order[idx]
                mask = available_masks[i]
                avail = mask & ~used_mask
                
                while avail:
                    bit = avail & -avail
                    avail -= bit
                    
                    index = bit.bit_length() - 1
                    result[i] = L + index
                    
                    if backtrack(idx + 1, used_mask | bit):
                        return True
                
                return False
            
            if backtrack(0, 0):
                # Verification
                if len(set(result)) != n:
                    continue
                if any(x < 1 or x > 1000000 for x in result):
                    continue
                valid = True
                for i in range(n):
                    if result[i] % divisors[i] != 0:
                        valid = False
                        break
                if valid:
                    return [int(x) for x in result]
        
    return divisors[:]
