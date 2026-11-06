"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # Precompute weights as Fractions for exact arithmetic in bounds
    # Weights are 1/i for i in 1..n
    weights = [Fraction(0)] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = Fraction(1, i)
        
    # Precompute suffix sums of weights as Fractions
    # suffix_sums[i] = sum(weights[i..n])
    suffix_sums = [Fraction(0)] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[i]
        
    # Bitset representation for the current set
    # Python integers act as arbitrary-precision bitsets
    # Bit i (0-indexed) corresponds to element i+1
    # So element x is represented by bit (x-1)
    
    # Precompute for each element x, the list of pairs (y, z) such that {x, y, z} is a 3-AP
    # Since we add elements in increasing order, when adding x, we only need to check if
    # there exist y, z in current_set such that x is the largest element of the AP.
    # i.e., x = 2z - y with y < z < x.
    # This is equivalent to: for each z in current_set, if (2z - x) is in current_set and (2z - x) < z.
    
    # To speed up the check, we can precompute for each x, the set of "conflicting" pairs?
    # No, the check depends on the current set.
    # However, we can optimize the check by iterating over the elements in the current set.
    # Since the set size is likely small (max size of 3-AP free set in [1,256] is around 100-120?),
    # and we use bitsets, we can do:
    # 
    # Can add x?
    # For each z in current_set:
    #     y = 2*z - x
    #     if y < z and bit(y-1) is set in current_set:
    #         return False
    #
    # Iterating over set bits in a Python integer is efficient.
    
    def get_set_bits(bitset):
        # Yield indices of set bits
        while bitset:
            lsb = bitset & -bitset
            idx = lsb.bit_length() - 1
            yield idx
            bitset ^= lsb
            
    # Alternatively, since n=256, we can just iterate 0..255 and check bits?
    # No, iterating set bits is better.
    
    # Let's refine the check function using bit operations.
    # To check if x can be added to S (bitset):
    # We need to check if there exists z in S such that y = 2z - x is in S and y < z.
    # This is equivalent to checking if there exists z in S such that (2z - x) is in S.
    # Note: if y = 2z - x, then y < z iff x > z.
    # So we only need to check z < x.
    # Actually, if y, z, x form an AP and x is the largest, then z < x.
    # So:
    # For each z in S where z < x:
    #     y = 2*z - x
    #     if y > 0 and bit(y-1) is set in S:
    #         return False
    
    def can_add(x, bitset):
        # x is 1-based
        # We check z in S such that z < x
        # We can iterate over set bits in bitset & ((1 << (x-1)) - 1)
        # This mask includes bits 0..x-2, i.e., elements 1..x-1
        mask = (1 << (x - 1)) - 1
        candidates = bitset & mask
        
        while candidates:
            lsb = candidates & -candidates
            z_idx = lsb.bit_length() - 1
            z = z_idx + 1  # z is 1-based
            
            y = 2 * z - x
            if y < 1:
                break # z is increasing? No, lsb gives smallest index first.
                # If y < 1, then for larger z, y will be even smaller?
                # y = 2z - x. As z increases, y increases.
                # So if y < 1 for the smallest z, it might be valid for larger z.
                # Wait, lsb gives the smallest z.
                # If y < 1, we just skip this z.
                pass
            
            if y >= 1 and y < z:
                # Check if y is in bitset
                y_bit = 1 << (y - 1)
                if bitset & y_bit:
                    return False
            
            candidates ^= lsb
            
        return True

    best_sum = Fraction(0)
    best_bitset = 0
    
    # Greedy Initialization
    # We want a good initial solution to prune effectively.
    greedy_bitset = 0
    greedy_sum = Fraction(0)
    
    for x in range(1, n + 1):
        if can_add(x, greedy_bitset):
            greedy_bitset |= (1 << (x - 1))
            greedy_sum += weights[x]
            
    best_sum = greedy_sum
    best_bitset = greedy_bitset
    
    # B&B Search
    # State: idx (next element to consider, 1-based), current_sum, current_bitset
    # We process elements 1..n.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def solve(idx, current_sum, current_bitset):
        nonlocal best_sum, best_bitset
        
        # Time check
        if time.time() > deadline:
            return
            
        if idx > n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_bitset = current_bitset
            return
            
        # Upper Bound: current_sum + suffix_sums[idx]
        # If this is <= best_sum, prune
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Option 1: Include idx
        # Check if idx can be added
        if can_add(idx, current_bitset):
            new_bitset = current_bitset | (1 << (idx - 1))
            new_sum = current_sum + weights[idx]
            solve(idx + 1, new_sum, new_bitset)
            
        # Option 2: Exclude idx
        # Even if we exclude, we might still beat best_sum
        solve(idx + 1, current_sum, current_bitset)
        
    # Start B&B
    # To make it faster, we can order the branching.
    # Usually, including high-weight items first is good.
    # Here we process 1..n, so 1 is first.
    
    solve(1, Fraction(0), 0)
    
    # Convert best_bitset to list of ints
    result_set = []
    b = best_bitset
    while b:
        lsb = b & -b
        idx = lsb.bit_length() - 1
        result_set.append(idx + 1)
        b ^= lsb
        
    # Ensure non-empty
    if not result_set:
        return [1]
        
    return result_set

# EVOLVE-BLOCK-END
