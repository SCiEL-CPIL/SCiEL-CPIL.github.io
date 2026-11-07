# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import deque

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _squarefree_part(n):
    """Compute the squarefree part of n using trial division."""
    if n <= 1:
        return n
    sf = 1
    p = 2
    while p * p <= n:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt ^= 1
        if cnt:
            sf *= p
        p += 1
    if n > 1:
        sf *= n
    return sf

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 10210
    TARGET = 210  # squarefree part we need to achieve via XOR
    
    # Precompute squarefree parts for all numbers 211..10210
    sf_parts = {}
    for n in range(211, MAX_VAL + 1):
        sf_parts[n] = _squarefree_part(n)
    
    best_result = None
    best_max = MAX_VAL + 1
    
    # For each M from 211 upward, check if there exists a subset S ⊆ {211..M}
    # with max(S) = M (or ≤ M) such that XOR of sf_parts of elements = TARGET
    # 
    # BFS approach: 
    # - State: current XOR value
    # - Start: 0 (empty set)
    # - Transitions: XOR with sf_part of any number in [211, M]
    # - Goal: reach TARGET
    # - Track: which numbers were used
    #
    # But BFS over all subsets is exponential. Instead, use a smarter approach:
    # For small M, we can do BFS since the number of distinct squarefree parts is limited.
    
    # Precompute: for each squarefree part value, list of numbers that have it
    sf_to_nums = {}
    for n in range(211, MAX_VAL + 1):
        sf = sf_parts[n]
        if sf not in sf_to_nums:
            sf_to_nums[sf] = []
        sf_to_nums[sf].append(n)
    
    # For each M, we want to find a subset of {211..M} with XOR = TARGET
    # and at least one element = M (to ensure max = M, though we just need max ≤ M)
    # Actually, we just need max ≤ M, and we're iterating M upward, so first M that works is optimal.
    
    # BFS: state is (xor_value, frozenset_of_numbers_used)
    # But this is too large. Instead, use a different approach:
    # 
    # Key insight: we only care about the XOR value and the max element used.
    # For each M, we can do a BFS where:
    # - We process numbers in [211, M]
    # - State: current XOR value
    # - We want to reach TARGET
    # - We track the set of numbers used (as a tuple for hashing)
    #
    # This is still potentially large, but for small M it's manageable.
    
    # Alternative: since we're looking for the minimum M, and M starts at 211,
    # let's try small subsets first.
    
    # Single element: sf_parts[M] == TARGET
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline - 5.0:
            break
        if M >= best_max:
            break
        
        if sf_parts[M] == TARGET:
            # Verify: M * 210 should be a perfect square
            if _is_perfect_square(M * 210):
                best_max = M
                best_result = [M]
                break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Pairs: sf_parts[a] XOR sf_parts[M] == TARGET, i.e., sf_parts[a] == TARGET XOR sf_parts[M]
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline - 5.0:
            break
        if M >= best_max:
            break
        
        target_sf = TARGET ^ sf_parts[M]
        # Find a in [211, M) with sf_parts[a] == target_sf
        # a = target_sf * t^2 for some t
        if target_sf > 0:
            t_max = isqrt((M - 1) // target_sf)
            for t in range(1, t_max + 1):
                a = target_sf * t * t
                if a > M or a <= 210:
                    if a > M:
                        break
                    continue
                if a == M:
                    continue
                # Verify
                if _is_perfect_square(a * M * 210):
                    best_max = M
                    best_result = [a, M]
                    break
        if best_result is not None:
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Triples and larger: use BFS
    # For each M, do a BFS to find a subset of {211..M} with XOR = TARGET
    # State: (xor_value, tuple_of_numbers)
    # Start: (0, ())
    # Goal: xor_value == TARGET
    # We want to minimize the number of elements (or just find any)
    
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline - 3.0:
            break
        if M >= best_max:
            break
        
        # BFS for subset with XOR = TARGET using numbers in [211, M]
        # To keep it manageable, limit the BFS depth and state space
        # State: (xor_value, frozenset_of_numbers)
        # But frozenset is expensive. Instead, track the max number used to ensure we use M.
        
        # Actually, let's just do a simple BFS tracking xor values and the set of numbers used
        # For small M, this should be fine.
        
        # Use a dict: xor_value -> (set_of_numbers_used, parent_xor, last_number)
        # This allows reconstruction
        
        # Simpler: just track reachable xor values with the set of numbers that achieves them
        # For each xor value, keep the smallest set (by max element or count)
        
        # Let's do: reachable[xor_val] = tuple of numbers used
        # Start: reachable[0] = ()
        # For each number n in [211, M]:
        #   For each xor_val in list(reachable.keys()):
        #     new_xor = xor_val ^ sf_parts[n]
        #     if new_xor not in reachable or len(reachable[new_xor]) > len(reachable[xor_val]) + 1:
        #       reachable[new_xor] = reachable[xor_val] + (n,)
        
        # This is O(M * |reachable|) which could be large, but for small M it's OK.
        
        reachable = {0: ()}
        
        for n in range(211, M + 1):
            if time.time() > deadline - 2.0:
                break
            sf_n = sf_parts[n]
            # For each current state, try adding n
            new_states = {}
            for xor_val, nums in list(reachable.items()):
                new_xor = xor_val ^ sf_n
                new_nums = nums + (n,)
                # Keep the solution with fewer elements, or if same, the one with smaller max
                if new_xor not in reachable or len(new_nums) < len(reachable[new_xor]):
                    new_states[new_xor] = new_nums
            # Merge
            for xor_val, nums in new_states.items():
                if xor_val not in reachable or len(nums) < len(reachable[xor_val]):
                    reachable[xor_val] = nums
            
            # Check if we reached TARGET
            if TARGET in reachable:
                nums = reachable[TARGET]
                # Verify
                prod = 210
                for x in nums:
                    prod *= x
                if _is_perfect_square(prod):
                    best_max = M
                    best_result = list(nums)
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback: known solution 840
    return [840]

# EVOLVE-BLOCK-END
