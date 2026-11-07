import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s
    
    N = 210
    MAX_VAL = 10210
    COUNT = 100
    
    # Target parity: 210 = 2^1 * 3^1 * 5^1 * 7^1, so parity = (1,1,1,1)
    target_parity = (1, 1, 1, 1)
    
    def parity_vec(n):
        """Compute parity vector for primes 2,3,5,7"""
        v = [0, 0, 0, 0]
        for i, p in enumerate([2, 3, 5, 7]):
            while n % p == 0:
                v[i] ^= 1
                n //= p
        return tuple(v)
    
    # Precompute parity vectors for all numbers in range
    # We need numbers strictly > 210 and <= 10210
    lo = N + 1  # 211
    hi = MAX_VAL  # 10210
    
    # For efficiency, precompute parity vectors
    # But first, let's try a greedy approach with local optimization
    
    # Strategy: Start with the 100 smallest numbers, then try to fix the parity
    # by swapping elements
    
    # Initial selection: 211 to 310
    initial = list(range(lo, lo + COUNT))
    
    def parity_sum(nums):
        s = [0, 0, 0, 0]
        for n in nums:
            pv = parity_vec(n)
            for i in range(4):
                s[i] ^= pv[i]
        return tuple(s)
    
    current_parity = parity_sum(initial)
    
    if current_parity == target_parity:
        result = [int(x) for x in initial]
        return result
    
    # We need to fix the parity difference
    # diff = current_parity XOR target_parity
    diff = tuple(a ^ b for a, b in zip(current_parity, target_parity))
    
    # We need to find a subset of the current set to remove and a subset of 
    # outside numbers to add, such that the parity change equals diff.
    # Simplest: swap one number in with one number out
    # parity_change = parity_vec(new) XOR parity_vec(old) = diff
    
    # Try single swaps
    current_set = set(initial)
    max_current = max(initial)
    
    # For each number in current set, for each number outside, check if swap fixes parity
    # This is O(n * m) which could be large, so let's be smart
    
    # Build a map from parity vector to list of numbers (outside current set)
    outside_parity_map = defaultdict(list)
    for n in range(lo, hi + 1):
        if n not in current_set:
            pv = parity_vec(n)
            outside_parity_map[pv].append(n)
    
    # For each number in current set, find if there's an outside number that fixes the parity
    found_swap = False
    best_max = max_current
    best_result = None
    
    for old in initial:
        old_parity = parity_vec(old)
        # We need new_parity such that old_parity XOR new_parity = diff
        # So new_parity = old_parity XOR diff
        needed_parity = tuple(a ^ b for a, b in zip(old_parity, diff))
        
        if needed_parity in outside_parity_map:
            candidates = outside_parity_map[needed_parity]
            # Pick the smallest candidate to minimize max
            for new in candidates:
                new_max = max(max_current, new)
                if new_max < best_max:
                    best_max = new_max
                    best_result = [n if n != old else new for n in initial]
                    found_swap = True
                    break
    
    if found_swap:
        # Verify
        result = [int(x) for x in best_result]
        # Check all constraints
        assert len(result) == COUNT
        assert len(set(result)) == COUNT
        assert all(isinstance(x, int) and not isinstance(x, bool) for x in result)
        assert all(210 < x <= 10210 for x in result)
        
        # Verify square condition
        prod = 210
        for x in result:
            prod *= x
        s = int(prod ** 0.5)
        # Use integer square root for exact check
        s2 = s * s
        if s2 < prod:
            s += 1
            s2 = s * s
        assert s2 == prod, f"Not a perfect square"
        
        return result
    
    # If single swap didn't work, try two swaps (remove 2, add 2)
    # This is more complex, so let's try a different approach:
    # Use a smarter search with the time budget
    
    # Alternative: use a local search / hill climbing approach
    # Start with 100 smallest, then iteratively swap to improve
    
    import random
    
    best_solution = None
    best_max_val = float('inf')
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random-ish set biased toward small numbers
        if restart == 0:
            current = list(range(lo, lo + COUNT))
        else:
            # Random selection with bias toward small numbers
            current = []
            available = list(range(lo, lo + 500))  # Focus on small range
            random.shuffle(available)
            current = available[:COUNT]
        
        current_set = set(current)
        current_parity = parity_sum(current)
        
        # Local search: try swaps to fix parity and reduce max
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_parity = parity_sum(current)
            
            if current_parity == target_parity:
                max_val = max(current)
                if max_val < best_max_val:
                    best_max_val = max_val
                    best_solution = list(current)
                # Try to reduce max by swapping large elements
                # Sort by value descending
                sorted_current = sorted(current, reverse=True)
                for i, old in enumerate(sorted_current):
                    if time.time() > deadline:
                        break
                    old_parity = parity_vec(old)
                    needed_parity = tuple(a ^ b for a, b in zip(old_parity, current_parity))
                    # Find smallest outside number with needed_parity
                    # that is smaller than old
                    if needed_parity in outside_parity_map:
                        for new in outside_parity_map[needed_parity]:
                            if new < old and new not in current_set:
                                # This swap keeps parity the same and reduces max
                                current[i] = new
                                current_set.remove(old)
                                current_set.add(new)
                                improved = True
                                break
                    if improved:
                        break
            else:
                # Need to fix parity - try single swap
                diff = tuple(a ^ b for a, b in zip(current_parity, target_parity))
                for i, old in enumerate(current):
                    if time.time() > deadline:
                        break
                    old_parity = parity_vec(old)
                    needed_parity = tuple(a ^ b for a, b in zip(old_parity, diff))
                    if needed_parity in outside_parity_map:
                        for new in outside_parity_map[needed_parity]:
                            if new not in current_set:
                                current[i] = new
                                current_set.remove(old)
                                current_set.add(new)
                                improved = True
                                break
                    if improved:
                        break
    
    if best_solution is not None:
        result = [int(x) for x in best_solution]
        # Final verification
        assert len(result) == COUNT
        assert len(set(result)) == COUNT
        assert all(isinstance(x, int) and not isinstance(x, bool) for x in result)
        assert all(210 < x <= 10210 for x in result)
        
        prod = 210
        for x in result:
            prod *= x
        # Integer square root
        s = int(prod ** 0.5)
        # Adjust
        while s * s < prod:
            s += 1
        while s * s > prod:
            s -= 1
        assert s * s == prod, "Not a perfect square"
        
        return result
    
    # Fallback: return the initial set (may not be valid, but we need to return something)
    # Actually, let's try a more systematic approach as last resort
    # Use the fact that we can find 100 numbers with the right parity
    
    # Systematic approach: find 99 numbers with parity (0,0,0,0) and 1 number with parity (1,1,1,1)
    # Or other combinations that sum to (1,1,1,1)
    
    # Parity vectors that sum to (1,1,1,1):
    # We need 100 vectors from the 16 possible that XOR to (1,1,1,1)
    
    # Let's find numbers with each parity type
    parity_to_numbers = defaultdict(list)
    for n in range(lo, hi + 1):
        pv = parity_vec(n)
        parity_to_numbers[pv].append(n)
    
    # Strategy: pick 1 number with parity (1,1,1,1) and 99 numbers with parity (0,0,0,0)
    # Or pick combinations
    
    # First, try: 1 number with (1,1,1,1) + 99 numbers with (0,0,0,0)
    if len(parity_to_numbers[(1,1,1,1)]) >= 1 and len(parity_to_numbers[(0,0,0,0)]) >= 99:
        chosen = [parity_to_numbers[(1,1,1,1)][0]]
        chosen.extend(parity_to_numbers[(0,0,0,0)][:99])
        if len(set(chosen)) == 100:
            result = [int(x) for x in chosen]
            return result
    
    # Try: 2 numbers with parities that XOR to (1,1,1,1) + 98 with (0,0,0,0)
    # Possible pairs: (a, b) where a XOR b = (1,1,1,1)
    # a XOR b = (1,1,1,1) means b = a XOR (1,1,1,1)
    for a in range(16):
        b = a ^ 0b1111  # (1,1,1,1) in binary
        if a <= b:  # Avoid duplicates
            pa = (a & 1, (a >> 1) & 1, (a >> 2) & 1, (a >> 3) & 1)
            pb = (b & 1, (b >> 1) & 1, (b >> 2) & 1, (b >> 3) & 1)
            if len(parity_to_numbers[pa]) >= 1 and len(parity_to_numbers[pb]) >= 1 and len(parity_to_numbers[(0,0,0,0)]) >= 98:
                chosen = [parity_to_numbers[pa][0], parity_to_numbers[pb][0]]
                chosen.extend(parity_to_numbers[(0,0,0,0)][:98])
                if len(set(chosen)) == 100:
                    result = [int(x) for x in chosen]
                    return result
    
    # This should have found a solution by now. If not, use a broader search.
    # As a last resort, return the smallest 100 numbers (may not satisfy parity, but we need to return)
    # Actually, let's do a proper search with the remaining time
    
    # Use a greedy approach with backtracking
    # Pick numbers one by one, tracking parity
    
    # This is getting complex. Let's use a simple but effective approach:
    # Find 100 numbers whose parity sums to target using a constructive method
    
    # We know that if we take numbers in pairs where each pair has parity (0,0,0,0),
    # plus one number with parity (1,1,1,1), we get the target.
    
    # Find the smallest number with parity (1,1,1,1)
    # And find 49 pairs of numbers each with the same parity (so their XOR is (0,0,0,0))
    
    # Actually, simpler: find 99 numbers with parity (0,0,0,0) and 1 with (1,1,1,1)
    # We already tried this above. Let's make sure we have enough numbers.
    
    # Count available numbers with each parity
    for pv in parity_to_numbers:
        pass  # Just for debugging
    
    # The above should work. If we get here, something went wrong.
    # Return a valid solution using a direct construction
    
    # Direct construction:
    # 210 = 2*3*5*7
    # We need product of selected * 210 = square
    # So product of selected must have parity (1,1,1,1) for primes 2,3,5,7
    
    # Let's just find the solution systematically
    # Pick the smallest 100 numbers and adjust
    
    # Use a different strategy: find 100 numbers where the combined parity is correct
    # by using a linear algebra approach over GF(2)
    
    # The parity vectors form a 4-dimensional space. We need 100 vectors that sum to (1,1,1,1).
    # This is always possible if we have enough numbers with various parities.
    
    # Greedy with correction:
    # 1. Pick 99 smallest numbers
    # 2. Compute their parity sum P
    # 3. Find the smallest number > 210 not in the set with parity P XOR (1,1,1,1)
    
    selected = list(range(lo, lo + 99))
    selected_set = set(selected)
    P = parity_sum(selected)
    needed = tuple(a ^ b for a, b in zip(P, target_parity))
    
    for n in range(lo, hi + 1):
        if n not in selected_set:
            if parity_vec(n) == needed:
                selected.append(n)
                break
    
    result = [int(x) for x in selected]
    
    # Verify
    assert len(result) == 100
    assert len(set(result)) == 100
    assert all(210 < x <= 10210 for x in result)
    
    prod = 210
    for x in result:
        prod *= x
    s = int(prod ** 0.5)
    while s * s < prod:
        s += 1
    while s * s > prod:
        s -= 1
    assert s * s == prod
    
    return result
