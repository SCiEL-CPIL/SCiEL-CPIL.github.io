"""Stronger algorithm for reciprocal weight of 3-progression-free sets on [1,256]."""
import time
import random
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _can_add_bitset(x, bitset, n=256):
    """Check if x can be added to set represented by bitset without creating a 3-AP. O(1) with bit ops."""
    # For each a in set, check if 2x-a or (a+x)/2 is also in set
    # We need to iterate over set elements, but we can use bit operations
    # Actually, let's use a different approach: precompute conflicts
    # For a given x, the forbidden pairs are:
    # - a, c where a + c = 2x (x is middle)
    # - a, b where a + b = 2x and b = (a+x)/2... wait
    # 
    # If we add x, a 3-AP is created if there exist a, b in S such that:
    # Case 1: a, x, c with c = 2x-a in S (x is middle)
    # Case 2: x, b, a with b = (x+a)/2 in S (x is first)
    # Case 3: a, b, x with b = (a+x)/2 in S (x is last)
    #
    # Cases 2 and 3 are the same: b = (a+x)/2 for some a in S
    #
    # So we need to check:
    # 1. For each a in S: is (2x - a) in S? (and 2x-a != x, 2x-a != a)
    # 2. For each a in S: is ((a+x)/2) in S? (if a+x is even, and (a+x)/2 != x, (a+x)/2 != a)
    #
    # With bitset, we can do this faster by precomputing.
    # Let's use a hybrid: iterate over set elements but use bitset for membership.
    
    # Get list of elements in set
    elements = []
    b = bitset
    while b:
        low = b & (-b)
        idx = low.bit_length() - 1
        elements.append(idx)
        b ^= low
    
    for a in elements:
        # Case 1: x is middle. c = 2x - a
        c = 2 * x - a
        if c != x and c != a and 1 <= c <= n:
            if (bitset >> c) & 1:
                return False
        # Case 2: x is endpoint. b = (a+x)/2
        s_sum = a + x
        if s_sum % 2 == 0:
            b_val = s_sum // 2
            if b_val != x and b_val != a and 1 <= b_val <= n:
                if (bitset >> b_val) & 1:
                    return False
    return True

def _is_3ap_free_bitset(bitset, n=256):
    """Check if set represented by bitset is 3-AP free."""
    elements = []
    b = bitset
    while b:
        low = b & (-b)
        idx = low.bit_length() - 1
        elements.append(idx)
        b ^= low
    
    for i, a in enumerate(elements):
        for j in range(i+1, len(elements)):
            c = elements[j]
            # Check if a, b, c form AP where b = (a+c)/2
            s = a + c
            if s % 2 == 0:
                b_val = s // 2
                if b_val != a and b_val != c and 1 <= b_val <= n:
                    if (bitset >> b_val) & 1:
                        return False
    return True

def _recip_sum_from_bitset(bitset, n=256):
    """Compute exact sum of reciprocals from bitset."""
    total = Fraction(0, 1)
    b = bitset
    while b:
        low = b & (-b)
        idx = low.bit_length() - 1
        total += Fraction(1, idx)
        b ^= low
    return total

def _greedy_by_reciprocal(n=256):
    """Greedy: add elements in order of smallest x first (largest reciprocal)."""
    bitset = 0
    for x in range(1, n+1):
        if _can_add_bitset(x, bitset, n):
            bitset |= (1 << x)
    return bitset

def _branch_and_bound(n=256, deadline=None, time_limit=50.0):
    """Branch and bound search for optimal 3-free set."""
    start_time = time.time()
    
    # Precompute suffix sums of reciprocals for pruning
    # suffix[i] = sum of 1/j for j from i to n
    suffix = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix[i] = suffix[i+1] + 1.0/i
    
    # Start with greedy solution
    best_bitset = _greedy_by_reciprocal(n)
    best_val = float(_recip_sum_from_bitset(best_bitset, n))
    
    # Branch and bound: decide for each x from 1 to n whether to include it
    # Use DFS with pruning
    def dfs(x, current_bitset, current_val):
        nonlocal best_bitset, best_val, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if x > n:
            if current_val > best_val:
                best_val = current_val
                best_bitset = current_bitset
            return
        
        # Upper bound: current_val + suffix[x]
        if current_val + suffix[x] <= best_val:
            return
        
        # Option 1: Don't include x
        dfs(x + 1, current_bitset, current_val)
        
        # Option 2: Include x if valid
        if _can_add_bitset(x, current_bitset, n):
            new_bitset = current_bitset | (1 << x)
            new_val = current_val + 1.0/x
            if new_val + suffix[x+1] > best_val:
                dfs(x + 1, new_bitset, new_val)
    
    # Start DFS from x=1
    dfs(1, 0, 0.0)
    
    return best_bitset

def _local_search_bitset(start_bitset, n=256, max_iters=50000, seed=0):
    """Local search on bitset representation."""
    rng = random.Random(seed)
    current = start_bitset
    current_val = float(_recip_sum_from_bitset(current, n))
    best = current
    best_val = current_val
    
    # Get list of elements in current set
    def get_elements(bitset):
        elems = []
        b = bitset
        while b:
            low = b & (-b)
            idx = low.bit_length() - 1
            elems.append(idx)
            b ^= low
        return elems
    
    for it in range(max_iters):
        if it % 1000 == 0:
            if time.time() > 55:
                break
        
        elems = get_elements(current)
        if not elems:
            break
        
        # Try removing a random element and adding a better one
        remove_idx = rng.randrange(len(elems))
        removed = elems[remove_idx]
        
        # Find candidates to add
        candidates = []
        for x in range(1, n+1):
            if not (current >> x) & 1:
                candidates.append(x)
        
        if not candidates:
            break
        
        # Try a few random candidates
        for _ in range(5):
            add_cand = rng.choice(candidates)
            temp_bitset = current & ~(1 << removed)
            if _can_add_bitset(add_cand, temp_bitset, n):
                new_bitset = temp_bitset | (1 << add_cand)
                new_val = current_val - 1.0/removed + 1.0/add_cand
                if new_val > current_val:
                    current = new_bitset
                    current_val = new_val
                    if current_val > best_val:
                        best_val = current_val
                        best = current
                    break
    
    return best

def construct():
    """Main entry point: find a 3-free subset of [1,256] maximizing sum of reciprocals."""
    start_time = time.time()
    
    # Phase 1: Branch and bound
    best_bitset = _branch_and_bound(256, time_limit=45.0)
    
    # Phase 2: Local search to polish
    if time.time() - start_time < 55:
        ls_result = _local_search_bitset(best_bitset, 256, max_iters=100000, seed=42)
        ls_val = float(_recip_sum_from_bitset(ls_result, 256))
        bb_val = float(_recip_sum_from_bitset(best_bitset, 256))
        if ls_val > bb_val:
            best_bitset = ls_result
    
    # Convert bitset to list
    result = []
    b = best_bitset
    while b:
        low = b & (-b)
        idx = low.bit_length() - 1
        result.append(idx)
        b ^= low
    
    return result if result else [1]

# EVOLVE-BLOCK-END
