"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # safety margin before 90s kill

def _time_remaining():
    return _DEADLINE - time.time()

def _is_subsum_free(S):
    """Check if set S is reciprocal-subsum-free.
    For each element x in S, check that 1/x != sum of 1/y for any nonempty subset of S\{x}.
    """
    S_list = sorted(S)
    n = len(S_list)
    if n == 0:
        return True
    # Precompute reciprocals
    recip = {x: Fraction(1, x) for x in S_list}
    # For each target element, check all nonempty subsets of others
    for i, target in enumerate(S_list):
        target_recip = recip[target]
        others = [x for j, x in enumerate(S_list) if j != i]
        m = len(others)
        # Enumerate all nonempty subsets of others
        # For efficiency, we can use the fact that if m is small this is fine
        # m is at most 23, which is 2^23 = 8.4M subsets - too many
        # Better approach: use the property that we only need to check if target_recip
        # is in the set of all subset sums of reciprocals of others
        # Since m can be up to 23, we need a smarter approach
        
        # Actually, let's use a different approach:
        # The condition is: 1/x = sum_{y in T} 1/y for some nonempty T subset of S\{x}
        # This is equivalent to: x * sum_{y in T} 1/y = 1, i.e., sum_{y in T} x/y = 1
        # Or: lcm(x, y1, ..., yk) / x = sum_{j} lcm(x, y1, ..., yk) / yj
        # This is getting complex. Let me just enumerate subsets but with early termination.
        
        # For m up to ~20, 2^m can be large. Let's use a meet-in-the-middle approach
        # or just check if m is small enough for brute force.
        
        # Actually, for the sizes we're dealing with (max 24 elements), m is at most 23.
        # 2^23 is too large. Let's use a different strategy.
        
        # Key insight: if 1/x = sum of reciprocals of a subset of others,
        # then the subset must consist of elements all > x (since each 1/y < 1/x for y > x,
        # and we need the sum to equal 1/x). Wait, that's not right - elements can be smaller too.
        
        # Actually no: if y < x, then 1/y > 1/x, so a single such element would already exceed 1/x.
        # So any element in the subset must have y > x.
        # This means we only need to consider elements in S that are greater than x.
        
        greater = [y for y in others if y > x]
        if not greater:
            continue
        
        # Now check all nonempty subsets of 'greater'
        # The number of elements greater than x in S is at most 24 - x
        # For x >= 13, this is at most 11 elements, so 2^11 = 2048 subsets - manageable
        # For x = 1, this could be 23 elements, but we're likely working with larger sets
        
        m2 = len(greater)
        if m2 <= 20:  # 2^20 = 1M, might be slow but doable with early termination
            for r in range(1, m2 + 1):
                for subset in itertools.combinations(greater, r):
                    s = sum(recip[y] for y in subset)
                    if s == target_recip:
                        return False
        else:
            # Use meet-in-the-middle for larger sets
            half = m2 // 2
            left = greater[:half]
            right = greater[half:]
            
            left_sums = set()
            for r in range(0, len(left) + 1):
                for subset in itertools.combinations(left, r):
                    s = sum(recip[y] for y in subset)
                    left_sums.add(s)
            
            right_sums = set()
            for r in range(0, len(right) + 1):
                for subset in itertools.combinations(right, r):
                    s = sum(recip[y] for y in subset)
                    right_sums.add(s)
            
            # Check if target_recip can be formed as left_sum + right_sum
            # where at least one of left_sum or right_sum is non-zero (not both empty subset)
            for ls in left_sums:
                if ls == 0:
                    continue
                if target_recip - ls in right_sums:
                    # Need to ensure we're not using the empty subset from both sides
                    if ls != 0 or (target_recip - ls) != 0:
                        return False
                # Also check if right_sum alone works (ls = 0 case)
            for rs in right_sums:
                if rs == 0:
                    continue
                if target_recip - rs in left_sums:
                    if rs != 0 or (target_recip - rs) != 0:
                        return False
            
            # Also check single-side sums
            if target_recip in left_sums and target_recip != 0:
                return False
            if target_recip in right_sums and target_recip != 0:
                return False
            
            # The above double-checks; let me simplify
            # Actually the logic above is redundant. Let me redo:
            # We need ls + rs = target_recip where (ls, rs) != (0, 0)
            # ls from left_sums, rs from right_sums
            # This is already handled if we check all pairs, but that's |left_sums| * |right_sums|
            # which could be large. The approach above checks for each ls if target-ls is in right_sums,
            # which is O(|left_sums|) with set lookups. That's correct.
            
            # But I need to be careful about the (0,0) case.
            # left_sums contains 0 (empty subset), right_sums contains 0 (empty subset).
            # We want to exclude the case where both are 0.
            # If ls = 0 and rs = target_recip, that's valid (non-empty right subset) unless target_recip = 0
            # If ls = target_recip and rs = 0, that's valid (non-empty left subset) unless target_recip = 0
            # If ls = 0 and rs = 0, that gives 0 = target_recip which is false since target_recip > 0
            
            # So actually the (0,0) case never triggers a false positive since target_recip > 0.
            # The check above is correct but overly complex. Let me simplify.
            
            for ls in left_sums:
                needed = target_recip - ls
                if needed in right_sums:
                    # This combination works. Need at least one non-empty subset.
                    # If ls == 0 and needed == 0, that's empty+empty = 0 != target_recip (since target > 0)
                    # So we're fine.
                    return False
    
    return True

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Start with a known good set: {13, 14, ..., 24} has 12 elements
    best = list(range(13, 25))
    
    # Verify it's subsum-free
    if not _is_subsum_free(best):
        # Fallback
        best = list(range(13, 25))
        # This should be subsum-free, but let's verify
        if not _is_subsum_free(best):
            # Very unlikely, but return something nonempty
            return [24]
    
    best_size = len(best)
    
    # Try to find larger sets using randomized search
    # Strategy: start with a base set and try to add elements
    
    # First, let's try to find a larger set by systematic search
    # Try all subsets of [1, 24] of increasing size, starting from size 13
    
    # Actually, let's use a smarter approach:
    # 1. Start with a large set of "safe" elements (large numbers)
    # 2. Try to add smaller elements one by one
    
    # Let's try different base sets and see what maximum size we can achieve
    
    # Approach: greedy with backtracking
    # Try to build a set by adding elements in some order, checking subsum-freeness
    
    # Let's try a randomized greedy approach
    random.seed(42)
    
    for trial in range(50):
        if _time_remaining() < 5:
            break
        
        # Random permutation of [1, 24]
        candidates = list(range(1, 25))
        random.shuffle(candidates)
        
        current = []
        for x in candidates:
            if _time_remaining() < 3:
                break
            new_set = current + [x]
            if _is_subsum_free(new_set):
                current = new_set
        
        if len(current) > best_size:
            best = current
            best_size = len(current)
        
        if best_size >= 14:  # We found something better
            # Try to improve further
            break
    
    # Try to add elements to the best set found
    if _time_remaining() > 10:
        # Try adding each element from [1, 24] not in best
        improved = True
        while improved:
            improved = False
            for x in range(1, 25):
                if _time_remaining() < 3:
                    break
                if x not in best:
                    new_set = best + [x]
                    if _is_subsum_free(new_set):
                        best = new_set
                        improved = True
                        break
    
    # Final verification
    if not _is_subsum_free(best):
        # Fallback to a known good set
        best = list(range(13, 25))
    
    return best

# EVOLVE-BLOCK-END
