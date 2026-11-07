"""Rewritten program for the suite task; deterministic greedy construction with fast validation."""
import itertools
import random

# EVOLVE-BLOCK-START

def _check_validity_fast(sums_set, max_possible_sum):
    """
    Check if a set of subset sums satisfies the no-divisibility condition.
    Uses a boolean array for O(1) lookups.
    """
    if not sums_set:
        return True
    
    # If 1 is in sums, it divides everything.
    if 1 in sums_set and len(sums_set) > 1:
        return False
        
    max_sum = max(sums_set)
    # Create a lookup table
    # max_sum can be up to 15*256 = 3840. 
    # A list of booleans of size 4000 is cheap.
    lookup = [False] * (max_sum + 1)
    for s in sums_set:
        lookup[s] = True
        
    for s1 in sums_set:
        if s1 == 1:
            continue
        # Check multiples
        multiple = 2 * s1
        while multiple <= max_sum:
            if lookup[multiple]:
                return False
            multiple += s1
    return True

def construct():
    """
    Return a list of distinct integers in 1..256 satisfying the condition, maximizing size.
    Uses backtracking to find a set of size 4 or 5.
    """
    
    # Helper to generate subset sums from a list of numbers
    def get_sums(nums):
        sums = set()
        n = len(nums)
        for i in range(1, 1 << n):
            s = 0
            for j in range(n):
                if i & (1 << j):
                    s += nums[j]
            sums.add(s)
        return sums

    # Try to find a set of size K
    for target_size in [5, 4, 3]:
        # Use a simple backtracking search
        # To keep it fast, we limit the search space or use heuristics.
        # We'll try a few random permutations of candidates sorted by some heuristic.
        
        # Heuristic: Try numbers that are "large" to keep sums large and distinct?
        # Or "small" to keep sums small?
        # Let's try a mix.
        
        found = None
        candidates = list(range(1, 257))
        
        # Try a few random seeds
        for seed in range(20):
            random.seed(seed)
            # Shuffle candidates
            random.shuffle(candidates)
            
            # We want to pick target_size numbers.
            # Backtrack
            def backtrack(current_nums, current_sums, remaining_candidates, depth):
                if depth == target_size:
                    return current_nums[:]
                    
                # If we have more than 0 remaining, try to add one
                # To speed up, we can prune if current_sums is invalid (shouldn't happen)
                
                for i, c in enumerate(remaining_candidates):
                    # Try adding c
                    new_sums = set(current_sums)
                    new_sums.add(c)
                    valid = True
                    # Check for duplicate sums
                    for s in current_sums:
                        ns = s + c
                        if ns in new_sums:
                            valid = False
                            break
                        new_sums.add(ns)
                    
                    if not valid:
                        continue
                        
                    if _check_validity_fast(new_sums, 256 * target_size):
                        res = backtrack(current_nums + [c], new_sums, remaining_candidates[i+1:], depth + 1)
                        if res:
                            return res
                            
                return None

            # Start backtracking
            # To avoid exponential blowup, we might need to limit the branching factor
            # or use a greedy approach within the backtrack.
            # For now, let's try it. If it's too slow, we fall back to greedy.
            
            # Optimization: Only try the first few candidates in the shuffled list for each step?
            # No, that's just greedy.
            # Let's try a limited depth search.
            
            # Actually, for target_size=4, the search space is C(256, 4) ~ 1.7e8. Too big for pure backtrack.
            # We need a smarter approach.
            
            # Greedy with random restarts is better for this size.
            # Let's stick to the original approach but improve the validity check and add a swap heuristic.
            
            # Re-implementing a faster greedy:
            current_nums = []
            current_sums = set()
            
            # Sort candidates by descending value? 
            # Larger values create larger sums, which are less likely to divide smaller sums.
            # But we need distinct sums.
            
            for c in candidates:
                # Try to add c
                new_sums = set(current_sums)
                new_sums.add(c)
                valid = True
                for s in current_sums:
                    ns = s + c
                    if ns in new_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                
                if not valid:
                    continue
                
                if _check_validity_fast(new_sums, 256 * target_size):
                    current_nums.append(c)
                    current_sums = new_sums
                    if len(current_nums) == target_size:
                        found = current_nums[:]
                        break
            
            if found:
                break
        
        if found:
            # Final verification
            if len(found) == target_size:
                return [int(x) for x in found]
        
    # Fallback to known valid set
    return [4, 7, 19]

# EVOLVE-BLOCK-END
