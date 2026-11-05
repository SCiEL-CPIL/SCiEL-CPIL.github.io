from fractions import Fraction
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    # Precompute reciprocals as Fractions
    recip = {x: Fraction(1, x) for x in range(1, 25)}
    
    # Key insight: We need to find the largest subset S of [1,24] such that
    # for no x in S does 1/x = sum of 1/y for any non-empty subset of S\{x}
    
    # Strategy: Use backtracking with pruning
    # Order elements by "difficulty" - smaller elements are harder to include
    # because their reciprocals are larger and easier to match
    
    # Precompute: for each element x, which subsets of other elements could potentially sum to 1/x
    # This is expensive, so we'll use a smarter approach
    
    def check_safe(x, current_set):
        """Check if adding x to current_set keeps the set valid.
        We only need to check: does 1/x equal sum of 1/y for any non-empty subset of current_set?
        (The other direction - whether elements in current_set can be formed from subsets including x - 
         is already handled by previous checks, but we need to verify both directions)
        """
        others = list(current_set)
        n = len(others)
        target = recip[x]
        
        if n == 0:
            return True
        
        # Use meet-in-the-middle for efficiency
        if n <= 12:
            # Brute force all non-empty subsets
            for mask in range(1, 1 << n):
                s = Fraction(0, 1)
                for i in range(n):
                    if mask & (1 << i):
                        s += recip[others[i]]
                        if s > target:
                            break
                if s == target:
                    return False
            return True
        else:
            # Meet-in-the-middle
            mid = n // 2
            left = others[:mid]
            right = others[mid:]
            
            # Generate all subset sums for left half (including empty)
            left_sums = [Fraction(0, 1)]
            for elem in left:
                new_sums = []
                for s in left_sums:
                    ns = s + recip[elem]
                    if ns <= target:
                        new_sums.append(ns)
                left_sums.extend(new_sums)
            
            # Generate all subset sums for right half (including empty)
            right_sums = [Fraction(0, 1)]
            for elem in right:
                new_sums = []
                for s in right_sums:
                    ns = s + recip[elem]
                    if ns <= target:
                        new_sums.append(ns)
                right_sums.extend(new_sums)
            
            # Convert right_sums to set for O(1) lookup
            right_set = set(right_sums)
            
            # Check if any left_sum + right_sum == target
            # where the combined subset is non-empty
            for ls in left_sums:
                needed = target - ls
                if needed in right_set:
                    # Combined subset is non-empty unless both are empty
                    if not (ls == Fraction(0, 1) and needed == Fraction(0, 1)):
                        return False
            
            return True
    
    def is_valid_set(s):
        """Check if the entire set s is reciprocal-subsum-free."""
        s_list = list(s)
        for x in s_list:
            others = [y for y in s_list if y != x]
            if not check_safe(x, set(others)):
                return False
        return True
    
    # Greedy approach with multiple orderings
    best_set = set()
    
    # Try greedy from different starting points
    orderings = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        [24, 1, 23, 2, 22, 3, 21, 4, 20, 5, 19, 6, 18, 7, 17, 8, 16, 9, 15, 10, 14, 11, 13, 12],  # Alternating
    ]
    
    for ordering in orderings:
        candidate = set()
        for x in ordering:
            if time.time() > deadline - 5:
                break
            new_candidate = candidate | {x}
            if check_safe(x, candidate):
                candidate = new_candidate
        
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Try to improve by adding excluded elements
    excluded = set(range(1, 25)) - best_set
    for x in sorted(excluded, reverse=True):
        if time.time() > deadline - 5:
            break
        if check_safe(x, best_set):
            best_set.add(x)
    
    # Try local search: swap elements to potentially increase size
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_list = sorted(best_set)
        excluded = sorted(set(range(1, 25)) - best_set, reverse=True)
        
        for i, x in enumerate(current_list):
            if time.time() > deadline - 10:
                break
            for y in excluded:
                if y in best_set:
                    continue
                # Try removing x and adding y
                new_set = best_set - {x} | {y}
                if len(new_set) > len(best_set):
                    if is_valid_set(new_set):
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break
    
    # Final attempt: try to add any remaining excluded elements
    excluded = sorted(set(range(1, 25)) - best_set, reverse=True)
    for x in excluded:
        if time.time() > deadline - 3:
            break
        if check_safe(x, best_set):
            best_set.add(x)
    
    # Ensure non-empty
    if not best_set:
        best_set = {24}
    
    return sorted(best_set)
