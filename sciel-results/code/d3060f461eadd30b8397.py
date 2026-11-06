import sys
from functools import lru_cache

def construct():
    """
    Returns a nonempty subset of [1, 24] that maximizes cardinality
    such that no reciprocal of a selected element equals the sum of
    reciprocals of a subset of the other selected elements.
    Uses Branch and Bound with integer arithmetic (LCM scaling)
    and Meet-in-the-Middle for subset sum feasibility.
    """
    
    # Precompute weights: LCM(1..24) / i
    LCM = 5354228880
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = LCM // i
        
    # Elements 1 to 24
    elements = list(range(1, 25))
    
    # Helper to check if target can be formed by subset of nums
    # Using Meet-in-the-Middle for O(2^(n/2)) performance
    def can_form_target(nums, target):
        if target == 0:
            return True
        if target < 0:
            return False
        n = len(nums)
        if n == 0:
            return False
        
        # Split nums into two halves
        mid = n // 2
        left = nums[:mid]
        right = nums[mid:]
        
        # Generate all subset sums for left half
        def gen_sums(arr):
            sums = [0]
            for x in arr:
                # Add x to existing sums
                sums += [s + x for s in sums]
            return sums
        
        left_sums = gen_sums(left)
        right_sums = gen_sums(right)
        
        # Store right sums in a set for O(1) lookup
        right_set = set(right_sums)
        
        # Check if any left sum + right sum == target
        for s in left_sums:
            if target - s in right_set:
                return True
        return False

    best_set = []
    
    def check_conflict(x, S):
        """
        Checks if adding x to S creates a violation.
        S is a list of selected elements.
        """
        wx = weights[x]
        
        # 1. Does x equal sum of subset of S?
        S_weights = [weights[y] for y in S]
        if can_form_target(S_weights, wx):
            return True
            
        # 2. Does any y in S equal sum of subset of (S \ {y}) U {x}?
        # This is equivalent to: w_y - w_x is a subset sum of S \ {y, x}
        # Note: x is not in S yet, so S \ {x} is just S.
        # We need to exclude y from S.
        for i, y in enumerate(S):
            wy = weights[y]
            if wy < wx:
                continue
            
            # Construct list of weights from S excluding y
            # Optimization: Instead of creating a new list every time,
            # we can just slice or filter. Given small N, filtering is fine.
            # But we can be smarter:
            # S_weights is already computed.
            # We need S_weights without index i.
            remaining_weights = S_weights[:i] + S_weights[i+1:]
            
            if can_form_target(remaining_weights, wy - wx):
                return True
                
        return False

    def backtrack(idx, current_set):
        nonlocal best_set
        
        # Pruning: if even adding all remaining can't beat best
        if len(current_set) + (24 - idx) <= len(best_set):
            return
            
        if idx == 24:
            if len(current_set) > len(best_set):
                best_set = current_set.copy()
            return
            
        x = elements[idx]
        
        # Option 1: Include x
        # Check if valid
        if not check_conflict(x, current_set):
            current_set.append(x)
            backtrack(idx + 1, current_set)
            current_set.pop()
            
        # Option 2: Exclude x
        backtrack(idx + 1, current_set)

    backtrack(0, [])
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(result)
