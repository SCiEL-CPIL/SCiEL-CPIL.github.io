import sys
from fractions import Fraction
from functools import lru_cache

def construct():
    """
    Returns a nonempty subset of [1, 24] that maximizes cardinality
    such that no reciprocal of a selected element equals the sum of
    reciprocals of a subset of the other selected elements.
    Uses Branch and Bound with integer arithmetic (LCM scaling).
    """
    
    # Precompute weights: LCM(1..24) / i
    # LCM(1..24) = 5354228880
    LCM = 5354228880
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = LCM // i
        
    elements = list(range(1, 25))
    
    # Helper to check if target can be formed by subset of nums
    # nums are positive integers
    def can_form_target(nums, target):
        if target == 0:
            return True
        if target < 0:
            return False
        if not nums:
            return False
        # Simple DFS with pruning
        # Sort nums descending to hit target faster or prune faster?
        # Actually, just iterate.
        # To speed up, we can use a set of reachable sums if nums is small, 
        # but here nums is up to 23 elements.
        # DFS is better for specific target.
        
        def dfs(idx, current_sum):
            if current_sum == target:
                return True
            if idx == len(nums) or current_sum > target:
                return False
            # Option 1: include nums[idx]
            if dfs(idx + 1, current_sum + nums[idx]):
                return True
            # Option 2: exclude nums[idx]
            if dfs(idx + 1, current_sum):
                return True
            return False
        
        return dfs(0, 0)

    best_set = []
    
    def check_conflict(x, S):
        """
        Checks if adding x to S creates a violation.
        S is a list of selected elements.
        """
        wx = weights[x]
        S_weights = [weights[y] for y in S]
        
        # 1. Does x equal sum of subset of S?
        if can_form_target(S_weights, wx):
            return True
            
        # 2. Does any y in S equal sum of subset of (S \ {y}) U {x}?
        # This is equivalent to: w_y - w_x is a subset sum of S \ {y, x}
        for y in S:
            wy = weights[y]
            if wy < wx:
                # w_y - w_x < 0, cannot be a sum of positive weights
                continue
            # We need to check if w_y - w_x can be formed by S excluding y and x
            # S_weights contains w_y. We need to exclude it.
            # Since x is not in S, we just exclude y from S_weights.
            remaining_weights = [w for w, val in zip(S_weights, S) if val != y]
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
