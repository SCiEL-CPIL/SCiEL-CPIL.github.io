import time

# Precompute divisors of 5040 = 2^4 * 3^2 * 5 * 7
_DIVS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20,
    21, 24, 28, 30, 35, 36, 40, 42, 45, 48, 56, 60, 63, 70,
    72, 80, 84, 90, 105, 112, 120, 126, 140, 144, 168, 180,
    210, 224, 240, 252, 280, 315, 336, 360, 420, 504, 560,
    630, 720, 840, 1008, 1260, 1680, 2520, 5040
)
_DIV_SET = frozenset(_DIVS)
TARGET = 4979

def _find_solution_k(target, divs, k, deadline):
    """
    Find k distinct divisors from `divs` that sum to `target`.
    Uses backtracking with pruning and time-check.
    Returns a list of k divisors, or None.
    """
    divs = sorted(divs, reverse=True)
    n = len(divs)
    
    # Pre-compute suffix sums for pruning
    # suffix_sums[i] = sum of divs[i:]
    # This helps prune if current_sum + sum of remaining possible largest values < target
    # But since we want exactly target, and we are picking from a set, 
    # a tighter bound is: current_sum + (sum of the largest (k - depth) remaining elements) >= target
    # And current_sum + (sum of the smallest (k - depth) remaining elements) <= target
    
    # For simplicity and speed with small k, we can just use the max possible sum from remaining.
    # Max possible sum from remaining indices (i+1..n-1) picking (k-depth) items is sum of the largest (k-depth) items in that range.
    # Since divs is sorted descending, the largest (k-depth) items starting from index i+1 are divs[i+1], divs[i+2], ..., divs[i+k-depth].
    
    def backtrack(start, depth, current_sum, current_path):
        if time.time() > deadline:
            return None
        
        if depth == k:
            if current_sum == target:
                return list(current_path)
            return None
        
        remaining_slots = k - depth
        # Pruning:
        # 1. Even if we pick the largest remaining elements, can we reach target?
        # Max possible additional sum is sum of next `remaining_slots` elements starting from `start`.
        # Since list is sorted descending, divs[start], divs[start+1]... are the largest.
        # We need to pick `remaining_slots` elements from indices start to n-1.
        # The maximum sum is sum(divs[start : start + remaining_slots])
        
        if start + remaining_slots > n:
            return None
            
        max_add = sum(divs[start : start + remaining_slots])
        if current_sum + max_add < target:
            return None
            
        # 2. Even if we pick the smallest remaining elements, can we exceed target?
        # Min possible additional sum is sum of the smallest `remaining_slots` elements from start to n-1.
        # That is divs[n-1], divs[n-2]...
        # Specifically, sum(divs[n-remaining_slots : n])
        # But we are constrained to indices >= start.
        # If start <= n - remaining_slots, the smallest `remaining_slots` elements available are divs[n-remaining_slots : n].
        # Wait, if start > n - remaining_slots, we can't pick that many elements. We already handled that.
        
        min_add = sum(divs[n - remaining_slots : n])
        if current_sum + min_add > target:
            return None
            
        for i in range(start, n - remaining_slots + 1):
            val = divs[i]
            new_sum = current_sum + val
            
            # Pruning based on current val
            # If new_sum > target, since divs are positive and sorted desc, further values will be smaller, but wait.
            # divs is sorted descending. So val is decreasing.
            # If new_sum > target, we might still find a solution with smaller vals? No, because we need to add MORE positive numbers.
            # So if new_sum > target, we can't add any positive number to it to reach target (unless we subtract, which we don't).
            # Wait, the target is fixed. If current_sum + val > target, then current_sum + val + ... > target.
            # So we can break?
            # No, because we are picking from a set. We don't have to pick val. We can pick a smaller val.
            # But if val is too big, we skip it. We don't break because smaller vals might work.
            
            if new_sum > target:
                continue
                
            current_path.append(val)
            result = backtrack(i + 1, depth + 1, new_sum, current_path)
            current_path.pop()
            
            if result is not None:
                return result
                
        return None
    
    return backtrack(0, 0, 0, [])

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s safety margin for 60s limit
    
    # Check k=1, 2, 3, 4
    for k in range(1, 5):
        if time.time() > deadline:
            break
        sol = _find_solution_k(TARGET, _DIVS, k, deadline)
        if sol is not None:
            return sol
            
    # If not found, return the known 5-divisor solution
    # 2520 + 1680 + 560 + 210 + 9 = 4979
    # 2520, 1680, 560, 210, 9 are all divisors of 5040?
    # 5040 / 2520 = 2 (Yes)
    # 5040 / 1680 = 3 (Yes)
    # 5040 / 560 = 9 (Yes)
    # 5040 / 210 = 24 (Yes)
    # 5040 / 9 = 560 (Yes)
    # All distinct? Yes.
    
    return [2520, 1680, 560, 210, 9]
