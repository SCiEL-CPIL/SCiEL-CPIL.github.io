"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 3-AP-free subset of [1, 256] maximizing the sum of reciprocals.
    Strategy:
    1. Start with a greedy construction prioritizing small elements (high reciprocal value).
    2. Use local search with 1-swap and 2-swap moves to escape local optima.
    3. Final polish pass for single additions.
    4. Deadline guard to ensure we finish within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 60s
    
    N = 256
    MAX_VAL = N
    
    # Fast AP check: given a boolean array `in_set` and candidate x,
    # check if adding x creates a 3-AP. O(N) worst case but with early exit.
    def is_valid_addition_fast(in_set, x):
        """Check if x can be added to the set represented by in_set without creating a 3-AP."""
        # Case 1: x is the first term (smallest). Need b, c in set with 2b = x+c, x < b < c.
        for b in range(x + 1, MAX_VAL + 1):
            if in_set[b]:
                c = 2 * b - x
                if c <= MAX_VAL and in_set[c]:
                    return False
        
        # Case 2: x is the middle term. Need a, c in set with 2x = a+c, a < x < c.
        for a in range(1, x):
            if in_set[a]:
                c = 2 * x - a
                if c <= MAX_VAL and in_set[c]:
                    return False
        
        # Case 3: x is the last term (largest). Need a, b in set with 2b = a+x, a < b < x.
        for b in range(1, x):
            if in_set[b]:
                a = 2 * b - x
                if a >= 1 and in_set[a]:
                    return False
        
        return True
    
    # Initial construction: Greedy from smallest to largest (prioritize high reciprocal value)
    in_set = [False] * (MAX_VAL + 1)
    S = set()
    for x in range(1, MAX_VAL + 1):
        if is_valid_addition_fast(in_set, x):
            in_set[x] = True
            S.add(x)
    
    current_score = sum(1.0 / x for x in S)
    best_S = S.copy()
    best_score = current_score
    best_in_set = in_set[:]
    
    # Local Search with fast checks
    # We want to remove large elements and add small elements to improve reciprocal sum.
    # A swap: remove x from S, add y to S where y < x (so 1/y > 1/x).
    # We need to check that S \ {x} ∪ {y} is still 3-AP-free.
    
    def try_swap(S, in_set, x, y):
        """
        Try to swap x (remove) with y (add). Returns (success, new_S, new_in_set, score_delta).
        x must be in S, y must not be in S, and y < x for improvement.
        """
        if x not in S or y in S or y >= x:
            return False, S, in_set, 0.0
        
        # Remove x, add y
        new_S = S - {x}
        new_in_set = in_set[:]
        new_in_set[x] = False
        new_in_set[y] = True
        
        # Check if new set is valid: only need to check APs involving y
        # (removing x can't create APs, adding y is the only risk)
        if not is_valid_addition_fast(new_in_set, y):
            return False, S, in_set, 0.0
        
        score_delta = 1.0 / y - 1.0 / x
        return True, new_S, new_in_set, score_delta
    
    def try_2swap(S, in_set, x1, x2, y1, y2):
        """
        Try to swap out x1, x2 and swap in y1, y2.
        Returns (success, new_S, new_in_set, score_delta).
        """
        if x1 not in S or x2 not in S or y1 in S or y2 in S:
            return False, S, in_set, 0.0
        if y1 >= x1 or y2 >= x2:
            return False, S, in_set, 0.0
        if y1 == y2:
            return False, S, in_set, 0.0
        
        # Remove x1, x2, add y1, y2
        new_S = S - {x1, x2}
        new_in_set = in_set[:]
        new_in_set[x1] = False
        new_in_set[x2] = False
        new_in_set[y1] = True
        new_in_set[y2] = True
        
        # Check if new set is valid: only need to check APs involving y1 and y2
        if not is_valid_addition_fast(new_in_set, y1):
            return False, S, in_set, 0.0
        if not is_valid_addition_fast(new_in_set, y2):
            return False, S, in_set, 0.0
        
        score_delta = (1.0 / y1 + 1.0 / y2) - (1.0 / x1 + 1.0 / x2)
        return True, new_S, new_in_set, score_delta
    
    # Iterate: repeatedly find and apply improving swaps
    max_iterations = 1000000  # Safety cap
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        improved = False
        
        # Get sorted lists
        S_list = sorted(S, reverse=True)  # Largest first (candidates for removal)
        not_in_S = sorted([i for i in range(1, MAX_VAL + 1) if i not in S])  # Smallest first (candidates for addition)
        
        # Try all possible 1-swaps: remove x (large), add y (small)
        for x in S_list:
            if time.time() > deadline:
                break
            if x not in S:
                continue
            
            # Try to add a smaller y
            for y in not_in_S:
                if time.time() > deadline:
                    break
                if y >= x:
                    break  # Only want y < x for improvement
                if y in S:
                    continue
                
                success, new_S, new_in_set, score_delta = try_swap(S, in_set, x, y)
                if success:
                    # Apply the swap
                    S = new_S
                    in_set = new_in_set
                    current_score += score_delta
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
                        best_in_set = in_set[:]
                    
                    # After a swap, restart the search from the top
                    break
            
            if improved:
                break
        
        if not improved:
            # No more improving 1-swaps; try 2-swaps to escape local optima
            any_2swap = False
            S_list_sorted = sorted(S, reverse=True)
            not_in_S_sorted = sorted([i for i in range(1, MAX_VAL + 1) if i not in S])
            
            for i, x1 in enumerate(S_list_sorted):
                if time.time() > deadline:
                    break
                if x1 not in S:
                    continue
                
                for j in range(i + 1, len(S_list_sorted)):
                    if time.time() > deadline:
                        break
                    x2 = S_list_sorted[j]
                    if x2 not in S:
                        continue
                    
                    for y1 in not_in_S_sorted:
                        if time.time() > deadline:
                            break
                        if y1 >= x1:
                            break
                        if y1 in S:
                            continue
                        
                        for y2 in not_in_S_sorted:
                            if time.time() > deadline:
                                break
                            if y2 >= x2:
                                break
                            if y2 in S or y2 == y1:
                                continue
                            
                            success, new_S, new_in_set, score_delta = try_2swap(S, in_set, x1, x2, y1, y2)
                            if success and score_delta > 0:
                                # Apply the 2-swap
                                S = new_S
                                in_set = new_in_set
                                current_score += score_delta
                                any_2swap = True
                                
                                if current_score > best_score:
                                    best_score = current_score
                                    best_S = S.copy()
                                    best_in_set = in_set[:]
                                
                                break
                        if any_2swap:
                            break
                    if any_2swap:
                        break
                if any_2swap:
                    break
            
            if not any_2swap:
                # No more 2-swaps; try adding elements without removing
                any_added = False
                for y in range(1, MAX_VAL + 1):
                    if time.time() > deadline:
                        break
                    if y not in S:
                        if is_valid_addition_fast(in_set, y):
                            S.add(y)
                            in_set[y] = True
                            current_score += 1.0 / y
                            any_added = True
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_S = S.copy()
                                best_in_set = in_set[:]
                
                if not any_added:
                    # No more additions possible either; we're at a local optimum
                    break
    
    return list(best_S)

# EVOLVE-BLOCK-END
