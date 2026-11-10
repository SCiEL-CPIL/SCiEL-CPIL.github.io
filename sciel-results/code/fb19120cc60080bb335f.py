# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85  # Leave 5s margin for safety
    best_set = []
    
    # Helper to compute dp bitsets for a list of numbers
    def compute_dp(nums):
        n = len(nums)
        max_sum = sum(nums)
        # dp[k] is an integer where bit s is 1 if there's a subset of size k summing to s
        dp = [0] * (n + 1)
        dp[0] = 1  # bit 0 is 1
        for x in nums:
            # Update in reverse order of k to avoid using x multiple times
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        return dp

    def is_valid_addition(A, x):
        """
        Check if x can be added to A without violating constraints.
        A is a list of ints.
        Returns True if valid.
        """
        if x in A:
            return False
            
        n = len(A)
        if n == 0:
            return True
            
        # 1. Check if x is the mean of any subset of A of size k >= 2
        # mean(T) = x => sum(T) = k * x
        dp_A = compute_dp(A)
        for k in range(2, n + 1):
            target_sum = k * x
            if target_sum > sum(A):
                break
            if (dp_A[k] >> target_sum) & 1:
                return False
                
        # 2. Check if any a in A is the mean of a subset of A U {x} containing x
        # Subset T U {x}, T subset of A \ {a}
        # sum(T) + x = (|T| + 1) * a
        # sum(T) = (|T| + 1) * a - x
        for a in A:
            A_excl_a = [y for y in A if y != a]
            if not A_excl_a:
                continue
            dp_excl = compute_dp(A_excl_a)
            m = len(A_excl_a)
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                if target_sum > sum(A_excl_a):
                    break # since j increases, target_sum increases? 
                         # (j+1)*a - x. Yes, increases with j.
                if (dp_excl[j] >> target_sum) & 1:
                    return False
                    
        return True

    def greedy_construct(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= 48:
                    break
        return selected

    # Strategy 1: Descending order (large numbers are "safer" against being means of smaller subsets? 
    # Actually, large numbers are less likely to be the average of two others if the others are also large and distinct?)
    # Let's try a few different orderings.
    
    orders = []
    
    # 1. Descending
    orders.append(list(range(512, 0, -1)))
    
    # 2. Ascending
    orders.append(list(range(1, 513)))
    
    # 3. Random
    for _ in range(10):
        r = list(range(1, 513))
        random.shuffle(r)
        orders.append(r)
        
    # 4. "Sparse" heuristic: Try to pick numbers that are far apart.
    # Just use random for now, but maybe weighted?
    
    best_size = 0
    best_result = []
    
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        res = greedy_construct(order)
        if len(res) > best_size:
            best_size = len(res)
            best_result = res
            
        # If we found a large set, try to improve it with local search?
        # For now, just keep the best greedy result.
        
    # If we have time, try local search on the best result found
    if best_size > 0 and time.time() < deadline - 10:
        current_best = best_result[:]
        # Try to swap elements to allow adding more?
        # Or just try adding more random elements?
        
        # Let's try to add more elements to current_best using random candidates
        unused = [x for x in range(1, 513) if x not in current_best]
        random.shuffle(unused)
        
        for x in unused:
            if time.time() > deadline - 2:
                break
            if len(current_best) >= 48:
                break
            if is_valid_addition(current_best, x):
                current_best.append(x)
        
        if len(current_best) > best_size:
            best_result = current_best
            best_size = len(current_best)

    return best_result
# EVOLVE-BLOCK-END
