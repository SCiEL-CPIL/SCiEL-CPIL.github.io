import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    N = 256
    
    # Precompute suffix sums of reciprocals for upper bound pruning
    # suffix_sum[i] = sum(1/j for j in range(i, N+1))
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / i
    
    best_score = 0.0
    best_set = set()
    
    # We use a recursive B&B approach, but implemented iteratively or with deep recursion limit.
    # Python recursion limit might be an issue for depth 256, but we can increase it or use an explicit stack.
    # However, B&B prunes heavily. Let's try recursion first with a high limit.
    sys.setrecursionlimit(3000)
    
    def is_valid_addition(S, x):
        # Check if adding x to S creates a 3-AP
        # 1. x as middle: a + c = 2x => c = 2x - a. If c in S, conflict.
        # 2. x as endpoint: a + x = 2c => c = (a+x)/2.
        #    If (a+x) even and c in S, conflict.
        
        for a in S:
            # x is middle
            c = 2 * x - a
            if c in S:
                return False
            # x is endpoint
            s = a + x
            if s % 2 == 0:
                c = s // 2
                if c in S:
                    return False
        return True

    def branch_and_bound(idx, current_sum, S):
        nonlocal best_score, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Update best if current sum is better
        if current_sum > best_score:
            best_score = current_sum
            best_set = S.copy()
            
        # Pruning: if current_sum + max possible remaining sum <= best_score, stop
        if idx > N:
            return
            
        if current_sum + suffix_sum[idx] <= best_score:
            return
            
        # Option 1: Include idx
        # Check validity
        if is_valid_addition(S, idx):
            S.add(idx)
            branch_and_bound(idx + 1, current_sum + 1.0 / idx, S)
            S.remove(idx)
            
        # Option 2: Exclude idx
        branch_and_bound(idx + 1, current_sum, S)
        
    # Initial call
    # We process numbers in order 1 to 256.
    # This order is optimal for B&B with reciprocal weights because it finds high-value solutions early.
    S = set()
    branch_and_bound(1, 0.0, S)
    
    # If time remains, do a local search polish on the best set found
    if time.time() < deadline - 5.0:
        S_best = best_set.copy()
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to swap x_out (in S) with x_in (not in S) where x_in < x_out
            x_outs = sorted([x for x in S_best], reverse=True)
            x_ins = sorted([x for x in range(1, N + 1) if x not in S_best])
            
            for x_out in x_outs:
                if time.time() > deadline - 5.0:
                    break
                S_temp = S_best.copy()
                S_temp.remove(x_out)
                for x_in in x_ins:
                    if x_in >= x_out:
                        break
                    if is_valid_addition(S_temp, x_in):
                        S_best = S_temp
                        S_best.add(x_in)
                        improved = True
                        break
                if improved:
                    break
                    
        # Final check
        if S_best:
            final_score = sum(1.0 / x for x in S_best)
            if final_score > best_score:
                best_score = final_score
                best_set = S_best

    return list(best_set)
