import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try limits in descending order
    # M=10^9 -> P~43k -> P^2~1.8e9 -> ~14GB (int64). Risky.
    # M=5e8 -> P~30k -> P^2~0.9e9 -> ~7GB (int64). Safe.
    # M=2e8 -> P~20k -> P^2~0.4e9 -> ~3GB (int64). Safe.
    
    limits_to_try = [1000000000, 500000000, 200000000, 100000000]
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    
    for M in limits_to_try:
        if time.time() > deadline:
            break
            
        try:
            # Generate powerful numbers
            P = generate_powerful(M)
            if len(P) < 100:
                continue
                
            P_np = np.array(P, dtype=np.int64)
            L = len(P_np)
            
            # Check memory estimate
            # 8 bytes per element
            mem_estimate = L * L * 8
            if mem_estimate > 15 * 1024**3: # 15 GB
                continue
                
            # Compute sums
            sums = np.add.outer(P_np, P_np)
            
            # Count
            # Max sum is 2*M
            counts = np.bincount(sums.ravel(), minlength=2*M + 1)
            
            # Find best N
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            
            # Recover pairs
            pairs = []
            P_set = set(P)
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            if len(pairs) > len(best_result['pairs']):
                best_result = {'N': best_N, 'pairs': pairs}
                
            # If we succeeded with the largest limit, we can return immediately?
            # Usually larger M yields better results.
            # But if 10^9 failed due to time (not memory), we try smaller.
            # If 10^9 succeeded, it's likely the best.
            # However, generating 10^9 might take time.
            # Let's just return the best found so far if we hit the deadline.
            # If we finished the loop, return best.
            
            # Optimization: If we processed a large M, break to save time for recovery?
            # No, recovery is fast.
            # But generating 10^9 might take 5-10s.
            # Outer product 1.8e9 elements takes time.
            # If we do 10^9, we might not have time for 5e8.
            # So we should only try one limit that fits well.
            
            # Let's return the result of the first successful large M.
            return best_result
            
        except MemoryError:
            continue
        except Exception:
            continue

    return best_result

def generate_powerful(limit):
    # Efficient generation
    max_v = int(limit ** (1/3)) + 2
    is_sq_free = [True] * (max_v + 1)
    if max_v >= 0: is_sq_free[0] = False
    if max_v >= 1: is_sq_free[1] = True
    for p in range(2, int(math.isqrt(max_v)) + 1):
        if is_sq_free[p]:
            p2 = p * p
            for j in range(p2, max_v + 1, p2):
                is_sq_free[j] = False
                
    square_free_v = [v for v in range(1, max_v + 1) if is_sq_free[v]]
    
    powerful_set = set()
    for v in square_free_v:
        v3 = v ** 3
        if v3 > limit:
            continue
        max_u = int(math.isqrt(limit // v3))
        for u in range(1, max_u + 1):
            x = u * u * v3
            if x <= limit:
                powerful_set.add(x)
                
    return sorted(powerful_set)
