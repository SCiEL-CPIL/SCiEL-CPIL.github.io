import math
import time
import sys
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
import scipy.sparse as sp

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    N = 256
    
    # Always include 1
    # Solve for 2..N
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # 1. Try MILP for Exact Solution
    try:
        # Build conflict constraints
        # x_i + x_j <= 1 for gcd(i, j) > 1
        row_idx = []
        col_idx = []
        data = []
        
        constraint_id = 0
        # Optimization: Only check pairs where gcd > 1
        # We can speed this up by precomputing smallest prime factor or just using gcd
        # 255 * 254 / 2 = ~32k checks. Very fast.
        
        for i in range(n):
            ci = candidates[i]
            for j in range(i + 1, n):
                cj = candidates[j]
                if math.gcd(ci, cj) > 1:
                    row_idx.append(constraint_id)
                    col_idx.append(i)
                    data.append(1)
                    
                    row_idx.append(constraint_id)
                    col_idx.append(j)
                    data.append(1)
                    
                    constraint_id += 1
                    
                    # Time check during construction (should be negligible)
                    if time.time() > deadline - 10.0:
                        raise TimeoutError("Construction taking too long")
        
        if constraint_id == 0:
            # No conflicts, take all
            return [1] + candidates

        A_ub = sp.csr_matrix((data, (row_idx, col_idx)), shape=(constraint_id, n))
        
        # Objective: Maximize sum(i * x_i) -> Minimize -sum(i * x_i)
        c = np.array([-float(ci) for ci in candidates])
        
        # Constraints: A_ub @ x <= 1
        lincons = LinearConstraint(A_ub, lb=0, ub=1)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x must be integer
        integrality = np.ones(n)
        
        # Solve
        res = milp(c=c, constraints=lincons, integrality=integrality, bounds=bounds)
        
        remaining = deadline - time.time()
        if remaining < 2.0:
            return _fallback_construct(N, max(0.5, remaining))
            
        if not res.success:
            # If solver fails, fallback
            return _fallback_construct(N, max(0.5, deadline - time.time()))
            
        x = res.x
        # Round to nearest integer to handle floating point noise
        x = np.round(x).astype(int)
        
        # Construct result
        result = [1]  # Always include 1
        
        for i in range(n):
            if x[i] == 1:
                result.append(candidates[i])
        
        # Sanity check: Verify pairwise coprimality
        # If invalid, fallback
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) != 1:
                    return _fallback_construct(N, max(0.5, deadline - time.time()))
                    
        # Final polish: local search to improve the MILP solution
        result = _polish_local_search(result, N, deadline - time.time())
        
        # Ensure types are Python int
        result = [int(r) for r in result]
        return result
        
    except Exception as e:
        # In case of any error, fallback
        return _fallback_construct(N, max(0.5, deadline - time.time()))

def _polish_local_search(selected_list, N, time_budget):
    """
    Final local-search polish: try to improve the set by adding elements
    or performing beneficial swaps.
    """
    if time_budget < 0.5:
        return list(selected_list)
    
    start = time.time()
    deadline = start + time_budget
    
    selected = set(selected_list)
    current_sum = sum(selected)
    
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # First, try to add any element that fits
    for n in range(1, N + 1):
        if time.time() > deadline - 0.5:
            break
        if n not in selected:
            if all(_gcd(n, s) == 1 for s in selected):
                selected.add(n)
                current_sum += n
    
    # Then try swaps: remove one, add one or two
    improved = True
    while improved and time.time() < deadline - 0.5:
        improved = False
        
        # Try single swaps
        selected_list = list(selected)
        for n_out in selected_list:
            if time.time() > deadline - 0.5:
                break
            if n_out == 1:
                continue
            rest = selected - {n_out}
            # Try to find a single n_in that improves
            for n_in in range(N, n_out, -1):
                if n_in in selected:
                    continue
                if all(_gcd(n_in, s) == 1 for s in rest):
                    if n_in > n_out:
                        selected.remove(n_out)
                        selected.add(n_in)
                        current_sum += n_in - n_out
                        improved = True
                        break
            if improved:
                break
        
        # Try double swaps: remove two, add two
        if not improved and time.time() < deadline - 1.0:
            selected_list = list(selected)
            for i in range(len(selected_list)):
                if time.time() > deadline - 0.5:
                    break
                n_out1 = selected_list[i]
                if n_out1 == 1:
                    continue
                for j in range(i + 1, len(selected_list)):
                    if time.time() > deadline - 0.5:
                        break
                    n_out2 = selected_list[j]
                    rest = selected - {n_out1, n_out2}
                    # Try to find two n_in's that improve
                    best_pair = None
                    for n_in1 in range(N, max(n_out1, n_out2), -1):
                        if n_in1 in selected:
                            continue
                        if not all(_gcd(n_in1, s) == 1 for s in rest):
                            continue
                        for n_in2 in range(N, n_in1, -1):
                            if n_in2 in selected:
                                continue
                            if not all(_gcd(n_in2, s) == 1 for s in rest):
                                continue
                            if _gcd(n_in1, n_in2) != 1:
                                continue
                            gain = (n_in1 + n_in2) - (n_out1 + n_out2)
                            if gain > 0:
                                best_pair = (n_in1, n_in2, gain)
                                break
                        if best_pair:
                            break
                    if best_pair:
                        selected.remove(n_out1)
                        selected.remove(n_out2)
                        selected.add(best_pair[0])
                        selected.add(best_pair[1])
                        current_sum += best_pair[2]
                        improved = True
                        break
                if improved:
                    break
    
    return list(selected)


def _fallback_construct(N, time_budget):
    """
    Fallback heuristic if ILP fails or times out.
    Uses Greedy + Local Search with random restarts.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    if deadline < 1.0:
        # Very little time left, return a simple valid set
        return [1] + [p for p in range(2, N+1) if is_prime(p)]
        
    # Precompute primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def is_coprime_set(nums):
        # nums is a set
        if len(nums) < 2:
            return True
        lst = list(nums)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    best_result = [1]
    best_sum = 1
    
    # Multi-start Greedy
    for seed in range(100):
        if time.time() > deadline - 1.0:
            break
            
        random.seed(seed)
        
        # Strategy 1: Greedy by descending value
        # Strategy 2: Randomized Greedy
        
        if seed % 2 == 0:
            # Deterministic Greedy
            selected = {1}
            for n in range(N, 1, -1):
                if all(gcd(n, s) == 1 for s in selected):
                    selected.add(n)
        else:
            # Randomized Greedy
            selected = {1}
            candidates = list(range(N, 1, -1))
            random.shuffle(candidates)
            for n in candidates:
                if all(gcd(n, s) == 1 for s in selected):
                    selected.add(n)
        
        # Local Search
        current_sum = sum(selected)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Try to add a number
            for n in range(1, N + 1):
                if n not in selected:
                    if all(gcd(n, s) == 1 for s in selected):
                        selected.add(n)
                        current_sum += n
                        improved = True
                        break # Restart inner loop with new state
            
            if not improved:
                # Try to swap
                # To make it faster, only try swapping if we can gain
                # We look for n_out in selected, n_in not in selected
                # such that n_in > n_out and {selected - {n_out}} U {n_in} is coprime
                # This is expensive O(N^2 * K).
                # Optimization: Only consider n_in that are coprime to (selected - {n_out})
                
                selected_list = list(selected)
                for n_out in selected_list:
                    if n_out == 1:
                        continue # Don't remove 1
                    # Check if we can find a better n_in
                    # n_in must be coprime to all s in selected where s != n_out
                    rest = selected - {n_out}
                    for n_in in range(N, n_out, -1): # Only try larger numbers
                        if n_in in selected:
                            continue
                        if all(gcd(n_in, s) == 1 for s in rest):
                            # Valid swap
                            selected.remove(n_out)
                            selected.add(n_in)
                            current_sum += n_in - n_out
                            improved = True
                            break
                    if improved:
                        break
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = list(selected)
            
    # Final validation
    if not is_coprime_set(set(best_result)):
        # Emergency fallback: Just primes
        return [1] + primes
    
    # Apply polish
    remaining = deadline - time.time()
    if remaining > 1.0:
        best_result = _polish_local_search(best_result, N, remaining)
    
    return [int(x) for x in best_result]

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    res = construct()
    print(f"Sum: {sum(res)}, Count: {len(res)}")
