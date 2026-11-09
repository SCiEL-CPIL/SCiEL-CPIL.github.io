import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = [L // i for i in range(1, n + 1)]
    
    def evaluate(coeffs):
        s = 0
        for i in range(n):
            s += coeffs[i] * weights[i]
        return s
    
    best_sum = None
    best_coeffs = None
    
    # Structural change: constructive search
    # Start from zero, add coefficients one by one to minimize |sum| while keeping sum != 0
    # Then refine with local search
    
    def local_search(coeffs, current_sum):
        """Improve a given solution using local search"""
        improved = True
        max_iters = 200
        iter_count = 0
        
        while improved and iter_count < max_iters:
            iter_count += 1
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        continue
                    
                    if abs(new_sum) < abs(current_sum):
                        coeffs[i] = new_val
                        current_sum = new_sum
                        improved = True
                        break
        
        return coeffs, current_sum
    
    # Phase 1: Constructive search with multiple random orderings
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random permutation of indices
        indices = list(range(n))
        random.shuffle(indices)
        
        coeffs = [0] * n
        current_sum = 0
        
        # Add coefficients one by one
        for idx in indices:
            if time.time() > deadline - 2.0:
                break
            
            best_val = 0
            best_abs = abs(current_sum) if current_sum != 0 else float('inf')
            
            for val in [-1, 0, 1]:
                new_sum = current_sum + val * weights[idx]
                if new_sum == 0:
                    continue
                if abs(new_sum) < best_abs:
                    best_abs = abs(new_sum)
                    best_val = val
            
            coeffs[idx] = best_val
            current_sum += best_val * weights[idx]
        
        # Phase 2: Local search refinement
        refined_coeffs, refined_sum = local_search(coeffs[:], current_sum)
        
        if refined_sum != 0:
            if best_sum is None or abs(refined_sum) < abs(best_sum):
                best_sum = refined_sum
                best_coeffs = refined_coeffs[:]
    
    # Phase 3: Additional random restarts with local search
    while time.time() < deadline:
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        s = evaluate(coeffs)
        
        if s == 0:
            # Find best neighbor
            best_neighbor_sum = None
            best_neighbor_coeffs = None
            for j in range(n):
                old_val = coeffs[j]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    ns = s + (new_val - old_val) * weights[j]
                    if ns != 0:
                        if best_neighbor_sum is None or abs(ns) < abs(best_neighbor_sum):
                            best_neighbor_sum = ns
                            best_neighbor_coeffs = coeffs[:]
                            best_neighbor_coeffs[j] = new_val
            
            if best_neighbor_coeffs:
                refined_coeffs, refined_sum = local_search(best_neighbor_coeffs, best_neighbor_sum)
                if refined_sum != 0:
                    if best_sum is None or abs(refined_sum) < abs(best_sum):
                        best_sum = refined_sum
                        best_coeffs = refined_coeffs[:]
        else:
            refined_coeffs, refined_sum = local_search(coeffs, s)
            if refined_sum != 0:
                if best_sum is None or abs(refined_sum) < abs(best_sum):
                    best_sum = refined_sum
                    best_coeffs = refined_coeffs[:]
    
    if best_coeffs is None:
        fallback = [0] * 30
        fallback[1] = 1
        fallback[2] = -1
        return fallback
    
    return best_coeffs
