import time
import numpy as np
from math import gcd, lcm
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Integer weights: w_i = L // (i+1) for coefficient c_i multiplying 1/(i+1)
    # Sum = sum(c_i * L/(i+1)) / L
    # Minimize |sum(c_i * w_i)| where w_i = L//(i+1)
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    def compute_value(coeffs):
        """Compute the integer numerator K = sum(c_i * w_i)"""
        return int(np.dot(coeffs, weights))
    
    def value_from_K(K):
        """Compute |K/L| as a float for comparison"""
        if K == 0:
            return float('inf')
        return abs(K) / L
    
    random.seed(42)
    np.random.seed(42)
    
    # Lattice-reduction-inspired approach:
    # Apply random orthogonal transformations to decorrelate weights,
    # then use greedy + local search on the transformed space
    
    best_coeffs = None
    best_K = None
    best_val = float('inf')
    
    def local_search(coeffs, K, max_iters=50000):
        """Local search: flip one coefficient at a time"""
        coeffs = coeffs.copy()
        K = int(K)
        val = value_from_K(K)
        
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            # Try all single flips, pick the best improving one
            best_delta = 0
            best_i = -1
            best_new_val = val
            
            # Sample a subset of positions for speed
            sample_size = min(n, 10)
            indices = random.sample(range(n), sample_size)
            
            for i in indices:
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta_K = (new_val - old_val) * weights[i]
                    new_K = K + delta_K
                    new_v = value_from_K(new_K)
                    if new_v < best_new_val:
                        best_new_val = new_v
                        best_i = i
                        best_delta = new_val - old_val
                        best_new_K = new_K
            
            if best_i >= 0 and best_delta != 0:
                coeffs[best_i] += best_delta
                K = best_new_K
                val = best_new_val
            else:
                # Stuck, try a random perturbation
                i = random.randint(0, n - 1)
                old_val = coeffs[i]
                new_val = random.choice([-1, 0, 1])
                while new_val == old_val:
                    new_val = random.choice([-1, 0, 1])
                coeffs[i] = new_val
                K += (new_val - old_val) * weights[i]
                val = value_from_K(K)
        
        return coeffs, K, val
    
    # Strategy 1: Greedy initialization with multiple random restarts
    # Use random linear combinations to find good starting points
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a random initial solution
        # Bias towards having more non-zero coefficients
        coeffs = np.zeros(n, dtype=np.int64)
        
        # Random subset of positions to be non-zero
        num_nonzero = random.randint(n // 2, n)
        nonzero_positions = random.sample(range(n), num_nonzero)
        for pos in nonzero_positions:
            coeffs[pos] = random.choice([-1, 1])
        
        K = compute_value(coeffs)
        
        if K == 0:
            # Try a small perturbation
            i = random.randint(0, n - 1)
            coeffs[i] = random.choice([-1, 0, 1])
            K = compute_value(coeffs)
            if K == 0:
                coeffs[random.randint(0, n-1)] = 1
                K = compute_value(coeffs)
        
        val = value_from_K(K)
        
        if val < best_val:
            best_val = val
            best_K = K
            best_coeffs = coeffs.copy()
        
        # Run local search from this starting point
        if time.time() < deadline - 5.0:
            coeffs_ls, K_ls, val_ls = local_search(coeffs, K, max_iters=20000)
            
            if val_ls < best_val:
                best_val = val_ls
                best_K = K_ls
                best_coeffs = coeffs_ls.copy()
        
        # Also try a greedy construction
        if time.time() < deadline - 5.0:
            # Greedy: add terms one by one to minimize running sum
            g_coeffs = np.zeros(n, dtype=np.int64)
            g_K = 0
            
            # Random order of terms
            order = list(range(n))
            random.shuffle(order)
            
            for pos in order:
                # Choose c[pos] to minimize |g_K + c[pos]*weights[pos]|
                best_c = 0
                best_abs = abs(g_K)
                for c in [-1, 0, 1]:
                    new_K = g_K + c * weights[pos]
                    new_abs = abs(new_K)
                    if new_abs < best_abs:
                        best_abs = new_abs
                        best_c = c
                
                g_coeffs[pos] = best_c
                g_K += best_c * weights[pos]
            
            if g_K == 0:
                # Perturb
                i = random.randint(0, n-1)
                g_coeffs[i] = random.choice([-1, 1])
                g_K = compute_value(g_coeffs)
            
            g_val = value_from_K(g_K)
            if g_val < best_val:
                best_val = g_val
                best_K = g_K
                best_coeffs = g_coeffs.copy()
            
            # Local search from greedy solution
            if time.time() < deadline - 5.0:
                coeffs_ls, K_ls, val_ls = local_search(g_coeffs, g_K, max_iters=20000)
                if val_ls < best_val:
                    best_val = val_ls
                    best_K = K_ls
                    best_coeffs = coeffs_ls.copy()
    
    # Final intensive local search from best solution
    if time.time() < deadline - 3.0 and best_coeffs is not None:
        coeffs_final, K_final, val_final = local_search(best_coeffs, best_K, max_iters=100000)
        if val_final < best_val:
            best_val = val_final
            best_K = K_final
            best_coeffs = coeffs_final
    
    # Verify best solution
    if best_coeffs is None:
        # Fallback
        fallback = [0] * 28 + [1, -1]
        return fallback
    
    final_K = compute_value(best_coeffs)
    if final_K == 0:
        # Fallback: try a simple nonzero solution
        fallback = [0] * 28 + [1, -1]
        return fallback
    
    return list(best_coeffs)
