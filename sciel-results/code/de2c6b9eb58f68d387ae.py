import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    size_u = n * m  # 12
    size_v = m * p  # 20
    size_w = p * n  # 15
    total_size = size_u + size_v + size_w  # 47
    
    def verify_decomposition(U, V, W, R):
        """Check if decomposition is exact using Fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    term = Fraction(0)
                    for r in range(R):
                        term += Fraction(U[u_idx][r]) * Fraction(V[v_idx][r]) * Fraction(W[w_idx][r])
                    if term != Fraction(1):
                        return False
        return True
    
    def compute_residual(U, V, W, R):
        """Compute the residual tensor for current decomposition."""
        residual = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    term = 0.0
                    for r in range(R):
                        term += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    residual[i, j, k] = 1.0 - term
        return residual
    
    def greedy_decomposition(max_rank, value_set, max_iterations=100):
        """
        Randomized greedy algorithm to find tensor decomposition.
        Iteratively selects rank-1 tensors that maximize residual reduction.
        """
        if time.time() > deadline:
            return None
        
        # Initialize with zero decomposition
        U = [[0] * max_rank for _ in range(size_u)]
        V = [[0] * max_rank for _ in range(size_v)]
        W = [[0] * max_rank for _ in range(size_w)]
        
        current_rank = 0
        best_solution = None
        best_rank = max_rank + 1
        
        # Target tensor is all ones (for matrix multiplication)
        target = np.ones((n, m, p), dtype=np.float64)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            if current_rank >= max_rank:
                break
            
            # Compute current residual
            residual = compute_residual(U, V, W, current_rank)
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < 1e-10:
                # Found exact solution
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_solution = {
                        "U": [row[:current_rank] for row in U],
                        "V": [row[:current_rank] for row in V],
                        "W": [row[:current_rank] for row in W]
                    }
                break
            
            # Try to find best rank-1 tensor to add
            best_improvement = 0
            best_u_col = None
            best_v_col = None
            best_w_col = None
            
            # Sample random rank-1 tensors
            num_samples = 50
            for _ in range(num_samples):
                if time.time() > deadline:
                    break
                
                u_col = [random.choice(value_set) for _ in range(size_u)]
                v_col = [random.choice(value_set) for _ in range(size_v)]
                w_col = [random.choice(value_set) for _ in range(size_w)]
                
                # Compute improvement: ||residual - u⊗v⊗w||^2 reduction
                # = 2 * <residual, u⊗v⊗w> - ||u⊗v⊗w||^2
                inner = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            inner += residual[i, j, k] * u_col[u_idx] * v_col[v_idx] * w_col[w_idx]
                
                norm_sq = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = u_col[u_idx] * v_col[v_idx] * w_col[w_idx]
                            norm_sq += val * val
                
                improvement = 2 * inner - norm_sq
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_u_col = u_col
                    best_v_col = v_col
                    best_w_col = w_col
            
            if best_improvement <= 0 or best_u_col is None:
                # No improvement possible, break
                break
            
            # Add the best rank-1 tensor
            U[current_rank] = best_u_col
            V[current_rank] = best_v_col
            W[current_rank] = best_w_col
            current_rank += 1
            
            # Check if we have an exact solution
            if current_rank <= max_rank:
                residual = compute_residual(U, V, W, current_rank)
                if np.linalg.norm(residual) < 1e-10:
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_solution = {
                            "U": [row[:current_rank] for row in U],
                            "V": [row[:current_rank] for row in V],
                            "W": [row[:current_rank] for row in W]
                        }
                    break
        
        if best_solution is not None:
            # Verify with exact arithmetic
            if verify_decomposition(best_solution["U"], best_solution["V"], best_solution["W"], best_rank):
                return best_solution
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Known best for 3x4x5 is around 30-35
    
    candidate_values = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
    candidate_values_float = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for R in range(20, 40):
        if time.time() > deadline:
            break
        
        # Try greedy decomposition
        result = greedy_decomposition(R, candidate_values_float, max_iterations=50)
        if result:
            return result
        
        # If we have time, try more iterations
        if time.time() < deadline - 30:
            result = greedy_decomposition(R, candidate_values_float, max_iterations=100)
            if result:
                return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
