"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 50 constructive decomposition
    def get_rank_50_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            
            # M2
            U[(i0+1)*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            col += 1
            
            # M5
            U[i0*m + j0][col] += 1
            U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            col += 1
            
            # M6
            U[(i0+1)*m + j0][col] += 1
            U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            
            # M7
            U[i0*m + (j0+1)][col] += 1
            U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        # Part 1: 2x4x4 (k=0..3)
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        
        # Part 2: 2x4x2 (k=4..5)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        # Part 3: 2x4x1 (k=6)
        for i in range(2):
            for j in range(4):
                U[i*m + j][col] = 1
                V[j*p + 6][col] = 1
                W[6*n + i][col] = 1
                col += 1
                
        return {"U": U, "V": V, "W": W}

    # Target tensor T[i, j, k] = 1
    # Residual function for optimization
    def residuals(x, rank):
        # x is flattened [U, V, W]
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        # Compute tensor T'
        # T'[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Using einsum for efficiency
        T_prime = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        # Target is all ones
        return (T_prime - 1.0).flatten()

    # Check if solution is valid and has integer/half-integer entries
    def is_valid_solution(U, V, W, tol=1e-5):
        # Check tensor error
        T_prime = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        error = np.max(np.abs(T_prime - 1.0))
        if error > tol:
            return False
        
        # Check entries are close to integers or half-integers
        # Multiply by 2 and check if close to integers
        for M in [U, V, W]:
            M2 = M * 2
            diff = np.abs(M2 - np.round(M2))
            if np.max(diff) > 1e-4:
                return False
        return True

    # Try to find a lower rank solution
    best_solution = get_rank_50_solution()
    best_rank = 50
    
    # Search ranks from 49 down to 45
    for rank in range(49, 44, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W
        
        # Try a few random restarts
        num_restarts = 5
        found = False
        
        for restart in range(num_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Random initialization
            x0 = np.random.randn(total_size) * 0.1
            
            try:
                # Use least_squares with a time limit per call if possible, 
                # but scipy doesn't have a built-in time limit. 
                # We rely on the outer loop deadline check.
                # To be safe, we can limit the number of iterations.
                result = least_squares(
                    residuals, 
                    x0, 
                    args=(rank,), 
                    method='lm', 
                    max_nfev=5000, 
                    xtol=1e-8, 
                    ftol=1e-8
                )
                
                if result.success or result.cost < 1e-6:
                    x_opt = result.x
                    U_flat = x_opt[:size_U]
                    V_flat = x_opt[size_U:size_U + size_V]
                    W_flat = x_opt[size_U + size_V:size_U + size_V + size_W]
                    
                    U = U_flat.reshape(n, m, rank)
                    V = V_flat.reshape(m, p, rank)
                    W = W_flat.reshape(p, n, rank)
                    
                    if is_valid_solution(U, V, W):
                        # Convert to list of lists with rounded values
                        U_list = [[0] * rank for _ in range(n * m)]
                        V_list = [[0] * rank for _ in range(m * p)]
                        W_list = [[0] * rank for _ in range(p * n)]
                        
                        for i in range(n):
                            for j in range(m):
                                for r in range(rank):
                                    U_list[i*m + j][r] = round(U[i, j, r] * 2) / 2
                        for j in range(m):
                            for k in range(p):
                                for r in range(rank):
                                    V_list[j*p + k][r] = round(V[j, k, r] * 2) / 2
                        for k in range(p):
                            for i in range(n):
                                for r in range(rank):
                                    W_list[k*n + i][r] = round(W[k, i, r] * 2) / 2
                                    
                        best_solution = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break # Break restart loop for this rank
                        
            except Exception as e:
                pass # Continue to next restart or rank

        if found:
            break # Found a solution for this rank, move to next lower rank

    return best_solution
# EVOLVE-BLOCK-END
