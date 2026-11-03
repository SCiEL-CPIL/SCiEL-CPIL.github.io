"""Program for exact tensor decomposition of matrix multiplication."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115s wall-clock limit (5s safety margin)
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact using Fraction for exact arithmetic."""
        num_cols_U = len(U[0]) if U else 0
        num_cols_V = len(V[0]) if V else 0
        num_cols_W = len(W[0]) if W else 0
        
        if num_cols_U != num_cols_V or num_cols_V != num_cols_W:
            return False
        
        rank = num_cols_U
        
        # Convert to Fraction for exact verification
        U_frac = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[j][r]) for r in range(rank)] for j in range(m*p)]
        W_frac = [[Fraction(W[k][r]) for r in range(rank)] for k in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                    if s != 1:
                        return False
        return True
    
    def schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def try_random_decomposition(target_rank, n, m, p, max_attempts=1000):
        """
        Try to find a rank-target_rank decomposition using random initialization.
        For fixed U and V, solve for W using linear algebra.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Randomly initialize U and V with small integers/half-integers
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(m * p)]
            
            # For each (i, k), solve for W[k*n+i, :]
            # The system is: for each j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system in W[k*n+i, :]
            
            W = [[0] * target_rank for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    c = k * n + i
                    # Build the m x target_rank matrix A_c where A_c[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A_c = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            A_c[j, r] = U[i * m + j][r] * V[j * p + k][r]
                    
                    # We need A_c @ W[c, :]^T = [1, 1, 1]^T
                    # This is an underdetermined system if target_rank > m
                    # Use least squares to find a solution, then check if it's exact
                    
                    # Actually, we need an exact solution. Let's use the null space approach.
                    # Find a particular solution and add null space components.
                    
                    # For simplicity, let's try to find a solution using the pseudoinverse
                    # and then check if it's exact
                    try:
                        # Use numpy to solve
                        b_vec = np.ones(m)
                        # Check if the system is consistent
                        # Compute the rank of A_c and [A_c | b_vec]
                        rank_A = np.linalg.matrix_rank(A_c)
                        rank_Ab = np.linalg.matrix_rank(np.column_stack([A_c, b_vec]))
                        
                        if rank_A != rank_Ab:
                            valid = False
                            break
                        
                        # Find a solution
                        # Use the null space to find all solutions
                        # For now, use the minimum norm solution
                        W_c = np.linalg.lstsq(A_c, b_vec, rcond=None)[0]
                        
                        # Check if this is an exact solution
                        residual = np.abs(A_c @ W_c - b_vec)
                        if np.max(residual) > 1e-10:
                            valid = False
                            break
                        
                        # Convert to fractions if possible, otherwise keep as floats
                        # For now, store as floats
                        W[c] = [float(x) for x in W_c]
                    except:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Verify the decomposition
            if verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find a lower-rank decomposition
    # Start from rank 21 (theoretical lower bound) up to 41
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # First, try a few promising ranks
    for target_rank in range(21, min(42, best_rank)):
        if time.time() > deadline - 10:
            break
        
        result = try_random_decomposition(target_rank, n, m, p, max_attempts=500)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found the lowest rank, we can stop
    
    if best_result is None:
        # Fall back to schoolbook
        best_result = schoolbook_decomposition(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
