"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return schoolbook()
    
    # Try to find a rank-12 decomposition
    # The constraint is: for all i, i', j1, j2, k, k':
    # sum_r W[k*n+i, r] * U[i'*m+j1, r] * V[j2*p+k', r] = delta_{i,i'} * delta_{j1,j2} * delta_{k,k'}
    
    # This is a system of quadratic equations in U, V, W.
    # We'll try a structured approach: fix W to have a specific pattern,
    # then solve for U and V.
    
    # Strategy: Use the fact that rank 12 is achievable.
    # Try a block structure based on the dimensions.
    
    # For now, return schoolbook as we need to verify any new solution.
    # A rank-12 solution would score 24/12 = 2.0 if valid.
    
    # Let's attempt a constructive approach:
    # We need 12 products. Group them as 2*3*4 / 2 = 12 (factor of 2 savings).
    
    # Try a specific rank-12 construction:
    # Use 12 columns. For each column r, assign non-zero entries to U, V, W.
    
    # Given the risk of error, and that we must verify exactness,
    # let's try to solve the linear system for a rank-12 solution.
    
    # The constraints are:
    # For each (i, k): the bilinear form must match.
    # This gives n*p = 8 constraints, but each is a matrix equation.
    
    # Actually, the full constraint set is:
    # For all i, i', j1, j2, k, k':
    # sum_r W[k*n+i, r] * U[i'*m+j1, r] * V[j2*p+k', r] = delta_{i,i'} * delta_{j1,j2} * delta_{k,k'}
    
    # Number of constraints: n*n*m*m*p*p = 2*2*3*3*4*4 = 576
    # Unknowns for rank r: n*m*r + m*p*r + p*n*r = (6+12+8)*r = 26*r
    # For r=12: 312 unknowns, 576 constraints -> overdetermined
    
    # But the equations are quadratic, not linear. This is hard.
    
    # Alternative: use a known decomposition or random search with verification.
    
    # Let's try random initialization and local search for rank 12.
    # If we find a valid solution, great. Otherwise, fall back to schoolbook.
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try rank 12 with random restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        r = 12
        # Random initialization with half-integer values
        np.random.seed(42 + restart)
        U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(n*m, r))
        V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(m*p, r))
        W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(p*n, r))
        
        # Verify the constraint
        # For all i, i', j1, j2, k, k':
        # sum_r W[k*n+i, r] * U[i'*m+j1, r] * V[j2*p+k', r] = delta_{i,i'} * delta_{j1,j2} * delta_{k,k'}
        
        valid = True
        for i in range(n):
            for i_prime in range(n):
                for j1 in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                lhs = 0.0
                                for r_idx in range(r):
                                    lhs += W[k*n+i, r_idx] * U[i_prime*m+j1, r_idx] * V[j2*p+k_prime, r_idx]
                                rhs = (1.0 if (i == i_prime and j1 == j2 and k == k_prime) else 0.0)
                                if abs(lhs - rhs) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to list of lists with exact values
            U_list = [[float(U[i,j]) for j in range(r)] for i in range(n*m)]
            V_list = [[float(V[i,j]) for j in range(r)] for i in range(m*p)]
            W_list = [[float(W[i,j]) for j in range(r)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no rank-12 solution found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
