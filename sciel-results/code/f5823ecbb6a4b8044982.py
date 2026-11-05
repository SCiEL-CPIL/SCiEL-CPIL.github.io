"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_eq = n * m * p  # 60 equations
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a 3-tensor decomposition. Let's try to find lower rank solutions.
    # Strategy: try ranks from lower bound up to schoolbook rank
    # Lower bound: max(n*m, m*p, n*p) = max(10, 30, 12) = 30
    
    lower_bound = max(n*m, m*p, n*p)
    
    # For each candidate rank, try to find a valid decomposition
    # Use a greedy approach with random restarts
    
    np.random.seed(42)
    
    def is_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is exact (within tolerance)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def check_half_integer(mat, tol=1e-9):
        """Check if all entries are half-integers (multiples of 0.5)."""
        for val in mat.flat:
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True
    
    # Try to find solutions with rank < schoolbook
    # Use a simple approach: for each rank, try random sparse patterns
    
    for candidate_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse half-integer matrices
            U = np.zeros((nm, candidate_rank))
            V = np.zeros((mp, candidate_rank))
            W = np.zeros((pn, candidate_rank))
            
            # Fill with random half-integers sparsely
            density = 0.3
            for r in range(candidate_rank):
                # Choose non-zero entries
                nz_U = np.random.choice(nm, size=max(1, int(nm*density)), replace=False)
                nz_V = np.random.choice(mp, size=max(1, int(mp*density)), replace=False)
                nz_W = np.random.choice(pn, size=max(1, int(pn*density)), replace=False)
                
                for idx in nz_U:
                    U[idx, r] = np.random.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
                for idx in nz_V:
                    V[idx, r] = np.random.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
                for idx in nz_W:
                    W[idx, r] = np.random.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
            
            if is_valid(U, V, W):
                if check_half_integer(U) and check_half_integer(V) and check_half_integer(W):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = candidate_rank
                    break
        if best_rank == candidate_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
