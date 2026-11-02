"""Find rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _is_half_integer(val):
    """Check if val is an integer or half-integer."""
    return abs(val * 2 - round(val * 2)) < 1e-9

def _all_half_integers(matrix):
    """Check if all entries in matrix are half-integers."""
    for row in matrix:
        for val in row:
            if not _is_half_integer(val):
                return False
    return True

def _to_half_int_list(matrix):
    """Convert matrix to list of lists with half-integer values."""
    return [[val for val in row] for row in matrix]

def construct():
    """Return {'U', 'V', 'W'} with minimal rank decomposition."""
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
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
    
    # For each rank r, try to find U, V, W such that:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # 
    # Strategy: fix U and V randomly, solve linear system for W
    # The system is linear in W: for each (i,j,k),
    # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    #
    # Unknowns: W[k*n+i,c] for k in 0..p-1, i in 0..n-1, c in 0..r-1
    # That's p*n*r unknowns
    # Equations: n*m*p = 24 equations
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from low to high
    for r in range(1, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Number of unknowns in W: p*n*r
        # Number of equations: n*m*p = 24
        # We need p*n*r >= 24 for a chance of solution
        # p*n = 8, so we need r >= 3
        
        if p * n * r < 24:
            continue
        
        # Try multiple random initializations for U and V
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random integer U and V
            # Use small integers to increase chance of half-integer W
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            
            # Build the linear system for W
            # Unknowns indexed by (k, i, c) -> flat index
            # Equations indexed by (i, j, k)
            
            num_unknowns = p * n * r
            A_matrix = np.zeros((24, num_unknowns))
            b_vector = np.ones(24)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            # W[k*n+i, c] has flat index: (k*n+i)*r + c
                            w_flat_idx = (k * n + i) * r + c
                            A_matrix[eq_idx, w_flat_idx] = U[i*m+j, c] * V[j*p+k, c]
                        eq_idx += 1
            
            # Solve the linear system
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution satisfies the equations
            if np.max(np.abs(A_matrix @ W_flat - b_vector)) > 1e-6:
                continue
            
            # Reshape W_flat to (p*n, r)
            W = W_flat.reshape(p * n, r)
            
            # Check if W entries are half-integers
            if not _all_half_integers(W.tolist()):
                continue
            
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Final validation
            if _validate(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                found = True
                break
        
        if found:
            # Found a decomposition of rank r, can stop
            break
    
    return best_result
