"""Optimized program for 2x3 * 3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook rank is n*m*p = 42
    best_rank = n * m * p
    best_result = None
    
    # Initialize with schoolbook as fallback
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # We'll try ranks from 21 up to 41
    
    np.random.seed(42)
    
    candidate_ranks = list(range(21, best_rank))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations with coordinate descent
        max_attempts = 30 if rank <= 25 else 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize U, V, W with random small values
            # Use values from a small set to help with half-integer projection
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Random initialization with small values
            for r in range(rank):
                # Each rank-1 component should contribute to the product
                # Initialize with random values that might lead to a solution
                U[:, r] = np.random.uniform(-1, 1, n * m)
                V[:, r] = np.random.uniform(-1, 1, m * p)
                W[:, r] = np.random.uniform(-1, 1, p * n)
            
            # Coordinate descent optimization
            U, V, W = _coordinate_descent(U, V, W, n, m, p, deadline)
            
            # Project to nearest half-integers
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            # Check if the half-integer solution is exact
            if _check_solution(U_half, V_half, W_half, n, m, p):
                best_result = _to_exact(U_half, V_half, W_half)
                best_rank = rank
                found = True
                break
            
            # If not exact, try a few more coordinate descent iterations
            # from the projected values
            U2, V2, W2 = U_half.copy(), V_half.copy(), W_half.copy()
            U2, V2, W2 = _coordinate_descent(U2, V2, W2, n, m, p, deadline, max_iters=50)
            
            U2_half = np.round(U2 * 2) / 2
            V2_half = np.round(V2 * 2) / 2
            W2_half = np.round(W2 * 2) / 2
            
            if _check_solution(U2_half, V2_half, W2_half, n, m, p):
                best_result = _to_exact(U2_half, V2_half, W2_half)
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    return best_result


def _coordinate_descent(U, V, W, n, m, p, deadline, max_iters=200):
    """Perform coordinate descent to minimize the error in the tensor decomposition.
    
    Error = sum_{i,j,k} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
    """
    rank = U.shape[1]
    
    for iteration in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Update U entries
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for r in range(rank):
                    # Compute optimal value for U[u_idx, r]
                    # The error terms involving U[u_idx, r] are for all k
                    # E_k = (sum_{r' != r} U[u_idx, r'] * V[j*p+k, r'] * W[k*n+i, r'] 
                    #         + U[u_idx, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
                    
                    # Let C_k = sum_{r' != r} U[u_idx, r'] * V[j*p+k, r'] * W[k*n+i, r']
                    # Let y_k = V[j*p+k, r] * W[k*n+i, r]
                    # Minimize sum_k (C_k + x * y_k - 1)^2
                    # Optimal x = -sum_k y_k * (C_k - 1) / sum_k y_k^2
                    
                    C = np.zeros(p)
                    y = np.zeros(p)
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        C[k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :]) - U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        y[k] = V[v_idx, r] * W[w_idx, r]
                    
                    denom = np.sum(y * y)
                    if denom > 1e-12:
                        U[u_idx, r] = -np.sum(y * (C - 1)) / denom
        
        # Update V entries
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                for r in range(rank):
                    # The error terms involving V[v_idx, r] are for all i
                    # Let C_i = sum_{r' != r} U[i*m+j, r'] * V[v_idx, r'] * W[k*n+i, r']
                    # Let y_i = U[i*m+j, r] * W[k*n+i, r]
                    
                    C = np.zeros(n)
                    y = np.zeros(n)
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        C[i] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :]) - U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        y[i] = U[u_idx, r] * W[w_idx, r]
                    
                    denom = np.sum(y * y)
                    if denom > 1e-12:
                        V[v_idx, r] = -np.sum(y * (C - 1)) / denom
        
        # Update W entries
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                for r in range(rank):
                    # The error terms involving W[w_idx, r] are for all j
                    # Let C_j = sum_{r' != r} U[i*m+j, r'] * V[j*p+k, r'] * W[w_idx, r']
                    # Let y_j = U[i*m+j, r] * V[j*p+k, r]
                    
                    C = np.zeros(m)
                    y = np.zeros(m)
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        C[j] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :]) - U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        y[j] = U[u_idx, r] * V[v_idx, r]
                    
                    denom = np.sum(y * y)
                    if denom > 1e-12:
                        W[w_idx, r] = -np.sum(y * (C - 1)) / denom
    
    return U, V, W


def _check_solution(U, V, W, n, m, p):
    """Check if U, V, W form an exact decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _to_exact(U, V, W):
    """Convert numpy arrays to list of lists with exact values."""
    U_exact = [[float(x) for x in row] for row in U]
    V_exact = [[float(x) for x in row] for row in V]
    W_exact = [[float(x) for x in row] for row in W]
    return {"U": U_exact, "V": V_exact, "W": W_exact}
# EVOLVE-BLOCK-END
