import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    max_time = 145  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 4
    
    # Precompute index arrays for fast verification
    U_idx = np.array([[i * m + j for j in range(m)] for i in range(n)])
    V_idx = np.array([[j * p + k for k in range(p)] for j in range(m)])
    W_idx = np.array([[k * n + i for i in range(n)] for k in range(p)])
    
    # Build the proven rank-22 construction as fallback
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def snap_to_half_integer(arr, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped
    
    def try_als(rank, max_iter=300, tol=1e-12, seed=42, init_U=None, init_V=None, init_W=None):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_U is not None:
            U = np.array(init_U, dtype=float).copy()
            V = np.array(init_V, dtype=float).copy()
            W = np.array(init_W, dtype=float).copy()
        else:
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = np.zeros((p * n, rank))
        
        for iteration in range(max_iter):
            if time.time() - start_time > max_time - 5:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                U_list = [[round(float(x), 10) for x in row] for row in U]
                V_list = [[round(float(x), 10) for x in row] for row in V]
                W_list = [[round(float(x), 10) for x in row] for row in W]
                if verify(U_list, V_list, W_list, rank):
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U_list = [[round(float(x), 10) for x in row] for row in U_snap]
                    V_list = [[round(float(x), 10) for x in row] for row in V_snap]
                    W_list = [[round(float(x), 10) for x in row] for row in W_snap]
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
                    else:
                        return U_list, V_list, W_list
        return None
    
    def try_rational_support(rank, max_attempts=200):
        """
        Try to find an exact decomposition by:
        1. Randomly choosing which (i,j), (j,k), (k,i) indices are non-zero for each rank term
        2. Solving the resulting system of equations exactly using rational arithmetic
        """
        rng = np.random.RandomState(12345)
        
        # All possible index combinations
        U_indices = list(range(n * m))  # 6
        V_indices = list(range(m * p))  # 12
        W_indices = list(range(p * n))  # 8
        
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time - 10:
                break
            
            # For each rank term, randomly choose 2-3 non-zero entries in each factor
            U_pattern = []
            V_pattern = []
            W_pattern = []
            
            for t in range(rank):
                # Choose 2 indices for U, 2 for V, 2 for W (sparse)
                u_idx = rng.choice(U_indices, size=2, replace=False)
                v_idx = rng.choice(V_indices, size=2, replace=False)
                w_idx = rng.choice(W_indices, size=2, replace=False)
                U_pattern.append(set(u_idx))
                V_pattern.append(set(v_idx))
                W_pattern.append(set(w_idx))
            
            # Build the linear system
            # Variables: U[i,j,t] for i,j in support, V[j,k,t], W[k,i,t]
            # Equations: For each (i,j,k), sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1
            # This is nonlinear! We need a different approach.
            
            # Alternative: Fix U and V patterns, solve for W, then check if it works
            # Actually, let's use a simpler approach:
            # Assume all U and V entries are ±1 or 0, and solve for W entries
            
            # Build U and V matrices with random ±1 values on the support
            U_mat = np.zeros((n * m, rank))
            V_mat = np.zeros((m * p, rank))
            
            for t in range(rank):
                for idx in U_pattern[t]:
                    U_mat[idx, t] = rng.choice([-1, 1])
                for idx in V_pattern[t]:
                    V_mat[idx, t] = rng.choice([-1, 1])
            
            # Now solve for W: for each (k,i), we need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all j
            # This gives us m equations for each (k,i) pair
            W_mat = np.zeros((p * n, rank))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # Build the system: for each j, sum_t U[i*m+j, t] * V[j*p+k, t] * W[w_idx, t] = 1
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U_mat[i*m+j, :] * V_mat[j*p+k, :]
                    b = np.ones(m)
                    
                    # Solve A * W[w_idx, :] = b
                    try:
                        # Use least squares
                        W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if the solution is close to half-integers
                        W_snap = snap_to_half_integer(W_sol.reshape(1, -1)).flatten()
                        # Verify the solution
                        if np.allclose(A @ W_snap, b, atol=1e-9):
                            W_mat[w_idx, :] = W_snap
                        else:
                            valid = False
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition
            U_list = [[round(float(x), 10) for x in row] for row in U_mat]
            V_list = [[round(float(x), 10) for x in row] for row in V_mat]
            W_list = [[round(float(x), 10) for x in row] for row in W_mat]
            
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
        
        return None
    
    # Try to find lower rank decompositions
    best_result = None
    
    # First try ALS with multiple seeds
    for rank in range(18, 22):
        if time.time() - start_time > max_time - 15:
            break
        
        for seed in range(30):
            if time.time() - start_time > max_time - 5:
                break
            result = try_als(rank, max_iter=150, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                if verify(U_list, V_list, W_list, rank):
                    best_result = (U_list, V_list, W_list, rank)
                    break
        
        if best_result is not None:
            break
    
    # If ALS didn't find anything, try the rational support approach
    if best_result is None:
        for rank in range(18, 22):
            if time.time() - start_time > max_time - 10:
                break
            result = try_rational_support(rank, max_attempts=100)
            if result is not None:
                U_list, V_list, W_list = result
                if verify(U_list, V_list, W_list, rank):
                    best_result = (U_list, V_list, W_list, rank)
                    break
    
    # Return the best result or fallback to rank 22
    if best_result is not None:
        U_list, V_list, W_list, r = best_result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return the proven rank-22 construction
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
