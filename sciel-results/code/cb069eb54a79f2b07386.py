import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 120  # seconds
SAFETY_MARGIN = 10  # seconds before limit to stop search

def _verify_decomposition(U, V, W, n, m, p, R, num_tests=5, seed=42):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randint(-3, 4, size=(n, m)).astype(float)
        B = rng.randint(-3, 4, size=(m, p)).astype(float)
        C_true = A @ B
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        C_approx = np.zeros((n, p))
        for r in range(R):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            w_vec = W[:, r]
            for i in range(n):
                for k in range(p):
                    C_approx[i, k] += w_vec[k * n + i] * u_val * v_val
                        
        if not np.allclose(C_approx, C_true, atol=1e-9):
            return False
    return True

def _polish_decomposition(U, V, W, n, m, p, R, max_iters=100, tol=1e-12):
    """Polish a near-solution to achieve higher precision."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    for iteration in range(max_iters):
        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A_sys = np.zeros((p, R))
                for k in range(p):
                    A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                try:
                    U_new[i*m+j, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                except:
                    pass
        U = U_new
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_sys = np.zeros((n, R))
                for i in range(n):
                    A_sys[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    V_new[j*p+k, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                except:
                    pass
        V = V_new
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A_sys = np.zeros((m, R))
                for j in range(m):
                    A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    W_new[k*n+i, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                except:
                    pass
        W = W_new
        
        # Check convergence
        if _verify_decomposition(U, V, W, n, m, p, R, num_tests=3, seed=42):
            # Do a final high-precision check
            if _verify_decomposition(U, V, W, n, m, p, R, num_tests=10, seed=123):
                return U, V, W
    
    return U, V, W

def _build_rank17(n, m, p):
    """Build the known rank-17 decomposition using Strassen on 2x2x2 block + schoolbook."""
    rank = 17
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = np.zeros((dim_U, rank), dtype=float)
    V = np.zeros((dim_V, rank), dtype=float)
    W = np.zeros((dim_W, rank), dtype=float)
    
    # --- Strassen's Algorithm for A[0:2, 0:2] * B[0:2, 0:2] ---
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    W[3, 1] = -1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1
    W[3, 2] = 1
    
    # M4 = A11 * (B10 - B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[1, 3] = 1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1
    W[0, 4] = -1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # --- Schoolbook for remaining terms ---
    # A[:, 2] * B[2, :]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # A[:, 0:2] * B[:, 2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    
    # A[:, 2] * B[2, 2]
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W, rank

def _try_find_rank16(n, m, p, deadline):
    """
    Attempt to find a rank-16 decomposition using structured search from rank-17.
    
    Strategy: Start from the known rank-17 solution and try to merge pairs of
    schoolbook terms using Strassen-style combinations. Also try alternating
    minimization with multiple random restarts.
    """
    R = 16
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6
    
    rng = np.random.RandomState(12345)
    
    # Approach 1: Try to merge two schoolbook terms from the rank-17 solution
    # The rank-17 solution has terms 7-16 as schoolbook terms.
    # Try to express pairs like (A_i2 * B_2k) + (A_i2 * B_2k') using fewer terms.
    
    # Actually, let's try a more systematic approach:
    # Use the rank-17 solution as a starting point and try to find a linear
    # combination of terms that can be replaced by a single term.
    
    U17, V17, W17, _ = _build_rank17(n, m, p)
    
    # Try alternating minimization starting from perturbed rank-17 (drop one term)
    for drop_idx in range(17):
        if time.time() > deadline:
            break
        
        # Create rank-16 initialization by dropping one term and redistributing
        U = np.zeros((dim_U, R), dtype=float)
        V = np.zeros((dim_V, R), dtype=float)
        W = np.zeros((dim_W, R), dtype=float)
        
        for r in range(17):
            if r < drop_idx:
                U[:, r] = U17[:, r]
                V[:, r] = V17[:, r]
                W[:, r] = W17[:, r]
            elif r > drop_idx:
                U[:, r-1] = U17[:, r]
                V[:, r-1] = V17[:, r]
                W[:, r-1] = W17[:, r]
        
        # Add small random perturbation
        U += rng.randn(dim_U, R) * 0.1
        V += rng.randn(dim_V, R) * 0.1
        W += rng.randn(dim_W, R) * 0.1
        
        # Alternating minimization
        for iteration in range(30):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_sys = np.zeros((p, R))
                    for k in range(p):
                        A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_sys = np.zeros((n, R))
                    for i in range(n):
                        A_sys[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_sys = np.zeros((m, R))
                    for j in range(m):
                        A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            W = W_new
        
        if _verify_decomposition(U, V, W, n, m, p, R, num_tests=5, seed=42+drop_idx):
            return U, V, W, R
    
    # Approach 2: Pure random restarts with alternating minimization
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        U = rng.randn(dim_U, R) * 0.5
        V = rng.randn(dim_V, R) * 0.5
        W = rng.randn(dim_W, R) * 0.5
        
        for iteration in range(20):
            if time.time() > deadline:
                break
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_sys = np.zeros((p, R))
                    for k in range(p):
                        A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            U = U_new
            
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_sys = np.zeros((n, R))
                    for i in range(n):
                        A_sys[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            V = V_new
            
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_sys = np.zeros((m, R))
                    for j in range(m):
                        A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            W = W_new
        
        if _verify_decomposition(U, V, W, n, m, p, R, num_tests=5, seed=42+100+attempt):
            return U, V, W, R
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # First, build the known rank-17 solution
    U17, V17, W17, rank17 = _build_rank17(n, m, p)
    
    # Verify the rank-17 solution
    if not _verify_decomposition(U17, V17, W17, n, m, p, rank17, num_tests=10, seed=42):
        # Should not happen, but fallback to schoolbook
        rank = 18
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        U = np.zeros((dim_U, rank), dtype=float)
        V = np.zeros((dim_V, rank), dtype=float)
        W = np.zeros((dim_W, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find rank 16 if we have time
    if time.time() < deadline:
        result = _try_find_rank16(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            # Polish the solution for higher precision
            if time.time() < deadline - 5:
                U, V, W = _polish_decomposition(U, V, W, n, m, p, rank, max_iters=50)
            # Final verification with more tests
            if _verify_decomposition(U, V, W, n, m, p, rank, num_tests=20, seed=999):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
