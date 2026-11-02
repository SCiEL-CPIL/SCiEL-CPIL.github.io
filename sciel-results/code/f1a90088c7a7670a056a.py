"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s

def _schoolbook():
    """Returns the trivial schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    return U, V, W

def _construct_tensor():
    """Constructs the tensor T[u, v, w] for the multiplication C = A * B."""
    n, m, p = DIMENSIONS
    # A is (n, m) -> flattened to size n*m
    # B is (m, p) -> flattened to size m*p
    # C is (n, p) -> flattened to size n*p
    # T[u, v, w] contributes to C_w if A_u and B_v are multiplied.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # u = i * m + j
    # v = j * p + k
    # w = k * n + i (following W[k*n+i, r] convention in prompt)
    
    T = np.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            u = i * m + j
            for k in range(p):
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _rank_decomposition_search():
    """
    Attempts to find a lower rank decomposition using numerical methods.
    Returns (U, V, W) if successful, else None.
    """
    n, m, p = DIMENSIONS
    T = _construct_tensor()
    T_flat = T.reshape(n * m * m * p, n * p)  # This reshaping is tricky for direct LSQ, let's use modes.
    
    # We look for T = sum_r u_r \otimes v_r \otimes w_r
    # Flatten T to matrix: Mode-3 unfolding T_(3) where rows are w indices, cols are (u, v) indices.
    # T_(3)[w, u*p+m*v] = T[u, v, w]
    
    best_result = None
    best_rank = (n * m * p) + 1 # Start higher than schoolbook
    
    # Try ranks slightly below schoolbook down to a reasonable bound
    # 3x3x8 -> 72. Try 71, 70, ... maybe 60.
    # Given the strict 120s limit, we can't search deeply. 
    # We will try a few ranks with limited iterations.
    
    target_ranks = [72, 71, 70, 68, 65] # Check a few
    
    for r in target_ranks:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Skip if we don't expect improvement (72 is schoolbook, we want < 72)
        if r >= best_rank:
            continue
            
        # Try 3 random initializations for this rank
        success = False
        for trial in range(3):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
                
            # Random init
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(n * p, r) * 0.5
            
            # Optimize
            for step in range(20): # Max 20 iterations
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
                    
                # Fix V, W -> Solve for U
                # T(u, v, w) = sum_r U[u, r] V[v, r] W[w, r]
                # For each u: T_u (vector of size m*p*n*p) approx U_u (vector r) dot (V kron W)^T?
                # Let M_vw be a matrix of size (m*p, n*p) where M_vw[v, w] = sum_r V[v, r] W[w, r]
                # No, it's a rank-r update.
                # We can use a simple gradient descent or alternating least squares.
                # ALS:
                # Flatten T to 2D matrix for mode-1 unfolding: T_(1) size (n*m, m*p*n*p)
                # T_(1) = U * (W \otimes V)^T ? No.
                # T[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
                # Let A1 = V \otimes W (Kronecker product)?
                # (V \otimes W)[v * n*p + w, r] = V[v, r] W[w, r]
                # Then T_(1)[u, v * n*p + w] = sum_r U[u, r] (V \otimes W)[v * n*p + w, r]
                # So T_(1) \approx U * (V \otimes W)^T
                # This matrix is (n*m) x (m*p*n*p). For 3,3,8:
                # U: 9 x r
                # V: 24 x r
                # W: 24 x r
                # V \otimes W: (24*24) x r = 576 x r
                # T_(1) is 9 x 576
                
                VW = np.kron(V, W) # Size (m*p*n*p) x r
                # T_(1) is T.reshape(n*m, -1)
                T1 = T.reshape(n * m, m * p * n * p)
                # Solve min || T1 - U * VW.T ||_F
                # U = T1 @ VW @ pinv(VW.T @ VW) ?
                # U = T1 @ VW @ (VW.T @ VW)^-1
                # Since r is small (approx 60-72), VW.T @ VW is r x r.
                try:
                    U = np.linalg.lstsq(VW, T1, rcond=None)[0].T # Solving (U * VW.T) = T1 -> U @ (VW.T) = T1
                    # Actually lstsq(A, B) solves min ||A X - B||. 
                    # We want min || T1 - U VW.T ||.
                    # T1 is 9x576. VW.T is r x 576.
                    # Let U_new = X. X (r x 9)? No U is 9 x r.
                    # T1 = U @ VW.T
                    # Transpose: T1.T = VW @ U.T
                    # Solve min || T1.T - VW @ U.T ||
                    U_new = np.linalg.lstsq(VW, T1.T, rcond=None)[0]
                    U = U_new.T
                except:
                    break
                
                # Similarly for V:
                # T_(2) is (m*p) x (n*m*n*p)
                # T1 = U \otimes W
                UW = np.kron(U, W)
                T2 = T.transpose(1, 0, 2).reshape(m * p, n * m * n * p)
                try:
                    V_new = np.linalg.lstsq(UW, T2.T, rcond=None)[0]
                    V = V_new.T
                except:
                    break
                    
                # Similarly for W:
                # T_(3) is (n*p) x (n*m*m*p)
                # T1 = U \otimes V
                UV = np.kron(U, V)
                T3 = T.transpose(2, 0, 1).reshape(n * p, n * m * m * p)
                try:
                    W_new = np.linalg.lstsq(UV, T3.T, rcond=None)[0]
                    W = W_new.T
                except:
                    break
                    
            # End of iterations
            
            # Check error
            # Recalculate T_est
            T_est = np.einsum('ur, vr, wr -> uvw', U, V, W)
            err = np.linalg.norm(T_est - T)
            
            if err < 1e-8:
                # Found exact numerical solution.
                # Now check if entries are close to integers or half-integers.
                # The prompt asks for integer/half-integer entries.
                # Scale if necessary? "integer/half-integer entries" usually implies 0.5, 1.0, etc.
                all_close = True
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                T_est_int = np.einsum('ur, vr, wr -> uvw', U_int, V_int, W_int)
                err_int = np.linalg.norm(T_est_int - T)
                
                if err_int < 1e-9:
                    return U_int.tolist(), V_int.tolist(), W_int.tolist(), r
                else:
                    # If not half-integer, maybe integer?
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    T_est_int = np.einsum('ur, vr, wr -> uvw', U_int, V_int, W_int)
                    err_int = np.linalg.norm(T_est_int - T)
                    if err_int < 1e-9:
                        return U_int.tolist(), V_int.tolist(), W_int.tolist(), r
                    
                    # If numerical solution exists but not integer/half-integer,
                    # and the task strictly requires integer/half-integer, we can't use it.
                    # But wait, "integer/half-integer entries that exactly decompose".
                    # If the numerical solution is irrational, it's invalid.
                    # We will not use it.
                    pass
            # End if err
        # End trial loop
        
    return None

def construct():
    # Try to find a better decomposition
    # Note: For 3x3x8, the rank is likely 72.
    # But we attempt the search as per the strategy.
    
    result = _rank_decomposition_search()
    if result is not None:
        U, V, W, r = result
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
# The main logic is wrapped in construct as required.
# The helpers are outside.
# EVOLVE-BLOCK-END
