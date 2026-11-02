"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Build the matrix multiplication tensor
    2. Use SVD to find the numerical rank (lower bound)
    3. Try to construct a valid decomposition at that rank with integer/half-integer entries
    4. If needed, incrementally increase rank and use local search
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the tensor as a 3D array: T[a, b, c]
    # a = i*m + j for i in [0,n), j in [0,m)
    # b = j*p + k for j in [0,m), k in [0,p)
    # c = k*n + i for k in [0,p), i in [0,n)
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # The schoolbook rank is n*m*p = 84
    schoolbook_rank = n * m * p
    
    # Helper to verify a decomposition
    def verify_decomposition(U, V, W, tol=1e-8):
        """Check if T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r] for all a,b,c"""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        max_error = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err = abs(val - T[a, b, c])
                    if err > max_error:
                        max_error = err
        return max_error < tol
    
    # Helper to create schoolbook decomposition
    def make_schoolbook(rank):
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try to find a decomposition with rank less than schoolbook
    # Use SVD-based approach with iterative refinement
    
    # Get SVD of the matricized tensor
    T_mat1 = T.reshape(nm * mp, pn)
    U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat1, full_matrices=False)
    
    # Try ranks from lower bound up to schoolbook_rank - 1
    # The tensor rank is at least max of numerical ranks of unfoldings
    tol_rank = 1e-10
    rank1 = np.sum(s_vals > tol_rank)
    
    # For matrix multiplication, the border rank is at most n*m*p
    # but the actual rank may be lower. Known bounds for 3x4x7:
    # Strassen showed that 2x2x2 can be done in 7 (vs 8 schoolbook)
    # For 3x4x7, we don't know the exact minimal rank, but it's <= 84
    
    # Try a range of ranks
    min_rank_to_try = max(1, int(rank1))
    max_rank_to_try = min(schoolbook_rank - 1, min_rank_to_try + 15)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, try SVD-based approach for each rank
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 30:
            break
        
        try:
            # Truncate SVD to this rank
            U_trunc = U_svd[:, :rank] * s_vals[:rank]
            Vt_trunc = Vt_svd[:rank, :]
            
            U_3d = U_trunc.reshape(nm, rank)
            V_3d = Vt_trunc.reshape(mp, rank)
            
            # Solve for W
            M = np.zeros((nm * mp, rank))
            for r in range(rank):
                M[:, r] = (U_3d[:, r] * V_3d[:, r]).flatten()
            
            W_3d = np.zeros((pn, rank))
            for c in range(pn):
                target = T[:, :, c].flatten()
                w, residuals, rank_ls, sv = np.linalg.lstsq(M, target, rcond=None)
                W_3d[c, :] = w
            
            # Verify
            if verify_decomposition(U_3d, V_3d, W_3d, tol=1e-8):
                # Try to round to half-integers
                U_rounded = np.round(U_3d * 2) / 2
                V_rounded = np.round(V_3d * 2) / 2
                W_rounded = np.round(W_3d * 2) / 2
                
                if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-8):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                else:
                    # Try iterative refinement to get closer to half-integers
                    for it in range(50):
                        if time.time() > deadline - 30:
                            break
                        # Gradient-based refinement
                        U_grad = np.zeros_like(U_3d)
                        V_grad = np.zeros_like(V_3d)
                        W_grad = np.zeros_like(W_3d)
                        
                        for a in range(nm):
                            for b in range(mp):
                                for c in range(pn):
                                    diff = T[a, b, c] - np.dot(U_3d[a, :], V_3d[b, :] * W_3d[c, :])
                                    U_grad[a, :] += diff * V_3d[b, :] * W_3d[c, :]
                                    V_grad[b, :] += diff * U_3d[a, :] * W_3d[c, :]
                                    W_grad[c, :] += diff * U_3d[a, :] * V_3d[b, :]
                        
                        lr = 0.01
                        U_3d -= lr * U_grad
                        V_3d -= lr * V_grad
                        W_3d -= lr * W_grad
                        
                        if verify_decomposition(U_3d, V_3d, W_3d, tol=1e-8):
                            U_rounded = np.round(U_3d * 2) / 2
                            V_rounded = np.round(V_3d * 2) / 2
                            W_rounded = np.round(W_3d * 2) / 2
                            if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-8):
                                return {
                                    "U": U_rounded.tolist(),
                                    "V": V_rounded.tolist(),
                                    "W": W_rounded.tolist()
                                }
                            break
                        
        except Exception:
            continue
    
    # If we couldn't find a better decomposition, return schoolbook
    U_sb, V_sb, W_sb = make_schoolbook(schoolbook_rank)
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
