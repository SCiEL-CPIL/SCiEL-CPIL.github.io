"""Matrix multiplication 2x3 by 3x7: rank-minimal exact tensor decomposition search."""
import numpy as np
import itertools
import time

DIMENSIONS = [2, 3, 7]
N, M, P = 2, 3, 7

# EVOLVE-BLOCK-START
def _build_T():
    """Build the tensor T[a,b,c] with a in [0,6), b in [0,3), c in [0,14).
    T[a,b,c] = 1 if a == b*P + (c % M) and (c // M) == b.
    """
    T = np.zeros((P * N, M, N * M))
    for b in range(M):
        for i in range(N):
            for j in range(M):
                a = b * P + i
                c = j * N + b
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, tol=1e-8):
    """Check U V^T == T (flattened)."""
    T = _build_T()
    Tflat = T.reshape(T.shape[0], -1)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # U is (P*N, rank), V is (M, rank), W is (N*M, rank)
    # For each a: (U[:,r] * V[:,r]) @ W[:,r]^T should give T[a]
    # Sum_r U[a,r] * (V[:,r] outer W[:,r])
    # = sum_r U[a,r] * np.outer(V[:,r], W[:,r]).reshape(-1)
    prod = np.einsum('ar,br,cr->ac', U, V, W)
    return np.allclose(prod, Tflat, atol=tol)

def _try_rank(r, T, max_iter=200, seed=0, time_limit=10.0):
    """Try to find a decomposition of rank r using alternating minimization.
    Returns (U, V, W) or None.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    for _ in range(max_iter):
        if time.time() - start > time_limit:
            return None
        # Random init
        U = rng.randn(T.shape[0], r)
        V = rng.randn(T.shape[1], r)
        W = rng.randn(T.shape[2], r)
        prev_err = np.inf
        for it in range(500):
            if time.time() - start > time_limit:
                break
            # Fix V,W: solve for U
            # T[a] = sum_r U[a,r] * (V[:,r] outer W[:,r]).reshape(-1)
            # Let M_a = sum_r U[a,r] * np.outer(V[:,r], W[:,r]).reshape(-1)
            # Build linear system: for each a, T[a] = U[a,0]*m_0 + ... + U[a,r-1]*m_{r-1}
            # where m_r = (V[:,r] outer W[:,r]).reshape(-1)
            # This is: Tflat = U @ B where B[a',r] = <V[:,r], W[:,r]>... no
            # Actually: prod[a,c] = sum_r U[a,r] * V[:,r][b] * W[:,r][c]
            # For fixed a: prod[a,:] = sum_r U[a,r] * (V[:,r] outer W[:,r]).reshape(-1)
            # So Tflat[a,:] = sum_r U[a,r] * M_r where M_r = (V[:,r] outer W[:,r]).reshape(-1)
            # Tflat = U @ M^T where M is (r, Tflat.shape[1])
            M_mat = np.zeros((r, T.shape[1] * T.shape[2]))
            for r_idx in range(r):
                M_mat[r_idx] = np.outer(V[:, r_idx], W[:, r_idx]).reshape(-1)
            # Tflat = U @ M_mat.T  =>  U = Tflat @ (M_mat.T)^+ 
            # Solve least squares: U @ M_mat.T = Tflat
            try:
                U_new = np.linalg.lstsq(M_mat.T, Tflat, rcond=None)[0]
            except:
                break
            err_u = np.linalg.norm(U_new - U)
            U = U_new
            
            # Fix U,W: solve for V
            # prod[a,c] = sum_r U[a,r] * V[:,r][b] * W[:,r][c]
            # For fixed b: prod[:,b,:] = sum_r (U[:,r] * W[:,r][b]... no
            # prod[a,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b: prod[:,b,:] = sum_r V[b,r] * (U[:,r] outer W[:,r]).reshape(-1)
            # Tflat[:,b,:] = V[:,b] @ N_mat.T where N_mat[r,:] = (U[:,r] outer W[:,r]).reshape(-1)
            N_mat = np.zeros((r, T.shape[0] * T.shape[2]))
            for r_idx in range(r):
                N_mat[r_idx] = np.outer(U[:, r_idx], W[:, r_idx]).reshape(-1)
            try:
                V_new = np.linalg.lstsq(N_mat.T, Tflat.reshape(T.shape[0], T.shape[1], T.shape[2]).transpose(0, 2, 1).reshape(-1, T.shape[1]).T, rcond=None)[0]
            except:
                break
            V = V_new
            
            # Fix U,V: solve for W
            # prod[a,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed c: prod[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])... 
            # prod[a,c] = sum_r W[c,r] * (U[:,r] * V[:,r])... no
            # prod[a,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed c: prod[:,c] (over a) = sum_r W[c,r] * (U[:,r] * V[b,r])... 
            # Actually prod[a,c] for fixed c: = sum_r W[c,r] * (U[:,r] * V[:,r])... 
            # Wait: prod[a,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed c, as function of a: prod[a,c] = sum_r W[c,r] * (U[:,r] * V[:,r])... no, V[b,r] is scalar for fixed b
            # prod[a,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
            # For fixed c: the vector prod[:,c] over a = sum_r W[c,r] * (U[:,r] * V[:,r])... 
            # Hmm, let me think differently.
            # prod[a,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Fix a: prod[a,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
            # So for each a: Tflat[a,:] = sum_r W[:,r] * (U[a,r] * V[b,r])... 
            # Let P_a[r] = U[a,r] * V[b,r] (but b varies...)
            # Actually: Tflat[a, b*M+c] = prod[a,c] where the index maps (b,c)
            # Let me use: for each (a, b): T[a, b, :] = sum_r W[:, r] * (U[a,r] * V[b,r])
            # So T.reshape(T.shape[0], T.shape[1], T.shape[2]) = sum_r W[:,r] * (U[:,r] * V[:,r]).T... 
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
            # For fixed a,b: T[a,b,:] = sum_r W[:,r] * (U[a,r] * V[b,r])
            # So T.reshape(T.shape[0]*T.shape[1], T.shape[2]) = Q @ W^T where Q[a*b+b', r] = U[a,r]*V[b',r]
            Q = (U[:, :, None] * V[None, :, :]).reshape(T.shape[0] * T.shape[1], r)
            try:
                W_new = np.linalg.lstsq(Q, T.reshape(T.shape[0] * T.shape[1], T.shape[2]), rcond=None)[0]
            except:
                break
            W = W_new
            
            # Check convergence
            prod = np.einsum('ar,br,cr->ac', U, V, W)
            err = np.linalg.norm(prod - Tflat)
            if err < prev_err * 0.999:
                prev_err = err
            if err < 1e-6:
                break
        
        if time.time() - start > time_limit:
            break
        prod = np.einsum('ar,br,cr->ac', U, V, W)
        err = np.linalg.norm(prod - Tflat)
        if err < 1e-6:
            return U, V, W
    return None

def _polish(U, V, W, T, max_iter=100, tol=1e-10):
    """Polish an approximate solution to high precision."""
    Tflat = T.reshape(T.shape[0], -1)
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    for _ in range(max_iter):
        # Solve for U
        M_mat = np.zeros((r, T.shape[1] * T.shape[2]))
        for r_idx in range(r):
            M_mat[r_idx] = np.outer(V[:, r_idx], W[:, r_idx]).reshape(-1)
        U = np.linalg.lstsq(M_mat.T, Tflat, rcond=None)[0]
        
        # Solve for V
        N_mat = np.zeros((r, T.shape[0] * T.shape[2]))
        for r_idx in range(r):
            N_mat[r_idx] = np.outer(U[:, r_idx], W[:, r_idx]).reshape(-1)
        T_reshape = Tflat.reshape(T.shape[0], T.shape[1], T.shape[2])
        V = np.linalg.lstsq(N_mat.T, T_reshape.transpose(0, 2, 1).reshape(-1, T.shape[1]).T, rcond=None)[0]
        
        # Solve for W
        Q = (U[:, :, None] * V[None, :, :]).reshape(T.shape[0] * T.shape[1], r)
        W = np.linalg.lstsq(Q, T.reshape(T.shape[0] * T.shape[1], T.shape[2]), rcond=None)[0]
        
        prod = np.einsum('ar,br,cr->ac', U, V, W)
        err = np.linalg.norm(prod - Tflat)
        if err < tol:
            break
    return U, V, W

def _to_half_integer(x):
    """Round to nearest half-integer if close."""
    x2 = round(x * 2)
    if abs(x * 2 - x2) < 1e-6:
        return x2 / 2.0
    return x

def construct():
    T = _build_T()
    Tflat = T.reshape(T.shape[0], -1)
    deadline = time.time() + 130.0
    
    # Try ranks from low to high
    # Lower bound: max(P*N, M, N*M) = max(14, 3, 6) = 14
    # But tensor rank is likely higher. Schoolbook is 42.
    # Try from 14 upward, but start with reasonable guesses.
    
    best = None
    for r in range(14, 42):
        if time.time() > deadline - 15:
            break
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            result = _try_rank(r, T, max_iter=5, seed=seed, time_limit=10.0)
            if result is not None:
                U, V, W = result
                # Polish
                U, V, W = _polish(U, V, W, T, max_iter=50, tol=1e-10)
                prod = np.einsum('ar,br,cr->ac', U, V, W)
                if np.allclose(prod, Tflat, atol=1e-8):
                    best = (r, U, V, W)
                    break
        if best is not None and best[0] == r:
            break
    
    if best is None:
        # Fallback to schoolbook
        rank = N * M * P
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        column = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][column] = 1
                    V[j * P + k][column] = 1
                    W[k * N + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    r, U, V, W = best
    # Convert to lists, round to half-integers if possible
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Try to round to half-integers
    U_list = [[_to_half_integer(x) for x in row] for row in U_list]
    V_list = [[_to_half_integer(x) for x in row] for row in V_list]
    W_list = [[_to_half_integer(x) for x in row] for row in W_list]
    
    # Verify after rounding
    U_arr = np.array(U_list)
    V_arr = np.array(V_list)
    W_arr = np.array(W_list)
    prod = np.einsum('ar,br,cr->ac', U_arr, V_arr, W_arr)
    if not np.allclose(prod, Tflat, atol=1e-6):
        # Use unrounded values
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
