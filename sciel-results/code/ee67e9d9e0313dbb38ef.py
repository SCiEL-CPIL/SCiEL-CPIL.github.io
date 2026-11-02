"""Matrix multiplication 3,4,8 rank-minimal decomposition search."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
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
    return U, V, W

def _try_remove_fix_W(U, V, W, idx, n, m, p):
    """Remove column idx, fix W, solve for U and V via linear system."""
    r = len(U[0])
    if r <= 1:
        return None, None, None
    
    remaining = [c for c in range(r) if c != idx]
    r_rem = len(remaining)
    
    Ui = np.array([U[i][idx] for i in range(n*m)])
    Vi = np.array([V[j][idx] for j in range(m*p)])
    Wi = np.array([W[k][idx] for k in range(p*n)])
    
    # Residual: R[i,j,k] = -Ui[i*m+j] * Vi[j*p+k] * Wi[k*n+i]
    R = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[i,j,k] = -Ui[i*m+j] * Vi[j*p+k] * Wi[k*n+i]
    
    # Fix W. Solve: sum_c dU[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] + 
    #                sum_c U[i*m+j,c]*dV[j*p+k,c]*W[k*n+i,c] = R[i,j,k]
    # This is bilinear. Linearize by fixing one set.
    # Strategy 1: set dV=0, solve for dU.
    # sum_c dU[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = R[i,j,k]
    # For fixed (j,k): let A[i,c] = V[j*p+k,c]*W[k*n+i,c], b[i] = R[i,j,k]
    # dU[i*m+j, :] must satisfy: for all i, dU[i*m+j,:] dot A[i,:] = b[i]
    # => A^T dU[i*m+j,:]^T = b => dU[i*m+j,:] = (A^T)^+ b
    
    Va_rem = np.array([[V[j][c] for c in remaining] for j in range(m*p)])
    Wa_rem = np.array([[W[k][c] for c in remaining] for k in range(p*n)])
    
    U_new = [[0]*r_rem for _ in range(n*m)]
    ok = True
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r_rem))
            for i in range(n):
                for ci in range(r_rem):
                    A[i, ci] = Va_rem[j*p+k, ci] * Wa_rem[k*n+i, ci]
            b = np.array([R[i,j,k] for i in range(n)])
            try:
                x, res, rk, sv = np.linalg.lstsq(A.T, b, rcond=1e-12)
            except:
                ok = False
                break
            if np.linalg.norm(A.T @ x - b) > 1e-8:
                ok = False
                break
            for i in range(n):
                U_new[i*m+j] = list(x)
        if not ok:
            break
    if not ok:
        return None, None, None
    
    V_new = [[V[j][c] for c in remaining] for j in range(m*p)]
    W_new = [[W[k][c] for c in remaining] for k in range(p*n)]
    
    if _verify(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None, None, None

def _try_remove_fix_V(U, V, W, idx, n, m, p):
    """Remove column idx, fix V, solve for U and W."""
    r = len(U[0])
    if r <= 1:
        return None, None, None
    remaining = [c for c in range(r) if c != idx]
    r_rem = len(remaining)
    
    Ui = np.array([U[i][idx] for i in range(n*m)])
    Vi = np.array([V[j][idx] for j in range(m*p)])
    Wi = np.array([W[k][idx] for k in range(p*n)])
    
    R = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[i,j,k] = -Ui[i*m+j] * Vi[j*p+k] * Wi[k*n+i]
    
    # Fix V. sum_c dU[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] + sum_c U[i*m+j,c]*V[j*p+k,c]*dW[k*n+i,c] = R
    # Set dU=0: sum_c U[i*m+j,c]*V[j*p+k,c]*dW[k*n+i,c] = R[i,j,k]
    # For fixed (i,j): let B[k,c] = U[i*m+j,c]*V[j*p+k,c], b[k] = R[i,j,k]
    # dW[k*n+i,:] must satisfy: for all k, dW[k*n+i,:] dot B[k,:] = b[k]
    # => B^T dW[k*n+i,:]^T = b
    
    Ua_rem = np.array([[U[i][c] for c in remaining] for i in range(n*m)])
    Va_rem = np.array([[V[j][c] for c in remaining] for j in range(m*p)])
    
    W_new = [[0]*r_rem for _ in range(p*n)]
    ok = True
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, r_rem))
            for k in range(p):
                for ci in range(r_rem):
                    B[k, ci] = Ua_rem[i*m+j, ci] * Va_rem[j*p+k, ci]
            b = np.array([R[i,j,k] for k in range(p)])
            try:
                x, res, rk, sv = np.linalg.lstsq(B.T, b, rcond=1e-12)
            except:
                ok = False
                break
            if np.linalg.norm(B.T @ x - b) > 1e-8:
                ok = False
                break
            for k in range(p):
                W_new[k*n+i] = list(x)
        if not ok:
            break
    if not ok:
        return None, None, None
    
    U_new = [[U[i][c] for c in remaining] for i in range(n*m)]
    V_new = [[V[j][c] for c in remaining] for j in range(m*p)]
    
    if _verify(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None, None, None

def _try_remove_fix_U(U, V, W, idx, n, m, p):
    """Remove column idx, fix U, solve for V and W."""
    r = len(U[0])
    if r <= 1:
        return None, None, None
    remaining = [c for c in range(r) if c != idx]
    r_rem = len(remaining)
    
    Ui = np.array([U[i][idx] for i in range(n*m)])
    Vi = np.array([V[j][idx] for j in range(m*p)])
    Wi = np.array([W[k][idx] for k in range(p*n)])
    
    R = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[i,j,k] = -Ui[i*m+j] * Vi[j*p+k] * Wi[k*n+i]
    
    # Fix U. sum_c U[i*m+j,c]*dV[j*p+k,c]*W[k*n+i,c] + sum_c U[i*m+j,c]*V[j*p+k,c]*dW[k*n+i,c] = R
    # Set dV=0: sum_c U[i*m+j,c]*V[j*p+k,c]*dW[k*n+i,c] = R[i,j,k]
    # For fixed (i,j): B[k,c] = U[i*m+j,c]*V[j*p+k,c], b[k]=R[i,j,k]
    # dW[k*n+i,:] satisfies B^T x = b
    
    Ua_rem = np.array([[U[i][c] for c in remaining] for i in range(n*m)])
    Va_rem = np.array([[V[j][c] for c in remaining] for j in range(m*p)])
    
    W_new = [[0]*r_rem for _ in range(p*n)]
    ok = True
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, r_rem))
            for k in range(p):
                for ci in range(r_rem):
                    B[k, ci] = Ua_rem[i*m+j, ci] * Va_rem[j*p+k, ci]
            b = np.array([R[i,j,k] for k in range(p)])
            try:
                x, res, rk, sv = np.linalg.lstsq(B.T, b, rcond=1e-12)
            except:
                ok = False
                break
            if np.linalg.norm(B.T @ x - b) > 1e-8:
                ok = False
                break
            for k in range(p):
                W_new[k*n+i] = list(x)
        if not ok:
            break
    if not ok:
        return None, None, None
    
    V_new = [[V[j][c] for c in remaining] for j in range(m*p)]
    U_new = [[U[i][c] for c in remaining] for i in range(n*m)]
    
    if _verify(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    random.seed(123)
    rank = len(U[0])
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline - 15:
            break
        
        # Try removing a random column using different strategies
        order = list(range(rank))
        random.shuffle(order)
        
        progress = False
        for idx in order:
            if time.time() > deadline - 15:
                break
            if idx >= rank:
                continue
            
            strategies = [
                _try_remove_fix_W,
                _try_remove_fix_V,
                _try_remove_fix_U,
            ]
            random.shuffle(strategies)
            
            for strat in strategies:
                U_new, V_new, W_new = strat(U, V, W, idx, n, m, p)
                if U_new is not None:
                    U, V, W = U_new, V_new, W_new
                    rank = len(U[0])
                    if rank < best_rank:
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                    progress = True
                    break
            if progress:
                break
        
        if not progress:
            # Try random restart: add a random column, then remove
            if rank > 2:
                # Add a random column
                U.append([random.gauss(0,1) for _ in range(rank)])
                V.append([random.gauss(0,1) for _ in range(rank)])
                W.append([random.gauss(0,1) for _ in range(rank)])
                rank += 1
                # Now try to remove columns to get back
                for _ in range(5):
                    if time.time() > deadline - 15:
                        break
                    order2 = list(range(rank))
                    random.shuffle(order2)
                    for idx2 in order2:
                        if idx2 >= rank:
                            continue
                        for strat in [_try_remove_fix_W, _try_remove_fix_V, _try_remove_fix_U]:
                            U_new, V_new, W_new = strat(U, V, W, idx2, n, m, p)
                            if U_new is not None:
                                U, V, W = U_new, V_new, W_new
                                rank = len(U[0])
                                if rank < best_rank:
                                    best_rank = rank
                                    best_U, best_V, best_W = U, V, W
                                break
                        if U is not U_new:
                            break
                    if rank <= best_rank:
                        break
    
    return {"U": best_U, "V": best_V, "W": best_W}
