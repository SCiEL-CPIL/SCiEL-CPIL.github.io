"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor for n x m times m x p."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _schoolbook(n, m, p):
    """Schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, rank

def _build_full_3x3x7(U33, V33, W33, rank33):
    """Build the full 3x3 x 3x7 decomposition from 3x3 x 3x3 sub-decompositions."""
    # The 3x7 matrix is split into three parts: columns 0-2, 3-5, 6
    U31 = np.zeros((9, 9))
    V31 = np.zeros((3, 9))
    W31 = np.zeros((3, 9))
    col = 0
    for i in range(3):
        for j in range(3):
            U31[i*3+j, col] = 1
            V31[j, col] = 1
            W31[i, col] = 1
            col += 1
    
    total_rank = rank33 + rank33 + 9
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((21, total_rank))
    W_full = np.zeros((21, total_rank))
    
    # Group 1: k=0,1,2
    U_full[:, 0:rank33] = U33
    for j in range(3):
        for k in range(3):
            V_full[j*7+k, 0:rank33] = V33[j*3+k, :]
    for i in range(3):
        for k in range(3):
            W_full[k*3+i, 0:rank33] = W33[k*3+i, :]
    
    # Group 2: k=3,4,5
    U_full[:, rank33:2*rank33] = U33
    for j in range(3):
        for k in range(3):
            V_full[j*7+(k+3), rank33:2*rank33] = V33[j*3+k, :]
    for i in range(3):
        for k in range(3):
            W_full[(k+3)*3+i, rank33:2*rank33] = W33[k*3+i, :]
    
    # Group 3: k=6
    U_full[:, 2*rank33:2*rank33+9] = U31
    for j in range(3):
        V_full[j*7+6, 2*rank33:2*rank33+9] = V31[j, :]
    for i in range(3):
        W_full[6*3+i, 2*rank33:2*rank33+9] = W31[i, :]
    
    return U_full, V_full, W_full

def _verify_full(U, V, W, n, m, p):
    """Verify the full decomposition."""
    T = _build_tensor(n, m, p)
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2 x 2x2.
    Returns (U, V, W) each of shape (4, 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (c+d)e
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = -1; W[3,1] = -1
    
    # M3 = a(f-h)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # M4 = d(g-e)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (a+b)h
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (c-a)(e+f)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (b-d)(g+h)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _try_rank_decomp_3x3x3(rank, max_time=10.0):
    """Try to find a rank-r decomposition for 3x3x3 using randomized search."""
    if _time_left() < max_time + 5:
        return None
    
    T = _build_tensor(3, 3, 3)
    T_norm = np.linalg.norm(T)
    
    np.random.seed(42)
    best_err = float('inf')
    best_UVW = None
    start = time.time()
    
    for trial in range(50):
        if time.time() - start > max_time:
            break
        if _time_left() < 10:
            break
        
        U = np.random.randn(9, rank) * 0.5
        V = np.random.randn(9, rank) * 0.5
        W = np.random.randn(9, rank) * 0.5
        
        for it in range(50):
            # Update U
            for r in range(rank):
                Vr = V[:, r]
                Wr = W[:, r]
                denom = np.sum(Vr**2) * np.sum(Wr**2)
                if denom < 1e-12:
                    continue
                if rank > 1:
                    R = T - np.einsum('ir,jr,kr->ijk', 
                                     np.delete(U, r, axis=1), 
                                     np.delete(V, r, axis=1), 
                                     np.delete(W, r, axis=1))
                else:
                    R = T
                U[:, r] = np.einsum('ijk,j,k->i', R, Vr, Wr) / denom
            
            # Update V
            for r in range(rank):
                Ur = U[:, r]
                Wr = W[:, r]
                denom = np.sum(Ur**2) * np.sum(Wr**2)
                if denom < 1e-12:
                    continue
                if rank > 1:
                    R = T - np.einsum('ir,jr,kr->ijk', 
                                     np.delete(U, r, axis=1), 
                                     np.delete(V, r, axis=1), 
                                     np.delete(W, r, axis=1))
                else:
                    R = T
                V[:, r] = np.einsum('ijk,i,k->j', R, Ur, Wr) / denom
            
            # Update W
            for r in range(rank):
                Ur = U[:, r]
                Vr = V[:, r]
                denom = np.sum(Ur**2) * np.sum(Vr**2)
                if denom < 1e-12:
                    continue
                if rank > 1:
                    R = T - np.einsum('ir,jr,kr->ijk', 
                                     np.delete(U, r, axis=1), 
                                     np.delete(V, r, axis=1), 
                                     np.delete(W, r, axis=1))
                else:
                    R = T
                W[:, r] = np.einsum('ijk,i,j->k', R, Ur, Vr) / denom
            
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - recon) / T_norm
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            if err < 1e-8:
                break
    
    if best_UVW is not None and best_err < 1e-6:
        U, V, W = best_UVW
        # Try to snap to half-integers
        U_h = np.round(U * 2) / 2.0
        V_h = np.round(V * 2) / 2.0
        W_h = np.round(W * 2) / 2.0
        recon_h = np.einsum('ir,jr,kr->ijk', U_h, V_h, W_h)
        err_h = np.linalg.norm(T - recon_h) / T_norm
        if err_h < 1e-6:
            return U_h, V_h, W_h
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a good 3x3 x 3x3 decomposition
    U33, V33, W33, rank33 = None, None, None, None
    
    # Try ranks from low to high
    for target_rank in [19, 20, 21, 22, 23]:
        if _time_left() < 20.0:
            break
        result = _try_rank_decomp_3x3x3(target_rank, max_time=8.0)
        if result is not None:
            U33, V33, W33 = result
            rank33 = target_rank
            break
    
    if U33 is None:
        # Fall back to schoolbook for 3x3x3
        U33, V33, W33, rank33 = _schoolbook(3, 3, 3)
    
    # Build the full 3x3 x 3x7 decomposition
    U_full, V_full, W_full = _build_full_3x3x7(U33, V33, W33, rank33)
    err = _verify_full(U_full, V_full, W_full, n, m, p)
    
    if err < 1e-6:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Final fallback: schoolbook for the full problem
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
