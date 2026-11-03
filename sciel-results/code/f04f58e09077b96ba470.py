"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-10):
    R = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(R - T)) < tol

def _round_to_half_integer(M, tol=1e-8):
    """Round matrix to nearest half-integers and verify."""
    M_rounded = np.round(M * 2) / 2
    return M_rounded

def _schoolbook():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _to_list(M):
    return [[float(x) for x in row] for row in M]

def _try_rank_decomposition(T, rank, nm, mp, pn, deadline, max_trials=200, max_iters=300):
    """Try to find a rank-rank decomposition using alternating minimization."""
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        seed = trial * 1000003 + rank * 7919
        np.random.seed(seed)
        
        # Random initialization with small scale
        U = np.random.randn(nm, rank) * 0.3
        V = np.random.randn(mp, rank) * 0.3
        W = np.random.randn(pn, rank) * 0.3
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Solve for U
            VW = np.einsum('br,cr->bcr', V, W)  # (mp, pn, rank)
            VW_flat = VW.transpose(2, 0, 1).reshape(rank, mp * pn)  # (rank, mp*pn)
            T_U = T.reshape(nm, -1)  # (nm, mp*pn)
            
            try:
                # U[a,:] should satisfy: sum_r U[a,r] * VW_flat[r, bc] = T_U[a, bc]
                # This is: VW_flat.T @ U[a,:] = T_U[a,:]
                # U.T = pinv(VW_flat.T) @ T_U.T
                U_new = (np.linalg.pinv(VW_flat.T) @ T_U.T).T
            except np.linalg.LinAlgError:
                break
            U = U_new
            
            # Solve for V
            UW = np.einsum('ar,cr->acr', U, W)  # (nm, pn, rank)
            UW_flat = UW.transpose(2, 0, 1).reshape(rank, nm * pn)  # (rank, nm*pn)
            T_V = T.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
            
            try:
                V_new = (np.linalg.pinv(UW_flat.T) @ T_V.T).T
            except np.linalg.LinAlgError:
                break
            V = V_new
            
            # Solve for W
            UV = np.einsum('ar,br->abr', U, V)  # (nm, mp, rank)
            UV_flat = UV.transpose(2, 0, 1).reshape(rank, nm * mp)  # (rank, nm*mp)
            T_W = T.transpose(2, 0, 1).reshape(pn, -1)  # (pn, nm*mp)
            
            try:
                W_new = (np.linalg.pinv(UV_flat.T) @ T_W.T).T
            except np.linalg.LinAlgError:
                break
            W = W_new
            
            # Check convergence periodically
            if it % 20 == 0 or it == max_iters - 1:
                R = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(R - T))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-12:
                    break
        
        # Check if we found a good solution
        R = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
        err = np.max(np.abs(R - T))
        
        if err < 1e-8:
            # Try to round to half-integers
            U_rounded = _round_to_half_integer(best_U)
            V_rounded = _round_to_half_integer(best_V)
            W_rounded = _round_to_half_integer(best_W)
            
            R_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            if np.max(np.abs(R_check - T)) < 1e-10:
                return rank, U_rounded, V_rounded, W_rounded
        elif err < 1e-6:
            # Try more aggressive rounding
            for scale in [1, 2, 4, 8]:
                U_rounded = np.round(best_U * scale) / scale
                V_rounded = np.round(best_V * scale) / scale
                W_rounded = np.round(best_W * scale) / scale
                
                R_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(R_check - T)) < 1e-10:
                    return rank, U_rounded, V_rounded, W_rounded
    
    return None, None, None, None

def _structured_search(T, nm, mp, pn, deadline):
    """
    Use structured initialization based on the matrix multiplication structure.
    The tensor has a specific pattern: T[a,b,c]=1 where a=i*m+j, b=j*p+k, c=k*n+i
    for i in [0,n), j in [0,m), k in [0,p).
    
    This structure suggests using decompositions that respect the block structure.
    """
    n, m, p = DIMENSIONS
    
    # Try rank = n*m + m*p + p*n - n - m - p + 1 (Strassen-like for general case)
    # For 2x3x6: 6 + 18 + 12 - 2 - 3 - 6 + 1 = 26
    # But let's try a range of structured ranks
    
    # First, try a direct construction based on the structure
    # The tensor can be viewed as: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try rank = m*p + n*p + n*m - something
    # Actually, let's try a systematic approach with better initialization
    
    for rank in range(18, 37):
        if time.time() > deadline:
            break
        
        result = _try_rank_decomposition(T, rank, nm, mp, pn, deadline, max_trials=50, max_iters=200)
        if result[0] is not None:
            return result
    
    return None, None, None, None

def construct():
    T = _build_target_tensor()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # First, try the structured search
    rank, U, V, W = _structured_search(T, nm, mp, pn, deadline)
    
    if rank is not None:
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
