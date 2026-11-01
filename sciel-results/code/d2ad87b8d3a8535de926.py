"""Matrix multiplication 2x3x9 — rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 9]


def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Return True if U,V,W exactly decompose the matrix multiplication tensor.

    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 if (i,j,k) is a valid index, 0 otherwise.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _validate_exact(U, V, W, n=2, m=3, p=9):
    """Exact validation using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True


def _try_als_decomposition(n, m, p, target_rank, iterations=20, seed=42):
    """
    Attempt to find a decomposition of rank target_rank using Alternating Least Squares.
    Returns (U, V, W) if successful (residual small), else None.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W
    U = rng.randn(n * m, target_rank)
    V = rng.randn(m * p, target_rank)
    W = rng.randn(p * n, target_rank)
    
    for it in range(iterations):
        # Update W
        new_W = np.zeros_like(W)
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros((m, target_rank))
                for j in range(m):
                    A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b_ik = np.ones(m)
                try:
                    w_ik, residuals, rank, sv = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                    new_W[k * n + i, :] = w_ik
                except np.linalg.LinAlgError:
                    pass
        W = new_W
        
        # Update U
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A_ij = np.zeros((p, target_rank))
                for k in range(p):
                    A_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b_ij = np.ones(p)
                try:
                    u_ij, residuals, rank, sv = np.linalg.lstsq(A_ij, b_ij, rcond=None)
                    new_U[i * m + j, :] = u_ij
                except np.linalg.LinAlgError:
                    pass
        U = new_U
        
        # Update V
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_jk = np.zeros((n, target_rank))
                for i in range(n):
                    A_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b_jk = np.ones(n)
                try:
                    v_jk, residuals, rank, sv = np.linalg.lstsq(A_jk, b_jk, rcond=None)
                    new_V[j * p + k, :] = v_jk
                except np.linalg.LinAlgError:
                    pass
        V = new_V
        
        # Check residual every few iterations
        if it % 5 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(s - 1.0) > max_err:
                            max_err = abs(s - 1.0)
            if max_err < 1e-6:
                return U.tolist(), V.tolist(), W.tolist()

    return None

def construct():
    """Return a rank-minimal exact decomposition of the 2x3x9 matrix mult tensor."""
    t_start = time.time()
    DEADLINE = t_start + 110.0

    n, m, p = 2, 3, 9
    rank_sb = n * m * p # 54

    # Schoolbook
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try to find lower rank using ALS
    target_ranks = [50, 45, 40, 36, 30, 27]
    
    for target_rank in target_ranks:
        if time.time() > DEADLINE:
            break
            
        found = None
        for seed in range(5):
            if time.time() > DEADLINE:
                break
            res = _try_als_decomposition(n, m, p, target_rank, iterations=20, seed=seed)
            if res:
                U_list, V_list, W_list = res
                # Round to nearest integer or half-integer
                U_np = np.array(U_list)
                V_np = np.array(V_list)
                W_np = np.array(W_list)
                
                # Try integer
                U_int = np.round(U_np).astype(int)
                V_int = np.round(V_np).astype(int)
                W_int = np.round(W_np).astype(int)
                
                if _validate(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    if _validate_exact(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                        found = (U_int.tolist(), V_int.tolist(), W_int.tolist())
                        break
                        
                # Try half-integer
                U_half = np.round(U_np * 2) / 2
                V_half = np.round(V_np * 2) / 2
                W_half = np.round(W_np * 2) / 2
                
                if _validate(U_half.tolist(), V_half.tolist(), W_half.tolist()):
                    if _validate_exact(U_half.tolist(), V_half.tolist(), W_half.tolist()):
                        found = (U_half.tolist(), V_half.tolist(), W_half.tolist())
                        break
                        
        if found:
            U_f, V_f, W_f = found
            if target_rank < best_rank:
                best_rank = target_rank
                best_result = {"U": U_f, "V": V_f, "W": W_f}
            if best_rank <= 27:
                break

    return best_result
