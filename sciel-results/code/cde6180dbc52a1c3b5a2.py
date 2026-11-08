"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import itertools
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for r in range(rank):
        T += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(mat):
    """Check if all entries are integer or half-integer (i.e., 2*x is integer)."""
    for row in mat:
        for val in row:
            if isinstance(val, (int, np.integer)):
                continue
            v = float(val)
            if abs(2 * v - round(2 * v)) > 1e-9:
                return False
    return True

def _to_rational_list(mat):
    """Convert a matrix of floats (that are half-integers) to list of Fractions."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = float(val)
            r.append(Fraction(round(2 * v), 2))
        result.append(r)
    return result

def all_half_int_arr(arr, tol=1e-6):
    """Check if all entries in array are integer or half-integer."""
    for val in arr.flat:
        v = float(val)
        if abs(2 * v - round(2 * v)) > tol:
            return False
    return True

def _to_list(arr):
    """Convert numpy array to list of lists with rounded half-integer values."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = float(val)
            rv = round(2 * v) / 2.0
            r.append(rv)
        result.append(r)
    return result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _try_structural_search(n, m, p, time_limit=100.0):
    """
    Try to find a rank-reduced decomposition using structural knowledge.
    Uses a faster approach: exploit the sparse structure of the tensor.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # Compute lower bounds using matricization ranks
    T_mat1 = T.reshape(n*m, m*p * p*n)
    T_mat2 = T.reshape(n*m * m*p, p*n)
    T_mat3 = T.reshape(n*m * p*n, m*p)
    
    rank_lb1 = np.linalg.matrix_rank(T_mat1)
    rank_lb2 = np.linalg.matrix_rank(T_mat2)
    rank_lb3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(rank_lb1, rank_lb2, rank_lb3)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from lower bound up to schoolbook
    for r in range(max(rank_lower, 30), n * m * p + 1):
        if time.time() - start_time > time_limit:
            break
        if r >= best_rank:
            continue
            
        found = False
        # Try a few random initializations with different W patterns
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Use different random W strategies
            if attempt < 30:
                W_rand = np.random.choice([-1, 0, 1], size=(p * n, r))
            elif attempt < 60:
                W_rand = np.random.randn(p * n, r) * 0.5
            else:
                W_rand = np.random.choice([-2, -1, 0, 1, 2], size=(p * n, r))
            
            if np.linalg.matrix_rank(W_rand) < r:
                continue
            
            try:
                WtW = W_rand.T @ W_rand
                WtW_inv = np.linalg.inv(WtW)
            except:
                continue
            
            # Compute X_vec using einsum for speed
            TW = np.einsum('abc,cr->abr', T.astype(np.float64), W_rand)
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            
            # Check if each slice X_vec[:,:,r_idx] is rank 1
            ok_all_rank1 = True
            U_sol = np.zeros((n * m, r))
            V_sol = np.zeros((m * p, r))
            
            for r_idx in range(r):
                C_r = X_vec[:, :, r_idx]
                if C_r.size == 0:
                    continue
                svals = np.linalg.svd(C_r, compute_uv=False)
                if len(svals) > 1 and svals[1] > 1e-8 * max(svals[0], 1e-12):
                    ok_all_rank1 = False
                    break
                if svals[0] < 1e-12:
                    U_sol[:, r_idx] = 0
                    V_sol[:, r_idx] = 0
                else:
                    U_sv, S_sv, Vt_sv = np.linalg.svd(C_r)
                    U_sol[:, r_idx] = U_sv[:, 0] * np.sqrt(S_sv[0])
                    V_sol[:, r_idx] = Vt_sv[0, :] * np.sqrt(S_sv[0])
            
            if not ok_all_rank1:
                continue
            
            # Verify the full decomposition
            T_reconstructed = np.zeros_like(T, dtype=np.float64)
            for r_idx in range(r):
                T_reconstructed += np.outer(np.outer(U_sol[:, r_idx], V_sol[:, r_idx]), W_rand[:, r_idx])
            
            if np.allclose(T_reconstructed, T.astype(np.float64), atol=1e-6):
                # Check if entries are half-integers
                all_half_int = all_half_int_arr(U_sol) and all_half_int_arr(V_sol) and all_half_int_arr(W_rand)
                if all_half_int:
                    best_rank = r
                    best_U = U_sol
                    best_V = V_sol
                    best_W = W_rand
                    found = True
                    break
        
        if found:
            # Continue searching for smaller ranks
            pass
    
    if best_U is not None:
        U_out = _to_list(best_U)
        V_out = _to_list(best_V)
        W_out = _to_list(best_W)
        return U_out, V_out, W_out
    
    return None

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better decomposition
    try:
        result = _try_structural_search(n, m, p, time_limit=100.0)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
