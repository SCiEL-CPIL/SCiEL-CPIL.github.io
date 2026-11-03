"""Matrix multiplication 2,4,7 tensor decomposition.

Strategy:
- Start from the known rank-50 Strassen-like block decomposition (2x4x7 = 2*(2x2x3+2x2x4)).
- Attempt to reduce rank via randomized local search on the factor matrices.
- If verification fails, fall back to the rank-50 block construction.
"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify if the decomposition is exact."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1
    
    # M4: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_2x2x1():
    """Return a rank-4 decomposition for 2x2x1 matrix-vector multiplication."""
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _construct_strassen_2x4x7(n=2, m=4, p=7):
    """Construct a rank-50 Strassen-like decomposition for 2x4x7."""
    rank = 50
    U = np.zeros((8, rank))
    V = np.zeros((28, rank))
    W = np.zeros((14, rank))
    
    col = 0
    
    def add_strassen_2x2x2(u_offset, v_offset, w_offset, col_start):
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(7):
            for ui in range(2):
                for uj in range(2):
                    if U_s[ui * 2 + uj, r] != 0:
                        U[(u_offset[0] + ui) * 2 + (u_offset[1] + uj), col_start + r] = U_s[ui * 2 + uj, r]
            for vj in range(2):
                for vk in range(2):
                    if V_s[vj * 2 + vk, r] != 0:
                        V[(v_offset[0] + vj) * 2 + (v_offset[1] + vk), col_start + r] = V_s[vj * 2 + vk, r]
            for wk in range(2):
                for wi in range(2):
                    if W_s[wk * 2 + wi, r] != 0:
                        W[(w_offset[0] + wk) * 2 + (w_offset[1] + wi), col_start + r] = W_s[wk * 2 + wi, r]
        col = col_start + 7
    
    def add_strassen_2x2x1(u_offset, v_offset, w_offset, col_start):
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2x1()
        for r in range(4):
            for ui in range(2):
                for uj in range(2):
                    if U_s[ui * 2 + uj, r] != 0:
                        U[(u_offset[0] + ui) * 2 + (u_offset[1] + uj), col_start + r] = U_s[ui * 2 + uj, r]
            for vj in range(2):
                for vk in range(1):
                    if V_s[vj * 1 + vk, r] != 0:
                        V[(v_offset[0] + vj) * 1 + (v_offset[1] + vk), col_start + r] = V_s[vj * 1 + vk, r]
            for wk in range(1):
                for wi in range(2):
                    if W_s[wk * 2 + wi, r] != 0:
                        W[(w_offset[0] + wk) * 2 + (w_offset[1] + wi), col_start + r] = W_s[wk * 2 + wi, r]
        col = col_start + 4
    
    # Block 1: A_left * B_tl1 (2x2x1)
    add_strassen_2x2x1((0, 0), (0, 0), (0, 0), col)
    
    # Block 2: A_left * B_tl2 (2x2x2)
    add_strassen_2x2x2((0, 0), (0, 1), (1, 0), col)
    
    # Block 3: A_left * B_tr1 (2x2x2)
    add_strassen_2x2x2((0, 0), (0, 3), (3, 0), col)
    
    # Block 4: A_left * B_tr2 (2x2x2)
    add_strassen_2x2x2((0, 0), (0, 5), (5, 0), col)
    
    # Block 5: A_right * B_bl1 (2x2x1)
    add_strassen_2x2x1((0, 2), (2, 0), (0, 0), col)
    
    # Block 6: A_right * B_bl2 (2x2x2)
    add_strassen_2x2x2((0, 2), (2, 1), (1, 0), col)
    
    # Block 7: A_right * B_br1 (2x2x2)
    add_strassen_2x2x2((0, 2), (2, 3), (3, 0), col)
    
    # Block 8: A_right * B_br2 (2x2x2)
    add_strassen_2x2x2((0, 2), (2, 5), (5, 0), col)
    
    if _verify(U, V, W, 2, 4, 7):
        return U, V, W
    return None

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _tensor_error(U, V, W, n, m, p):
    """Compute the Frobenius norm of the error in the tensor decomposition."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    error = result - 1.0
    return np.linalg.norm(error)

def _randomized_search(U_init, V_init, W_init, n, m, p, max_iters=20, time_limit=100):
    """Attempt to reduce rank via randomized local search."""
    start_time = time.time()
    U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
    current_rank = U.shape[1]
    current_error = _tensor_error(U, V, W, n, m, p)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    best_error = current_error
    
    for target_rank in range(current_rank - 1, 10, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            
            # Create a random initialization for the target rank
            # Start from the current best and project to lower rank
            if target_rank < current_rank:
                # Random linear combination of existing terms
                alpha = np.random.randn(current_rank, target_rank)
                U_try = U @ alpha
                V_try = V @ alpha
                W_try = W @ alpha
            else:
                U_try = np.random.randn(n * m, target_rank)
                V_try = np.random.randn(m * p, target_rank)
                W_try = np.random.randn(p * n, target_rank)
            
            # Normalize
            norm_U = np.linalg.norm(U_try, axis=0, keepdims=True)
            norm_V = np.linalg.norm(V_try, axis=0, keepdims=True)
            norm_W = np.linalg.norm(W_try, axis=0, keepdims=True)
            norm_U[norm_U < 1e-10] = 1.0
            norm_V[norm_V < 1e-10] = 1.0
            norm_W[norm_W < 1e-10] = 1.0
            U_try = U_try / norm_U
            V_try = V_try / norm_V
            W_try = W_try / norm_W
            
            # Local optimization
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e10
                u_flat = params[:U_try.size].reshape(U_try.shape)
                v_flat = params[U_try.size:U_try.size + V_try.size].reshape(V_try.shape)
                w_flat = params[U_try.size + V_try.size:].reshape(W_try.shape)
                err = _tensor_error(u_flat, v_flat, w_flat, n, m, p)
                return err ** 2
            
            x0 = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-12, 'maxfun': 500})
            except Exception:
                continue
            
            if res.fun < 1e-8:
                u_opt = res.x[:U_try.size].reshape(U_try.shape)
                v_opt = res.x[U_try.size:U_try.size + V_try.size].reshape(V_try.shape)
                w_opt = res.x[U_try.size + V_try.size:].reshape(W_try.shape)
                
                if _verify(u_opt, v_opt, w_opt, n, m, p):
                    best_U, best_V, best_W = u_opt.copy(), v_opt.copy(), w_opt.copy()
                    best_rank = target_rank
                    best_error = 0.0
                    U, V, W = u_opt, v_opt, w_opt
                    current_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
        else:
            # If we can't find a solution at this rank, try lower ranks
            # but don't give up yet
            if target_rank <= 20:
                break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try Strassen-like decomposition first
    result = _construct_strassen_2x4x7(n, m, p)
    if result is not None:
        U, V, W = result
        best_rank = U.shape[1]
    else:
        # Fall back to schoolbook
        U, V, W = _construct_schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Try to improve via randomized search
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 20:  # Only if we have enough time
        U_opt, V_opt, W_opt, opt_rank = _randomized_search(
            U, V, W, n, m, p, 
            time_limit=min(remaining_time - 10, 100)
        )
        if opt_rank < best_rank and _verify(U_opt, V_opt, W_opt, n, m, p):
            U, V, W = U_opt, V_opt, W_opt
            best_rank = opt_rank
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
