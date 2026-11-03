"""Matrix multiplication 2,4,7 tensor decomposition.

Strategy:
- Construct the full 56-dimensional tensor T[i,j,k] = 1 if i in [0,2), j in [0,4), k in [0,7)
  and (i,j,k) corresponds to a multiplication term C[i,k] = sum_j A[i,j]*B[j,k].
  Actually T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matrix mult).
  
  The tensor is: T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6}.
  Wait, that's not right. The matrix multiplication tensor T_{ijk} is defined such that
  (C)_{ik} = sum_j A_{ij} B_{jk}, so T_{ijk} = 1 for all valid i,j,k.

  Actually the matmul tensor has T[i,j,k] = 1 for all i,j,k in their ranges.
  The decomposition U*V*W must satisfy:
  sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.

- Start from the rank-50 Strassen-like block construction.
- Use randomized rank-reduction: try to find a valid decomposition at rank R < 50
  by starting with random initializations and using iterative optimization.
- The key insight for structural improvement: instead of using L-BFGS-B on the full
  parameter vector (which is slow), use an alternating least squares approach where
  we fix two factor matrices and solve for the third via linear algebra.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]
DEADLINE = 145  # Safety margin before 150s kill

def _tensor_entries(n, m, p):
    """Return list of (i,j,k) indices for the matmul tensor."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

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

def _als_step_U(V, W, n, m, p, rank):
    """Given V and W, find U that minimizes ||T - U*V*W||_F.
    
    We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
    
    For fixed V,W, this is linear in U. Let's expand:
    For each (i,j): sum_k [sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]] = 1
    
    Let's define for each (i,j): 
    sum_r U[i,j,r] * (sum_k V[j,k,r] * W[k,i,r]) = 1
    
    So if we define M[i,j,r] = sum_k V[j,k,r] * W[k,i,r], then:
    sum_r U[i,j,r] * M[i,j,r] = 1
    
    This is an underdetermined system (one equation, `rank` unknowns per (i,j)).
    We want the minimum-norm solution: U[i,j,:] = M[i,j,:] / sum(M[i,j,:]^2)
    """
    U = np.zeros((n * m, rank), dtype=np.float64)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            # M[i,j,r] = sum_k V[j,k,r] * W[k,i,r]
            M = np.zeros(rank, dtype=np.float64)
            for k in range(p):
                M += V_[j, k, :] * W_[k, i, :]
            norm_sq = np.dot(M, M)
            if norm_sq > 1e-12:
                U[i * m + j, :] = M / norm_sq
    
    return U

def _als_step_V(U, W, n, m, p, rank):
    """Given U and W, find V that minimizes ||T - U*V*W||_F."""
    V = np.zeros((m * p, rank), dtype=np.float64)
    U_ = U.reshape(n, m, rank)
    W_ = W.reshape(p, n, rank)
    
    for j in range(m):
        for k in range(p):
            # sum_r V[j,k,r] * (sum_i U[i,j,r] * W[k,i,r]) = 1
            M = np.zeros(rank, dtype=np.float64)
            for i in range(n):
                M += U_[i, j, :] * W_[k, i, :]
            norm_sq = np.dot(M, M)
            if norm_sq > 1e-12:
                V[j * p + k, :] = M / norm_sq
    
    return V

def _als_step_W(U, V, n, m, p, rank):
    """Given U and V, find W that minimizes ||T - U*V*W||_F."""
    W = np.zeros((p * n, rank), dtype=np.float64)
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    
    for k in range(p):
        for i in range(n):
            # sum_r W[k,i,r] * (sum_j U[i,j,r] * V[j,k,r]) = 1
            M = np.zeros(rank, dtype=np.float64)
            for j in range(m):
                M += U_[i, j, :] * V_[j, k, :]
            norm_sq = np.dot(M, M)
            if norm_sq > 1e-12:
                W[k * n + i, :] = M / norm_sq
    
    return W

def _als_optimize(U_init, V_init, W_init, n, m, p, rank, max_iters=200, tol=1e-10):
    """Alternating least squares optimization for the tensor decomposition."""
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    
    prev_err = _tensor_error(U, V, W, n, m, p)
    
    for it in range(max_iters):
        U_new = _als_step_U(V, W, n, m, p, rank)
        V_new = _als_step_V(U_new, W, n, m, p, rank)
        W_new = _als_step_W(U_new, V_new, n, m, p, rank)
        
        err = _tensor_error(U_new, V_new, W_new, n, m, p)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        U, V, W = U_new, V_new, W_new
    
    return U, V, W

def _construct_strassen_2x4x7(n=2, m=4, p=7):
    """Construct a rank-50 Strassen-like decomposition for 2x4x7."""
    
    def _strassen_2x2x2():
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
        rank = 4
        U = np.zeros((4, rank))
        V = np.zeros((2, rank))
        W = np.zeros((2, rank))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

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

def _random_rank_reduction(U0, V0, W0, n, m, p, start_rank, min_rank, time_limit):
    """Try to reduce rank using ALS with random restarts."""
    start_time = time.time()
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = start_rank
    best_err = _tensor_error(U0, V0, W0, n, m, p)
    
    for target_rank in range(start_rank - 1, max(min_rank - 1, 5), -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        n_restarts = 5 if target_rank <= 40 else 3
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random perturbation of the current best
            scale = 0.1 + 0.2 * restart
            U_init = U0.copy() + scale * np.random.randn(U0.shape[0], target_rank)
            V_init = V0.copy() + scale * np.random.randn(V0.shape[0], target_rank)
            W_init = W0.copy() + scale * np.random.randn(W0.shape[0], target_rank)
            
            # Ensure proper shapes
            U_init = U_init[:, :target_rank]
            V_init = V_init[:, :target_rank]
            W_init = W_init[:, :target_rank]
            
            # Run ALS
            U_opt, V_opt, W_opt = _als_optimize(U_init, V_init, W_init, n, m, p, target_rank, max_iters=300)
            
            err = _tensor_error(U_opt, V_opt, W_opt, n, m, p)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                best_rank = target_rank
            
            if err < 1e-8 and _verify(U_opt, V_opt, W_opt, n, m, p):
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
        elif best_err < 1e-6:
            # Very close, might be numerical issues; keep going
            continue
        else:
            # If error is large, we likely can't reduce further
            if target_rank <= 30:
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
    
    # Try to improve via ALS-based rank reduction
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 15:
        U_opt, V_opt, W_opt, opt_rank = _random_rank_reduction(
            U, V, W, n, m, p,
            start_rank=best_rank,
            min_rank=10,
            time_limit=min(remaining_time - 8, 130)
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
