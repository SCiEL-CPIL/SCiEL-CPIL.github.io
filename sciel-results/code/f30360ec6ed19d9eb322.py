import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = 130
    
    start_time = time.time()
    
    # Build the true multiplication tensor
    dim_a = n * m  # 8
    dim_b = m * p  # 16
    dim_c = p * n  # 8
    
    T_true = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    # Try to find lower rank decompositions (24, 25, 26, 27)
    for rank in [24, 25, 26, 27]:
        if time.time() > deadline - 10:
            break
            
        # Run ALS with multiple restarts
        for restart in range(100):
            if time.time() > deadline - 5:
                break
                
            result = _als_decompose(T_true, rank, dim_a, dim_b, dim_c, restart)
            if result is not None:
                return result
                
    # Fallback to rank 28
    return _strassen_28()

def _als_decompose(T, rank, dim_a, dim_b, dim_c, seed):
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(dim_a, rank)
    V = np.random.randn(dim_b, rank)
    W = np.random.randn(dim_c, rank)
    
    # Normalize to prevent overflow
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 0 and norm_v > 0 and norm_w > 0:
            U[:, r] /= norm_u
            V[:, r] /= norm_v
            W[:, r] /= norm_w
            
    for it in range(50):
        # Update U
        M = np.zeros((rank, dim_b * dim_c))
        for r in range(rank):
            M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
        T_flat = T.reshape(dim_a, dim_b * dim_c)
        
        try:
            U, residuals, rank_u, s = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
        except:
            continue
            
        # Update V
        M = np.zeros((rank, dim_a * dim_c))
        for r in range(rank):
            M[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
        T_flat_v = T.transpose(1, 0, 2).reshape(dim_b, dim_a * dim_c)
        
        try:
            V, residuals, rank_v, s = np.linalg.lstsq(M.T, T_flat_v.T, rcond=None)
        except:
            continue
            
        # Update W
        M = np.zeros((rank, dim_a * dim_b))
        for r in range(rank):
            M[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
        T_flat_w = T.transpose(2, 0, 1).reshape(dim_c, dim_a * dim_b)
        
        try:
            W, residuals, rank_w, s = np.linalg.lstsq(M.T, T_flat_w.T, rcond=None)
        except:
            continue
            
        # Check convergence and rounding
        T_recon = np.zeros((dim_a, dim_b, dim_c))
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        error = np.linalg.norm(T_recon - T)
        if error < 1e-6:
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_check = np.zeros((dim_a, dim_b, dim_c))
            for r in range(rank):
                T_check += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                
            if np.allclose(T_check, T, atol=1e-6):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
    return None

def _strassen_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Standard 2x2 Strassen blocks
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1 = (A00+A11)(B00+B11)
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2 = (A10+A11)B00
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3 = A00(B10-B11)
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4 = A11(B01-B10)
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5 = (A00+A01)B11
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6 = (A10-A00)(B00+B01)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7 = (A01-A11)(B10+B11)
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off):
        for s in range(7):
            col = off + s
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c
    
    # Four 2x2 blocks
    emit_block(0, 0, 0, 0, 0)    # Top-left
    emit_block(0, 0, 2, 2, 7)    # Top-right
    emit_block(2, 2, 0, 0, 14)   # Bottom-left
    emit_block(2, 2, 2, 2, 21)   # Bottom-right
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    
    # Build the tensor from the decomposition
    T_decomp = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    
    for r in range(rank):
        u_col = np.array([U[i][r] for i in range(dim_a)], dtype=np.float64)
        v_col = np.array([V[j][r] for j in range(dim_b)], dtype=np.float64)
        w_col = np.array([W[k][r] for k in range(dim_c)], dtype=np.float64)
        
        T_decomp += np.einsum('a,b,c->abc', u_col, v_col, w_col)
    
    # Check against the true tensor
    T_true = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    if not np.allclose(T_decomp, T_true, atol=1e-8):
        return False
    
    # Check that entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
    
    return True

def _try_rank_27_nnls(T_flat, T_true, dim_a, dim_b, dim_c, trial):
    """Try to find a rank-27 decomposition using NNLS approach."""
    n, m, p = DIMENSIONS
    np.random.seed(42 + trial)
    
    # Strategy: Start from rank-28 and try to merge two terms
    # or use NNLS to find sparse decomposition
    
    # First approach: Try to find a rank-27 by solving a sparse problem
    # Flatten the tensor differently: for each c, we have a matrix M_c of shape (dim_a * dim_b)
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For fixed c: T[:, :, c] = sum_r W[c,r] * (U[:,r] * V[:,r])^T
    
    # Try a randomized approach: start from rank-28, try to eliminate one term
    U28 = np.array(_strassen_28()["U"], dtype=np.float64)
    V28 = np.array(_strassen_28()["V"], dtype=np.float64)
    W28 = np.array(_strassen_28()["W"], dtype=np.float64)
    
    # Try each column as a candidate to eliminate
    for col_to_elim in range(28):
        if time.time() > 130:
            return None
        
        # Create rank-27 by removing this column
        mask = [i for i in range(28) if i != col_to_elim]
        U27 = U28[:, mask].copy()
        V27 = V28[:, mask].copy()
        W27 = W28[:, mask].copy()
        
        # Add small random perturbations and try to fix
        for _ in range(50):
            if time.time() > 130:
                return None
            
            # Random perturbation
            U27 += np.random.normal(0, 0.001, U27.shape)
            V27 += np.random.normal(0, 0.001, V27.shape)
            W27 += np.random.normal(0, 0.001, W27.shape)
            
            # Check if valid
            if _verify_decomposition(U27.tolist(), V27.tolist(), W27.tolist()):
                return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Second approach: Use NNLS on the flattened tensor
    # For each c, try to decompose T[:, :, c] as sum_r W[c,r] * (U[:,r] * V[:,r])^T
    # This is equivalent to finding low-rank approximations
    
    # Try a different strategy: use the fact that the tensor has structure
    # and try to find a rank-27 decomposition by solving a system of equations
    
    # For now, return None if we couldn't find rank 27
    return None

def _try_rank_26_nnls(T_flat, T_true, dim_a, dim_b, dim_c, trial):
    """Try to find a rank-26 decomposition."""
    # More aggressive search for rank 26
    return None

# EVOLVE-BLOCK-END
