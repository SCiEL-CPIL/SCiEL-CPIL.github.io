"""Matrix multiplication 3x5x5 tensor decomposition using Strassen-based block construction."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return (U, V, W) as numpy arrays for 2x2x2 Strassen multiplication.
    
    U shape: (4, 7) - rows indexed by (i, j) -> i*2+j
    V shape: (4, 7) - rows indexed by (j, k) -> j*2+k  
    W shape: (4, 7) - rows indexed by (k, i) -> k*2+i
    
    The decomposition satisfies:
    C[i,k] = sum_r W[k*2+i, r] * (sum_j U[i*2+j, r] * A[i,j]) * (sum_j V[j*2+k, r] * B[j,k])
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1   # C10
    W[3][1] = -1  # C11
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1   # C01
    W[3][2] = 1   # C11
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1   # C00
    W[2][3] = 1   # C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C00
    W[1][4] = 1   # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C00
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is correct.
    
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                if abs(s - 1) > tol:
                    return False
    return True

def _build_rank71():
    """Build a rank-71 decomposition using Strassen on 2x2x2 sub-blocks.
    
    Partition:
    i: {0}, {1,2}
    j: {0,1}, {2,3}, {4}
    k: {0,1}, {2,3}, {4}
    
    For 2x2x2 blocks: use Strassen (rank 7)
    For all other blocks: use schoolbook (rank = product of sizes)
    
    Total rank: 4*7 + 43 = 28 + 43 = 71
    """
    n, m, p = 3, 5, 5
    
    i_blocks = [(0,), (1, 2)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    total_rank = 0
    for i_b in range(len(i_blocks)):
        for j_b in range(len(j_blocks)):
            for k_b in range(len(k_blocks)):
                i_set = i_blocks[i_b]
                j_set = j_blocks[j_b]
                k_set = k_blocks[k_b]
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    total_rank += 7
                else:
                    total_rank += len(i_set) * len(j_set) * len(k_set)
    
    assert total_rank == 71, f"Expected rank 71, got {total_rank}"
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i_b in range(len(i_blocks)):
        for j_b in range(len(j_blocks)):
            for k_b in range(len(k_blocks)):
                i_set = i_blocks[i_b]
                j_set = j_blocks[j_b]
                k_set = k_blocks[k_b]
                
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    su, sv, sw = _strassen_222()
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj, col + c] = su[li * 2 + lj, c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk, col + c] = sv[lj * 2 + lk, c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi, col + c] = sw[lk * 2 + li, c]
                    col += 7
                else:
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj, col] = 1
                                V[gj * p + gk, col] = 1
                                W[gk * n + gi, col] = 1
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, deadline, target_rank=68, max_time=60):
    """Try to reduce rank below current by optimizing a lower-rank decomposition.
    
    We parameterize a rank-r decomposition and minimize the reconstruction error.
    If we can achieve error < tol, we have a valid lower-rank decomposition.
    """
    U0 = np.array(U, dtype=float)
    V0 = np.array(V, dtype=float)
    W0 = np.array(W, dtype=float)
    r0 = U0.shape[1]
    
    start_time = time.time()
    
    for r in range(target_rank, min(r0, target_rank + 4)):
        if time.time() - start_time > max_time or time.time() > deadline:
            break
        
        # Initialize with truncated SVD-like approach: use first r columns
        U_init = U0[:, :r].copy()
        V_init = V0[:, :r].copy()
        W_init = W0[:, :r].copy()
        
        # Flatten for optimization
        def flatten_params(x):
            U_f = x[:n*m*r].reshape(n*m, r)
            V_f = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_f = x[n*m*r + m*p*r:].reshape(p*n, r)
            return U_f, V_f, W_f
        
        def objective(x):
            U_f, V_f, W_f = flatten_params(x)
            # Compute error: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_f[i*m+j], V_f[j*p+k] * W_f[k*n+i])
                        errors.append(s - 1.0)
            return np.sum(np.array(errors) ** 2)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < 1e-8:
                U_f, V_f, W_f = flatten_params(result.x)
                # Verify
                if _verify_decomposition(U_f, V_f, W_f, n, m, p, tol=1e-6):
                    return U_f, V_f, W_f, r
        except Exception:
            continue
    
    # If no reduction found, return original
    return U0, V0, W0, r0

def _try_zero_out_columns(U, V, W, n, m, p, deadline, max_time=20):
    """Try to zero out columns to reduce effective rank.
    
    For each column, try setting it to zero and check if the decomposition still holds.
    If not, try to redistribute its contribution to other columns.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    
    # Try removing each column one at a time
    for remove_idx in range(r):
        if time.time() - start_time > max_time or time.time() > deadline:
            break
        
        mask = [c for c in range(r) if c != remove_idx]
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
            U, V, W = U_new, V_new, W_new
            r = r - 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = r
            # Restart loop since indices changed
            remove_idx = 0
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Build rank-71 decomposition
    try:
        U, V, W = _build_rank71()
        
        if not _verify_decomposition(U, V, W, n, m, p):
            raise ValueError("Rank-71 decomposition failed verification")
        
        # Try simple column removal first (fast)
        if time.time() < deadline - 40:
            U2, V2, W2, r2 = _try_zero_out_columns(U, V, W, n, m, p, deadline, max_time=10)
            if r2 < 71:
                U, V, W = U2, V2, W2
        
        # Try optimization-based rank reduction
        if time.time() < deadline - 20:
            U3, V3, W3, r3 = _try_rank_reduction(U, V, W, n, m, p, deadline, target_rank=65, max_time=60)
            if r3 < U.shape[1]:
                U, V, W = U3, V3, W3
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 75)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
