import time
import numpy as np
import scipy.optimize as opt
import scipy.linalg as sla

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 125  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28) - clean version."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        r = start_r
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    return U, V, W

def residuals(x, n=2, m=4, p=4, rank=None):
    if rank is None:
        rank = int(len(x) / (n*m + m*p + p*n))
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    return T_recon.flatten() - 1.0

def _tensor_to_matrices(T, n, m, p):
    """Convert tensor T[i,j,k] to three matrices for matrix multiplication.
    U_mat: (n*m, p), V_mat: (n*m, p), W_mat: (n*m, p)
    T[i,j,k] = sum_r U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
    """
    # Reshape T to (n*m, p) for each "mode"
    # Mode 1: U_mat[i*m+j, k] = T[i,j,k]
    U_mat = T.reshape(n*m, p)
    # Mode 2: V_mat[j*p+k, i] ... wait, this is getting complicated.
    
    # Let's use the standard unfolding:
    # T_{i,j,k} = sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}
    # This is a 3-way tensor with specific index mappings.
    
    # For the matrix formulation, we need:
    # T = sum_r U_r * V_r * W_r where these are "slices"
    
    # Actually, the standard way to convert a 3-tensor to matrix multiplication:
    # Flatten in a specific way.
    
    # Let's just return the three "matrices" as reshaped views
    U_mat = T.reshape(n*m, p)  # (i,j) -> i*m+j, k -> col
    V_mat = T.transpose(1, 2, 0).reshape(m*p, n)  # (j,k) -> j*p+k, i -> col
    W_mat = T.transpose(2, 0, 1).reshape(p*n, m)  # (k,i) -> k*n+i, j -> col
    
    return U_mat, V_mat, W_mat

def _try_rank_via_matrix_mult(T, target_rank, n, m, p, max_iters=100):
    """
    Use matrix multiplication to find a low-rank decomposition.
    This is a heuristic based on the fact that the tensor can be seen as
    a matrix multiplication problem.
    """
    if time_up():
        return None
    
    # Get the three matrices
    U_mat, V_mat, W_mat = _tensor_to_matrices(T, n, m, p)
    
    # We want to find rank-r factors such that:
    # T[i,j,k] = sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is equivalent to finding a low-rank approximation of a certain matrix.
    
    # One approach: fix two matrices and solve for the third.
    # Another: use alternating least squares.
    
    # Let's try a simple ALS approach:
    # Initialize U, V, W randomly or from SVD of one of the matrices.
    
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    for it in range(max_iters):
        if time_up():
            return None
        
        # Update U: for each (i,j), U[i*m+j, :] = argmin ||T[i,j,:] - V[j*p+?, :] * W[:, i*m+j]||
        # This is tricky because V and W have different index structures.
        
        # Let's use a different approach: convert to a linear system.
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Fix V and W, solve for U:
        # For each (i,j,k): T[i,j,k] = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
        # So U[i*m+j, :] is the solution to a linear system.
        
        # This is getting complex. Let's just use least_squares with a good initialization.
        
        break  # Skip ALS, use least_squares instead
    
    return None

def try_rank_reduction(target_rank, n=2, m=4, p=4, max_restarts=15):
    """Try to reduce rank from Strassen's 28 to target_rank."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    
    U_s = np.array(strassen_decomposition()[0], dtype=float)
    V_s = np.array(strassen_decomposition()[1], dtype=float)
    W_s = np.array(strassen_decomposition()[2], dtype=float)
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        if restart == 0:
            # Direct truncation
            U_init = U_s[:, :target_rank].copy()
            V_init = V_s[:, :target_rank].copy()
            W_init = W_s[:, :target_rank].copy()
        elif restart == 1:
            # Truncation with small perturbation
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * 0.01
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * 0.01
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * 0.01
        else:
            # Larger perturbation
            scale = 0.02 + 0.005 * restart
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * scale
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * scale
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), 
                                       method='lm', max_nfev=3000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-10:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
            
    return None

def try_row_splitting(n=2, m=4, p=4, target_row_rank=12):
    """
    Tensor splitting: decompose into two 1x4x4 row tensors.
    Row i tensor: T[i,j,k] = 1 for all j,k.
    Find low-rank decomposition for each row, then combine.
    """
    if time_up():
        return None
    
    rank = 24
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U_s = np.array(strassen_decomposition()[0], dtype=float)
    V_s = np.array(strassen_decomposition()[1], dtype=float)
    W_s = np.array(strassen_decomposition()[2], dtype=float)
    
    best_result = None
    best_error = float('inf')
    
    # Row-aware initialization: try to make rows use disjoint or overlapping
    # sets of rank dimensions
    for restart in range(10):
        if time_up():
            break
        
        if restart == 0:
            # Rows use disjoint halves of the rank
            U_init = np.zeros((n*m, rank))
            V_init = np.zeros((m*p, rank))
            W_init = np.zeros((p*n, rank))
            # Row 0 uses columns 0-11, Row 1 uses columns 12-23
            U_init[0:4, 0:12] = U_s[0:4, 0:12]
            U_init[4:8, 12:24] = U_s[4:8, 0:12]
            V_init[:, 0:12] = V_s[:, 0:12]
            V_init[:, 12:24] = V_s[:, 0:12]
            W_init[0:4, 0:12] = W_s[0:4, 0:12]
            W_init[4:8, 12:24] = W_s[4:8, 0:12]
        else:
            # Overlapping initialization with perturbation
            scale = 0.01 + 0.005 * restart
            U_init = U_s[:, :rank].copy() + np.random.randn(n*m, rank) * scale
            V_init = V_s[:, :rank].copy() + np.random.randn(m*p, rank) * scale
            W_init = W_s[:, :rank].copy() + np.random.randn(p*n, rank) * scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, rank),
                                       method='lm', max_nfev=2000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-10:
                U = best_result[:size_U].reshape(n*m, rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, rank)
                W = best_result[size_U + size_V : ].reshape(p*n, rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
            
    return None

def try_lateral_merge(n=2, m=4, p=4, target_rank=24):
    """
    Lateral thinking: merge the four Strassen blocks into a non-block-diagonal
    rank-24 decomposition by exploiting the asymmetry of 2x4x4 dimensions.
    """
    if time_up():
        return None
    
    U_s = np.array(strassen_decomposition()[0], dtype=float)
    V_s = np.array(strassen_decomposition()[1], dtype=float)
    W_s = np.array(strassen_decomposition()[2], dtype=float)
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    
    best_result = None
    best_error = float('inf')
    
    # Multiple initialization strategies for the lateral merge
    strategies = []
    
    # Strategy 1: Block-coupled initialization
    U_init1 = np.zeros((n*m, target_rank))
    V_init1 = np.zeros((m*p, target_rank))
    W_init1 = np.zeros((p*n, target_rank))
    
    for r in range(28):
        target_r = int(r * target_rank / 28)
        if target_r < target_rank:
            U_init1[:, target_r] += U_s[:, r]
            V_init1[:, target_r] += V_s[:, r]
            W_init1[:, target_r] += W_s[:, r]
    strategies.append((U_init1, V_init1, W_init1, "linear_map"))
    
    # Strategy 2: Row-coupled initialization
    U_init2 = np.zeros((n*m, target_rank))
    V_init2 = np.zeros((m*p, target_rank))
    W_init2 = np.zeros((p*n, target_rank))
    
    for r in range(28):
        target_r = r % target_rank
        U_init2[:, target_r] += U_s[:, r]
        V_init2[:, target_r] += V_s[:, r]
        W_init2[:, target_r] += W_s[:, r]
    strategies.append((U_init2, V_init2, W_init2, "row_coupled"))
    
    # Strategy 3: Random projection
    U_init3 = np.zeros((n*m, target_rank))
    V_init3 = np.zeros((m*p, target_rank))
    W_init3 = np.zeros((p*n, target_rank))
    
    for r in range(target_rank):
        src_cols = np.random.choice(28, size=3, replace=False)
        weights = np.random.randn(3)
        weights /= np.linalg.norm(weights)
        U_init3[:, r] = np.sum([weights[i] * U_s[:, src_cols[i]] for i in range(3)], axis=0)
        V_init3[:, r] = np.sum([weights[i] * V_s[:, src_cols[i]] for i in range(3)], axis=0)
        W_init3[:, r] = np.sum([weights[i] * W_s[:, src_cols[i]] for i in range(3)], axis=0)
    strategies.append((U_init3, V_init3, W_init3, "random_proj"))
    
    # Strategy 4: Block-merge with shared terms
    U_init4 = np.zeros((n*m, target_rank))
    V_init4 = np.zeros((m*p, target_rank))
    W_init4 = np.zeros((p*n, target_rank))
    
    for r in range(7):
        U_init4[:, r] = U_s[:, r] + U_s[:, 21 + r]
        V_init4[:, r] = V_s[:, r] + V_s[:, 21 + r]
        W_init4[:, r] = W_s[:, r] + W_s[:, 21 + r]
    
    for r in range(7):
        U_init4[:, 7 + r] = U_s[:, 7 + r] + U_s[:, 14 + r]
        V_init4[:, 7 + r] = V_s[:, 7 + r] + V_s[:, 14 + r]
        W_init4[:, 7 + r] = W_s[:, 7 + r] + W_s[:, 14 + r]
    
    for r in range(14, target_rank):
        src_r = np.random.randint(0, 28)
        U_init4[:, r] = U_s[:, src_r]
        V_init4[:, r] = V_s[:, src_r]
        W_init4[:, r] = W_s[:, src_r]
    strategies.append((U_init4, V_init4, W_init4, "block_merge"))
    
    # Strategy 5: Pure perturbation from truncated Strassen
    U_init5 = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * 0.05
    V_init5 = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * 0.05
    W_init5 = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * 0.05
    strategies.append((U_init5, V_init5, W_init5, "perturb"))
    
    # Try each strategy
    for idx, (U_init, V_init, W_init, name) in enumerate(strategies):
        if time_up():
            break
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank),
                                       method='lm', max_nfev=5000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-10:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
        
        # If not converged, try a second pass with the result as init
        if best_result is not None and err > 1e-10 and not time_up():
            U_init2 = best_result[:size_U].reshape(n*m, target_rank)
            V_init2 = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
            W_init2 = best_result[size_U + size_V : ].reshape(p*n, target_rank)
            
            x0_2 = np.concatenate([U_init2.flatten(), V_init2.flatten(), W_init2.flatten()])
            
            try:
                result2 = opt.least_squares(residuals, x0_2, args=(n, m, p, target_rank),
                                            method='lm', max_nfev=3000)
                err2 = np.linalg.norm(result2.fun)
                
                if err2 < best_error:
                    best_error = err2
                    best_result = result2.x
                    
                if err2 < 1e-10:
                    U = best_result[:size_U].reshape(n*m, target_rank)
                    V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                    W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                    
                    valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                    if valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception:
                pass
    
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try lateral merge for rank 24 (most promising)
    if best_rank > 24 and not time_up():
        res = try_lateral_merge(n, m, p, 24)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = 24
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try rank 24 with row-splitting approach
    if best_rank > 24 and not time_up():
        res = try_row_splitting(n, m, p)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = 24
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try rank 24 with standard reduction
    if best_rank > 24 and not time_up():
        res = try_rank_reduction(24, n, m, p)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = 24
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try intermediate ranks if 24 failed
    if best_rank > 24 and not time_up():
        for r in [25, 26, 27]:
            if time_up():
                break
            res = try_rank_reduction(r, n, m, p)
            if res is not None:
                U2, V2, W2 = res
                valid, err = verify_decomposition(U2, V2, W2, n, m, p)
                if valid and r < best_rank:
                    best_rank = r
                    best_result = {"U": U2, "V": V2, "W": W2}
                    break
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
