import numpy as np
from datetime import datetime, timedelta
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    U0, V0, W0 = _rank50_baseline(n, m, p)
    best_rank = 50
    best_U, best_V, best_W = U0, V0, W0
    
    start_time = datetime.now()
    timeout = timedelta(seconds=110)
    
    # Try to reduce rank by merging columns
    improved = _merge_columns_search(best_U, best_V, best_W, n, m, p, start_time, timeout)
    if improved is not None:
        U, V, W = improved
        if _verify_decomposition(U, V, W, n, m, p):
            best_rank = len(U[0])
            best_U, best_V, best_W = U, V, W
    
    # Try structured decomposition exploiting block structure
    structured = _try_structured_decomp(n, m, p, start_time, timeout)
    if structured is not None:
        U, V, W = structured
        if _verify_decomposition(U, V, W, n, m, p):
            new_rank = len(U[0])
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _merge_columns_search(U, V, W, n, m, p, start_time, timeout):
    """Try to reduce rank by merging pairs of columns."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = U_np.shape[1]
    improved = False
    
    # Precompute tensor slices for efficiency
    def _check_rank1_merge(r1, r2):
        """Check if sum of rank-1 terms at r1 and r2 is rank 1, and extract factors if so."""
        # Build the 3D tensor: T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
        T_sum = np.einsum('a,b,c->abc', U_np[:, r1], V_np[:, r1], W_np[:, r1]) + \
                np.einsum('a,b,c->abc', U_np[:, r2], V_np[:, r2], W_np[:, r2])
        
        # Check rank by flattening: T as matrix (n*m) x (m*p*p*n)
        T_mat = T_sum.reshape(n*m, m*p*p*n)
        
        # SVD to check rank
        try:
            _, s, _ = np.linalg.svd(T_mat, full_matrices=False)
            if len(s) > 1 and s[1] < 1e-10:
                # Rank 1! Extract factors
                U_r, s_r, Vt_r = np.linalg.svd(T_mat, full_matrices=False)
                if s_r[0] > 1e-10:
                    u = U_r[:, 0]
                    v_w = Vt_r[0, :]
                    # Normalize: u * (v_w / |v_w|^2) gives the decomposition
                    # Actually: T = u * v_w^T, so we need u' * v * w^T = u * v_w^T
                    # where v_w = v * w^T (reshaped)
                    v_w_2d = v_w.reshape(m*p, p*n)
                    U_r2, s_r2, Vt_r2 = np.linalg.svd(v_w_2d, full_matrices=False)
                    if len(s_r2) > 1 and s_r2[1] < 1e-10 and s_r2[0] > 1e-10:
                        v = U_r2[:, 0]
                        w = Vt_r2[0, :]
                        # Scale to match: u * (v * w^T) should equal T_mat
                        # u' = u, v' = v * s_r2[0], w' = w
                        # Or: u' = u / sqrt(s_r2[0]), v' = v * sqrt(s_r2[0]), w' = w
                        # Let's just use: u, v, w as extracted and verify
                        return u, v, w
        except:
            pass
        return None
    
    for iteration in range(100):
        if datetime.now() - start_time > timeout:
            break
        
        merged = False
        for r1 in range(current_rank):
            if datetime.now() - start_time > timeout:
                break
            for r2 in range(r1+1, current_rank):
                result = _check_rank1_merge(r1, r2)
                if result is not None:
                    u, v, w = result
                    # Replace columns r1 and r2 with the new column
                    U_np = np.delete(U_np, [r1, r2], axis=1)
                    V_np = np.delete(V_np, [r1, r2], axis=1)
                    W_np = np.delete(W_np, [r1, r2], axis=1)
                    U_np = np.column_stack([U_np, u])
                    V_np = np.column_stack([V_np, v])
                    W_np = np.column_stack([W_np, w])
                    current_rank -= 1
                    merged = True
                    improved = True
                    break
            
            if merged:
                break
        
        if not merged:
            break
    
    if improved:
        return U_np.tolist(), V_np.tolist(), W_np.tolist()
    
    return None

def _rank50_baseline(n, m, p):
    """Construct the known rank-50 decomposition."""
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1
        V[v_e][col+5] += 1; V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1
        V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0; w_01 = k1 * n + 0
        w_10 = k0 * n + 1; w_11 = k1 * n + 1
        
        W[w_00][col] += 1; W[w_00][col+3] += 1
        W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        
        W[w_11][col] += 1; W[w_11][col+1] -= 1
        W[w_11][col+2] += 1; W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = 0 * p + k8; v_g = 1 * p + k8
    u_a = 0 * m + 0; u_b = 0 * m + 1
    u_c = 1 * m + 0; u_d = 1 * m + 1
    
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0; w_18 = k8 * n + 1
    W[w_08][col] += 1; W[w_08][col+1] += 1
    W[w_18][col+2] += 1; W[w_18][col+3] += 1
    col += 4
    
    u_ar0 = 0 * m + 2; u_ar1 = 1 * m + 2
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
    
    return U, V, W

def _try_structured_decomp(n, m, p, start_time, timeout):
    """Try to find a lower-rank decomposition exploiting the block structure.
    
    The 2x3x9 multiplication can be viewed as:
    - For each output position (i, k), C[i,k] = sum_j A[i,j] * B[j,k]
    - We can try to decompose this using a tensor network structure
    
    Strategy: Try rank-1 decompositions for each "slice" and see if they can be shared.
    """
    if datetime.now() - start_time > timeout:
        return None
    
    # The multiplication tensor T[a,b,c] where:
    # a = i*m + j (input A index, i in [0,n), j in [0,m))
    # b = j'*p + k (input B index, j' in [0,m), k in [0,p))
    # c = k'*n + i (output C index, k' in [0,p), i in [0,n))
    # T[a,b,c] = 1 if j == j' and k == k' and i == i, else 0
    
    # This is a tensor of shape (n*m, m*p, p*n) = (6, 27, 18)
    # Rank of matrix multiplication C_{n,m} x C_{m,p} is known to be at most n*m*p
    # For 2x3x9, trivial rank is 2*3*9 = 54, but we have 50.
    
    # Try to find a decomposition by exploiting the fact that the tensor has
    # a specific structure: T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    # One approach: use the fact that we can factor through the middle dimension
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let's try a decomposition with rank = n*p + m*p + n*m - something
    # Actually, let's try to find a sparse decomposition
    
    # Try rank 45-49 by looking for linear dependencies
    U_np = np.zeros((n*m, 50))
    V_np = np.zeros((m*p, 50))
    W_np = np.zeros((p*n, 50))
    
    # Start from the baseline and try to find null space vectors
    U_base, V_base, W_base = _rank50_baseline(n, m, p)
    U_np = np.array(U_base, dtype=float)
    V_np = np.array(V_base, dtype=float)
    W_np = np.array(W_base, dtype=float)
    
    # Try to find if any column is a linear combination of others
    # This is equivalent to finding the rank of the "coefficient" tensor
    
    # For now, just return None to indicate no improvement found
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    if U_np.shape[0] != n * m:
        return False
    if V_np.shape[0] != m * p:
        return False
    if W_np.shape[0] != p * n:
        return False
    if U_np.shape[1] != rank or V_np.shape[1] != rank or W_np.shape[1] != rank:
        return False
    
    # Use vectorized computation for speed
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Vectorized computation:
        # C[i,k] = sum_j sum_r A[i,j] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_r (sum_j A[i,j] * V[j*p+k, r]) * W[k*n+i, r]
        
        # Reshape for efficient computation
        # For each r, compute: C_r[i,k] = (A @ V_r) * W_r[i,k]
        # where V_r is the r-th column of V reshaped to (m,p), 
        # and W_r is the r-th column of W reshaped to (p,n) transposed
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # V_r: column r of V, shape (m*p,) -> reshape to (m, p)
            V_r = V_np[:, r].reshape(m, p)
            # W_r: column r of W, shape (p*n,) -> reshape to (p, n) then transpose to (n, p)
            W_r = W_np[:, r].reshape(p, n).T
            
            # C_r[i,k] = sum_j A[i,j] * V_r[j,k] * W_r[i,k]
            # = (A @ V_r) * W_r (element-wise)
            C_computed += (A @ V_r) * W_r
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True
