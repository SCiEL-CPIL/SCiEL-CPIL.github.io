"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T[i,j,k] for A (n x m) * B (m x p) = C (n x p).
    T[i, j, k] = 1 if i < n, j < m, k < p (this is the standard tensor).
    Actually, the tensor for matrix multiplication is:
    T_{i,k}^{j} = 1, where i is row of A/C, k is col of B/C, j is the "inner" index.
    
    The decomposition is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    So the tensor T_{i,j,k} has T[i,j,k] = 1 for all valid i,j,k.
    
    Wait, the task says the output factor follows the official trace convention W[k*n+i,r].
    Let me re-read: "W[k*n+i,r]" - so W is indexed by (k, i) where k is column of C, i is row of C.
    
    The tensor decomposition: C_{i,k} = sum_r U_{i*m+j?} ... 
    
    Actually, let me think about this more carefully.
    
    We have A (n x m), B (m x p), C (n x p).
    C_{i,k} = sum_{j=0}^{m-1} A_{i,j} * B_{j,k}
    
    The tensor T has entries T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition means we find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    T[i*m+j, k*n+i'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r]
    
    Wait, that's not quite right either. Let me look at the schoolbook decomposition in the seed.
    
    In the seed:
    U[i*m+j, column] = 1
    V[j*p+k, column] = 1
    W[k*n+i, column] = 1
    
    So for each (i,j,k) triple, we create one column r where:
    U[i*m+j, r] = 1
    V[j*p+k, r] = 1
    W[k*n+i, r] = 1
    
    This means the tensor entry T[i*m+j, j*p+k, k*n+i] in the "flattened" sense... 
    
    Actually, the tensor is a 3-tensor. The decomposition expresses the tensor as:
    T_{abc} = sum_r U_{a,r} * V_{b,r} * W_{c,r}
    
    where a = i*m+j (flattened A index), b = j*p+k (flattened B index), c = k*n+i (flattened C index).
    
    But wait, the tensor T should be:
    T[a, b, c] = 1 if the indices are consistent, i.e., if a = i*m+j, b = j'*p+k, c = k'*n+i' with j == j' and i == i' and k == k'.
    
    So T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k.
    
    And the decomposition T_{abc} = sum_r U_{ar} V_{br} W_{cr} means:
    For the schoolbook case, each (i,j,k) gives one rank-1 term with U[i*m+j, r] = V[j*p+k, r] = W[k*n+i, r] = 1.
    
    So we need to find the minimum rank r such that there exist U, V, W with the above property.
    """
    # T is a 3-tensor of size (n*m) x (m*p) x (p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' with j == j' and i == i' and k == k'
    # Actually, T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0.
    # 
    # But actually for the matrix multiplication tensor, T[a, b, c] is defined as:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1
    # 
    # So the tensor is sparse - only n*m*p entries are 1, rest are 0.
    
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _tensor_residual(T, U, V, W):
    """
    Compute the residual ||T - sum_r U[:,r] outer V[:,r] outer W[:,r]||_F
    """
    r = U.shape[1]
    # Reconstruct tensor
    recon = np.zeros(T.shape, dtype=np.float64)
    for rr in range(r):
        recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    residual = T - recon
    return np.linalg.norm(residual, 'fro')

def _verify_exact(T, U, V, W, tol=1e-8):
    """
    Verify that the decomposition is exact (within tolerance).
    """
    r = U.shape[1]
    recon = np.zeros(T.shape, dtype=np.float64)
    for rr in range(r):
        recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    residual = T - recon
    return np.max(np.abs(residual)) < tol

def _als_step(T, U, V, W, rank, time_limit, start_time, max_iter=100):
    """
    Perform ALS (Alternating Least Squares) iterations to refine the decomposition.
    Returns updated (U, V, W).
    """
    a_size = T.shape[0]
    b_size = T.shape[1]
    c_size = T.shape[2]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Fix V and W, solve for U
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
        # This is: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # Vectorized: T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Build the "right-hand side" for U:
        # For each a, we need: T[a, b, c] for all b, c
        # = sum_r U[a,r] * (V[b,r] * W[c,r])
        
        # Reshape: T_flat[a, b*c] = sum_r U[a,r] * (V[b,r]*W[c,r])
        # Let M_U[b, c, r] = V[b,r] * W[c,r]
        # Then T_flat[a, b*c] = sum_r U[a,r] * M_U[b, c, r]
        # = sum_r U[a,r] * (M_U.reshape(b*c, r))[b*c, r]
        
        M_U = np.einsum('br,cr->bcr', V, W).reshape(b_size * c_size, rank)
        T_flat = T.reshape(a_size, b_size * c_size)
        
        # Solve: T_flat = U @ M_U.T
        # U = T_flat @ pinv(M_U.T)
        try:
            U_new, _, _, _ = np.linalg.lstsq(M_U, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass
        
        # Fix U and W, solve for V
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each b: T[:,:,b] = sum_r V[b,r] * (U[:,r] outer W[:,r])
        M_V = np.einsum('ar,cr->acr', U, W).reshape(a_size * c_size, rank)
        T_flat_v = T.reshape(a_size, c_size, b_size).transpose(0, 2, 1).reshape(b_size, a_size * c_size)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_V, T_flat_v.T, rcond=None)
            V = V_new.T
        except:
            pass
        
        # Fix U and V, solve for W
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each c: T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
        M_W = np.einsum('ar,br->abr', U, V).reshape(a_size * b_size, rank)
        T_flat_w = T.reshape(a_size, b_size, c_size).transpose(2, 0, 1).reshape(c_size, a_size * b_size)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_W, T_flat_w.T, rcond=None)
            W = W_new.T
        except:
            pass
    
    return U, V, W

def _try_rank(T, n, m, p, rank, time_limit, seed=0):
    """
    Try to find a rank-`rank` decomposition of T.
    Uses ALS (Alternating Least Squares) with multiple restarts.
    Returns (U, V, W) if found, else None.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    a_size = n * m
    b_size = m * p
    c_size = p * n
    
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(seed + attempt)
        
        # Initialize with random values
        U = np.random.randn(a_size, rank) * 0.1
        V = np.random.randn(b_size, rank) * 0.1
        W = np.random.randn(c_size, rank) * 0.1
        
        # Run ALS
        U, V, W = _als_step(T, U, V, W, rank, time_limit, start_time, max_iter=50)
        
        # Check if exact
        if _verify_exact(T, U, V, W, tol=1e-8):
            return U, V, W
        
        # Try with different initialization strategies
        # Strategy 1: Initialize with schoolbook-like patterns
        if attempt % 3 == 1:
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            # Randomly assign 1s
            for r in range(rank):
                idx_a = np.random.randint(0, a_size)
                idx_b = np.random.randint(0, b_size)
                idx_c = np.random.randint(0, c_size)
                U[idx_a, r] = 1.0
                V[idx_b, r] = 1.0
                W[idx_c, r] = 1.0
        
        # Strategy 2: Initialize with sparse random patterns
        elif attempt % 3 == 2:
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            for r in range(rank):
                for _ in range(2):
                    U[np.random.randint(0, a_size), r] = 1.0
                for _ in range(2):
                    V[np.random.randint(0, b_size), r] = 1.0
                for _ in range(2):
                    W[np.random.randint(0, c_size), r] = 1.0
        
        # Run ALS again
        U, V, W = _als_step(T, U, V, W, rank, time_limit, start_time, max_iter=50)
        
        if _verify_exact(T, U, V, W, tol=1e-8):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*p, m*p, n*m) for matrix multiplication
    # For 2x3 x 3x8: max(16, 24, 6) = 24
    # So we try ranks from 24 to schoolbook_rank - 1
    
    time_start = time.time()
    time_limit_total = 130  # Leave 20s margin for 150s limit
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try ranks from lower bound up
    lower_bound = max(n * p, m * p, n * m)
    
    for rank in range(lower_bound, best_rank):
        elapsed = time.time() - time_start
        if elapsed > time_limit_total:
            break
        
        remaining_time = time_limit_total - elapsed
        per_try_time = min(remaining_time / max(1, best_rank - rank), 20.0)
        
        result = _try_rank(T, n, m, p, rank, per_try_time, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            break  # Found a valid decomposition at this rank
    
    if best_decomp is not None:
        U, V, W = best_decomp
        # Convert to lists with rounded values if close to integers/half-integers
        def _to_list(mat):
            result = []
            for row in mat:
                row_list = []
                for val in row:
                    # Round to nearest integer or half-integer if close
                    nearest_int = round(val)
                    nearest_half = round(2 * val) / 2.0
                    if abs(val - nearest_int) < 1e-6:
                        row_list.append(int(nearest_int))
                    elif abs(val - nearest_half) < 1e-6:
                        row_list.append(nearest_half)
                    else:
                        row_list.append(float(val))
                result.append(row_list)
            return result
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    else:
        # Fall back to schoolbook
        return {"U": [[1.0 if c == r else 0.0 for c in range(schoolbook_rank)] for r in range(n * m)],
                "V": [[1.0 if c == r else 0.0 for c in range(schoolbook_rank)] for r in range(m * p)],
                "W": [[1.0 if c == r else 0.0 for c in range(schoolbook_rank)] for r in range(p * n)]}

# Let me fix the schoolbook fallback properly
def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    schoolbook_rank = n * m * p
    
    time_start = time.time()
    time_limit_total = 130
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Try ranks from lower bound up
    lower_bound = max(n * p, m * p, n * m)
    
    for rank in range(lower_bound, best_rank):
        elapsed = time.time() - time_start
        if elapsed > time_limit_total:
            break
        
        remaining_time = time_limit_total - elapsed
        per_try_time = min(remaining_time / max(1, best_rank - rank), 20.0)
        
        result = _try_rank(T, n, m, p, rank, per_try_time, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            break
    
    def _to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(2 * val) / 2.0
                if abs(val - nearest_int) < 1e-6:
                    row_list.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-6:
                    row_list.append(nearest_half)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    else:
        # Schoolbook fallback
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
