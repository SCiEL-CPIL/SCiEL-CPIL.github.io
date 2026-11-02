import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _build_correct_tensor():
    """
    Build the matrix multiplication tensor for 3x3 * 3x5.
    T[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k].
    In standard multiplication, A[i, j] * B[j, k] contributes to C[i, k].
    So T[i, j, k] = 1 for all valid i, j, k?
    Wait, the tensor indices in the decomposition are:
    U: (i, j)
    V: (j, k)
    W: (k, i)
    The term U[i, j, r] * V[j, k, r] * W[k, i, r] corresponds to the contribution of A[i, j] and B[j, k] to C[i, k].
    In standard matrix multiplication, this contribution is ALWAYS 1 (coefficient is 1).
    So T[i, j, k] = 1 for all i, j, k?
    
    If T is all ones, rank is 1.
    Why did the seed code fail to find rank 1?
    Maybe the indices in W are different?
    "W[k*n+i, r]"
    If W index is k*n+i, then W[k, i, r].
    The sum is over r.
    If T is all ones, U=1, V=1, W=1 works.
    
    Is it possible the task implies a DIFFERENT tensor?
    "Matrix multiplication 3,3,5"
    Usually, this means the rank of the multiplication map.
    The rank of the all-ones tensor is 1.
    The rank of the matrix multiplication tensor (which is sparse) is ~39.
    
    Let's re-read carefully: "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 5 matrix."
    This phrasing strongly implies the **matrix multiplication tensor**, which is sparse.
    The seed code's `_build_tensor` returning all ones is likely a **bug** or **placeholder** in the seed.
    The fact that it searches for ranks 20-44 confirms that the intended tensor is NOT all ones (which would have rank 1).
    The intended tensor is the sparse one where $T_{i, j, k} = 1$ if the indices form a valid chain $i \to j \to k \to i$?
    Actually, for the sparse tensor, $T_{i, j, k}$ is 1 ONLY if the term $A_{ij} B_{jk}$ is part of the product $C_{ik}$.
    In the flattened index space:
    $U$ row $i \cdot m + j$.
    $V$ row $j \cdot p + k$.
    $W$ row $k \cdot n + i$.
    The term $U_{ij} V_{jk} W_{ki}$ exists.
    For this to be the matrix multiplication tensor, we need:
    $T_{ij, jk, ki} = 1$.
    And $T_{ij, j'k', k'i'} = 0$ if the indices don't match the chain?
    Specifically, if $j \neq j'$, the term $A_{ij} B_{j'k'}$ is not a standard term.
    So $T$ is non-zero only when $j=j'$ and $k=k'$ and $i=i''$?
    Wait, the indices of the tensor $T$ in the decomposition sum are $(i, j), (j, k), (k, i)$.
    The tensor $T$ has entries $T_{(i,j), (j',k'), (k'',i'')}$.
    For the decomposition to represent matrix multiplication, we need:
    $\sum_r U_{(i,j), r} V_{(j',k'), r} W_{(k'',i'), r} = T_{(i,j), (j',k'), (k'',i')}$.
    The standard multiplication tensor has $T_{(i,j), (j',k'), (k'',i')} = 1$ if $j=j'$ and $k=k'$ and $i''=i$?
    No. The term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    So the "output" is $(i, k)$.
    In the decomposition, the output index is $(k, i)$ (from W).
    So $T_{(i,j), (j,k), (k,i)} = 1$.
    All other entries are 0.
    
    So the tensor is sparse with $n \cdot m \cdot p$ non-zero entries.
    """
    n, m, p = DIMENSIONS
    # Flattened dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _compute_error(U, V, W, T):
    """
    Compute error using einsum for speed.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    T: (n*m, m*p, p*n)
    """
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    # Reshape to 3D
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    # Contract over R
    # Result[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W is indexed by k, i. In W_3d, index is (k, i, r).
    # So we need W_3d[k, i, r].
    # einsum: 'ijr,jkr,kir->ijk'
    # U_3d: i, j, r
    # V_3d: j, k, r
    # W_3d: k, i, r -> need to permute or use correct indices.
    # W_3d is (p, n, R) -> (k, i, r).
    # So 'kir' is correct.
    
    try:
        reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        # Fallback to loops if einsum fails (shouldn't happen)
        reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    reconstructed[i, j, k] = val
    
    # T is (n*m, m*p, p*n). Reshape to (n, m, p) for comparison?
    # T[u_idx, v_idx, w_idx] where u_idx=i*m+j, v_idx=j*p+k, w_idx=k*n+i.
    # This mapping is exactly (i, j, k) -> (u, v, w).
    # So we can reshape T to (n, m, p) if we permute correctly?
    # T is stored in flattened indices.
    # T_3d[i, j, k] should be T[i*m+j, j*p+k, k*n+i].
    # We can just compare the flattened arrays?
    # No, the indices are permuted.
    # Let's construct T_3d correctly.
    
    T_3d = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_3d[i, j, k] = T[i*m+j, j*p+k, k*n+i]
                
    error = np.sum((reconstructed - T_3d) ** 2)
    return error

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _try_rank(target_rank, T, time_budget, seed_offset=0):
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start_time = time.time()
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        np.random.seed(42 + seed_offset + restart)
        
        # Initialize
        U = np.random.randn(dim_u, target_rank) * 0.5
        V = np.random.randn(dim_v, target_rank) * 0.5
        W = np.random.randn(dim_w, target_rank) * 0.5
        
        current_error = _compute_error(U, V, W, T)
        
        # Gradient Descent / Local Search
        # We will use a simple coordinate descent with random steps
        for iteration in range(2000):
            if time.time() - start_time > time_budget:
                break
            
            # Pick a random factor and column
            factor = np.random.randint(3) # 0:U, 1:V, 2:W
            c = np.random.randint(target_rank)
            
            # Save
            if factor == 0:
                orig = U[:, c].copy()
            elif factor == 1:
                orig = V[:, c].copy()
            else:
                orig = W[:, c].copy()
            
            best_local_error = current_error
            best_local_val = orig
            
            # Try 5 random perturbations
            for trial in range(5):
                scale = 0.5
                step = np.random.randn(len(orig)) * scale
                
                if factor == 0:
                    U[:, c] = orig + step
                    err = _compute_error(U, V, W, T)
                elif factor == 1:
                    V[:, c] = orig + step
                    err = _compute_error(V, U, W, T) # Note: U, V, W order in function
                else:
                    W[:, c] = orig + step
                    err = _compute_error(U, V, W, T)
                
                if err < best_local_error:
                    best_local_error = err
                    best_local_val = orig + step
            
            # Apply best
            if factor == 0:
                U[:, c] = best_local_val
            elif factor == 1:
                V[:, c] = best_local_val
            else:
                W[:, c] = best_local_val
            
            current_error = best_local_error
            
            if current_error < best_error:
                best_error = current_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if current_error < 1e-8:
                break
                
    if best_error < 1e-8 and best_U is not None:
        # Round
        U_r = np.array([[_round_to_half_integer(x) for x in row] for row in best_U])
        V_r = np.array([[_round_to_half_integer(x) for x in row] for row in best_V])
        W_r = np.array([[_round_to_half_integer(x) for x in row] for row in best_W])
        
        err_r = _compute_error(U_r, V_r, W_r, T)
        if err_r < 1e-9:
            return U_r, V_r, W_r, err_r
            
    return None, None, None, best_error

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    T = _build_correct_tensor()
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try lower ranks
    # Likely rank is around 30-40
    for target_rank in range(30, rank_schoolbook):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Allocate time
        time_per_rank = min(5.0, remaining / (rank_schoolbook - target_rank))
        
        U, V, W, err = _try_rank(target_rank, T, time_per_rank, seed_offset=target_rank)
        
        if U is not None:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Return schoolbook
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
