import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _compute_decomposition_error(U, V, W, T):
    """
    Compute the error between the tensor product sum and T.
    """
    try:
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return float('inf')
    error = np.linalg.norm(reconstructed - T)
    return error

def _try_rank_structured(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R using structured approach.
    """
    start_time = time.time()
    
    # The tensor T[u,v,w] = 1 iff u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    # For rank R, we need R terms.
    # Each term r contributes U[u,r]*V[v,r]*W[w,r] to T[u,v,w]
    
    # Key insight: For each (u,v), T[u,v,:] is a vector in R^{p*n}
    # It must be in the span of {W[:,r] * U[u,r]*V[v,r] : r=1..R}
    # So T[u,v,:] = sum_r c_r(u,v) * W[:,r] where c_r(u,v) = U[u,r]*V[v,r]
    
    # This means the matrix C(u,v) = [c_1, c_2, ..., c_R] satisfies:
    # T[u,v,:] = C(u,v) @ W
    # where W is (p*n, R)
    
    # So W must be such that all T[u,v,:] are in its column space.
    
    # Strategy: 
    # 1. Pick R linearly independent vectors from the set of all T[u,v,:]
    # 2. Use them to determine W (up to scaling)
    # 3. Then solve for U and V
    
    # Actually, let's try a different approach: 
    # Use the fact that the tensor is separable in a structured way.
    
    # For each rank term r, it "covers" some subset of (i,j,k) triples.
    # The contribution is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try to find a decomposition by solving a system of equations.
    # For each (u,v,w) where T[u,v,w] != 0, we have:
    # sum_r U[u,r] * V[v,r] * W[w,r] = 1
    # For all other (u,v,w): sum_r U[u,r] * V[v,r] * W[w,r] = 0
    
    # This is a system of nonlinear equations. But we can linearize by fixing two of U,V,W.
    
    # Approach: Fix W, then the equations become linear in U and V.
    # Specifically, for each w, let W_w be the vector [W[w,1], ..., W[w,R]]
    # Then T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # This is like: T[:, :, w] = sum_r (U[:,r] * V[:,r]) * W[w,r]
    # So T[:, :, w] = sum_r W[w,r] * (U[:,r] * V[:,r])
    
    # Let M_w = T[:, :, w] (shape (n*m, m*p))
    # Let P_r = U[:,r] * V[:,r] (element-wise product, shape (n*m, m*p))
    # Then M_w = sum_r W[w,r] * P_r
    
    # So {P_1, ..., P_R} must span all M_w vectors.
    
    # This is getting complex. Let me try a more direct approach:
    # Use random initialization with the structure in mind, then use local search.
    
    # Better: Use the schoolbook decomposition as a starting point and try to merge terms.
    
    # Schoolbook: R = n*m*p = 24 terms, one for each (i,j,k)
    # Can we merge some terms?
    
    # For two terms (i1,j1,k1) and (i2,j2,k2) to be merged into one term,
    # we need: U[u,r]*V[v,r]*W[w,r] = delta_{u,u1}*delta_{v,v1}*delta_{w,w1} + delta_{u,u2}*delta_{v,v2}*delta_{w,w2}
    # This is generally not possible with a single rank-1 tensor unless the supports are nested.
    
    # Let's try a different strategy: use SVD on the tensor to get a low-rank approximation,
    # then round to integer/half-integer values.
    
    # Reshape T to a matrix: (n*m, m*p*p*n)
    T_mat = T.reshape(n * m, m * p * p * n)
    
    # SVD
    try:
        U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
    except Exception:
        return None
    
    # The rank of T is the number of nonzero singular values
    # For our tensor, it should be n*m*p = 24 (full rank)
    
    # Truncate to rank R
    if R >= min(T_mat.shape):
        U_trunc = U_svd[:, :R]
        s_trunc = s[:R]
        Vt_trunc = Vt[:R, :]
    else:
        U_trunc = U_svd[:, :R]
        s_trunc = s[:R]
        Vt_trunc = Vt[:R, :]
    
    # Reconstruct: T_approx = U_trunc @ diag(s_trunc) @ Vt_trunc
    # = (U_trunc * s_trunc) @ Vt_trunc
    
    # Now we need to factor this as sum_r U[:,r] * V[:,r] * W[:,r]
    # This is not straightforward. The SVD gives a matrix factorization, not a tensor one.
    
    # Let me try yet another approach: use the fact that we know the exact values.
    # The tensor has only 0s and 1s. We need exact decomposition.
    
    # Use a SAT-like approach: try small values and solve.
    
    # For each (i,j,k), we need the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for all other (u,v,w), the sum = 0.
    
    # Let's fix U and V, then W is determined linearly.
    # Actually, if we fix U and V, then for each w, we have:
    # T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # This is a linear system in W[w,:].
    
    # But U and V are also unknown.
    
    # Let's try: fix the support pattern. Each rank term r has a "support" 
    # which is the set of (u,v,w) where U[u,r]*V[v,r]*W[w,r] != 0.
    
    # For the schoolbook, each term has support on exactly one (u,v,w).
    # For a lower rank, some terms must have larger support.
    
    # This is very hard to search combinatorially.
    
    # Let me try a pragmatic approach: use the existing local search but with better initialization.
    # Initialize with the schoolbook decomposition and try to zero out some entries.
    
    # Schoolbook initialization
    U_init = np.zeros((n * m, R))
    V_init = np.zeros((m * p, R))
    W_init = np.zeros((p * n, R))
    
    if R >= n * m * p:
        # Use schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_init[i * m + j, col] = 1
                    V_init[j * p + k, col] = 1
                    W_init[k * n + i, col] = 1
                    col += 1
        # If R > n*m*p, we have extra zero columns
        # If R == n*m*p, we're done
        if R == n * m * p:
            return U_init, V_init, W_init
    else:
        # R < n*m*p: need to find a compressed decomposition
        # This is the hard case
        # Use random initialization with values from the set
        U_init = np.random.choice(values, size=(n * m, R))
        V_init = np.random.choice(values, size=(m * p, R))
        W_init = np.random.choice(values, size=(p * n, R))
    
    # Now do local search
    def _local_search_improved(U, V, W, T, values, max_iter=5000, timeout=10):
        start_time = time.time()
        current_error = _compute_decomposition_error(U, V, W, T)
        
        n_u, n_v, n_w = U.size, V.size, W.size
        
        for _ in range(max_iter):
            if time.time() - start_time > timeout:
                break
            if current_error == 0.0:
                break
                
            total_vars = n_u + n_v + n_w
            var_idx = random.randint(0, total_vars - 1)
            
            if var_idx < n_u:
                mat = U
                idx = var_idx
            elif var_idx < n_u + n_v:
                mat = V
                idx = var_idx - n_u
            else:
                mat = W
                idx = var_idx - n_u - n_v
                
            old_val = mat.flat[idx]
            new_val = random.choice(values)
            
            mat.flat[idx] = new_val
            new_error = _compute_decomposition_error(U, V, W, T)
            
            if new_error < current_error:
                current_error = new_error
            else:
                mat.flat[idx] = old_val
                
        return current_error
    
    best_error = float('inf')
    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    
    num_restarts = 10
    time_per_restart = time_limit / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if R >= n * m * p:
            # For R >= schoolbook, use schoolbook initialization
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            # Try to zero out some entries to reduce effective rank
            # But this doesn't help if we need exactly R columns
            # Actually, we need the matrices to have shape (., R), so we can't reduce columns
            # We can set some columns to zero, but that's equivalent to having fewer terms
            pass
        else:
            # Random initialization
            U = np.random.choice(values, size=(n * m, R))
            V = np.random.choice(values, size=(m * p, R))
            W = np.random.choice(values, size=(p * n, R))
        
        err = _local_search_improved(U, V, W, T, values, max_iter=3000, timeout=time_per_restart)
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error == 0.0:
            break
    
    if best_error == 0.0:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Define search space
    values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try ranks from low to high
    # Known that rank of 2x3x4 matmul is at most 24, likely lower
    # Try from 15 to 23
    for R in range(15, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / max(ranks_left, 1)
        time_budget = min(time_budget, 15.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank_structured(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            err = _compute_decomposition_error(U, V, W, T)
            if err < 1e-10:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
                
    return best_result
