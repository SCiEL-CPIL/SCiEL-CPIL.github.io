import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Build the target tensor: T[(i,j),(j',k),(i',k')] = delta_i_i' * delta_j_j' * delta_k_k'
    # Flatten: index U by i*m+j, V by j'*p+k, W by k'*n+i'
    # T[u, v, w] = 1 if (i,j) from u, (j',k) from v, (i',k') from w satisfy i=i', j=j', k=k'
    
    # For efficiency, note that T[u,v,w] = 1 iff:
    # u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k
    # i.e., u % m == v // p and v % p == w // n and w % n == u // m
    
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1.0
    
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_rank = nm * mp * pn  # 120
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Use a greedy/iterative approach: try rank r from lower bound upward
    # Lower bound: max(n^2, m^2, p^2, n*p, n*m, m*p) but actually the correct lower bound for matmul tensor rank is more subtle
    # For now, try a range of ranks and use optimization
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U @ V @ W reconstructs the target tensor."""
        # Residual: for each (u,v,w), sum_r U[u,r]*V[v,r]*W[w,r] should equal target[u,v,w]
        # Efficient: contract over r
        # R[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        # = sum_r U[u,r] * (V @ W^T)[v,w]... no
        # Actually: R = np.einsum('ur,vr,wr->uvw', U, V, W)
        residual = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.max(np.abs(residual - target))
    
    def try_rank(r, max_iters=2000, lr=0.01, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        np.random.seed(seed)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute residual gradient
            # R[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
            # dR/du[u,v,w] = sum_r V[v,r] W[w,r] (for the r-th term, but we need per-element)
            # Gradient of loss = ||R - target||^2 w.r.t. U[u,r]:
            # = 2 * sum_{v,w} (R[u,v,w] - target[u,v,w]) * V[v,r] * W[w,r]
            
            # Compute R
            R = np.einsum('ur,vr,wr->uvw', U, V, W)
            diff = R - target
            
            # Gradient for U: grad_U[u,r] = sum_{v,w} diff[u,v,w] * V[v,r] * W[w,r]
            # = sum_v (sum_w diff[u,v,w] * W[w,r]) * V[v,r]
            # = sum_v (diff[u,v,:] @ W[:,r]) * V[v,r]
            # Let's compute: tmp1[v,r] = sum_w diff[:,v,w] * W[w,r] = diff @ W (but diff is 3D)
            # diff[u,v,w], W[w,r]: sum_w diff[u,v,w]*W[w,r] = np.einsum('uvw,wr->uvr', diff, W)
            # Then grad_U[u,r] = sum_v einsum_result[u,v,r] * V[v,r] = np.einsum('uvr,vr->ur', tmp, V)
            
            tmp_U = np.einsum('uvw,wr->uvr', diff, W)  # (nm, mp, r)
            grad_U = np.einsum('uvr,vr->ur', tmp_U, V)  # (nm, r)
            
            tmp_V = np.einsum('uvw,ur->vur', diff, U)  # (mp, nm, r)
            grad_V = np.einsum('vur,ur->vr', tmp_V, U)  # (mp, r)
            
            tmp_W = np.einsum('uvw,vr->uwr', diff, V)  # (nm, pn, r)
            grad_W = np.einsum('uwr,ur->wr', tmp_W, U)  # (pn, r)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 100 == 0:
                err = np.max(np.abs(R - target))
                if err < 1e-6:
                    # Verify with tighter tolerance
                    if check_decomposition(U, V, W, tol=1e-5):
                        return U, V, W
        
        final_err = check_decomposition(U, V, W, tol=1e-5)
        if final_err < 1e-5:
            return U, V, W
        return None
    
    # Try ranks from a reasonable lower bound
    # Known lower bounds: rank >= max(n*m, m*p, p*n, n*p, n*m*p / something)
    # For 4x5x6, try from 60 upward (half of schoolbook)
    for r in range(60, 121):
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = try_rank(r, max_iters=1000, lr=0.005, seed=seed + r)
            if result is not None:
                U, V, W = result
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break
        if r < best_rank:
            break
    
    # Convert to lists with integer/half-integer entries if possible
    # Round to nearest half-integer
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    U_out = round_to_half(best_U)
    V_out = round_to_half(best_V)
    W_out = round_to_half(best_W)
    
    # Verify the rounded version
    if not check_decomposition(U_out, V_out, W_out, tol=1e-6):
        # Use unrounded if rounding breaks it
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    # Convert to list of lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
