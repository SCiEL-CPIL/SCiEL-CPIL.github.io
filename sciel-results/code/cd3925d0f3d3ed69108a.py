import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor M (sparse representation)
    # M[u, v, w] = 1 if (i,j,k) match
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # We store non-zeros as a list of (u, v, w)
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zeros.append((u, v, w))
                
    # Precompute for each u, the list of (v, w) pairs
    u_to_vw = {u: [] for u in range(size_u)}
    for u, v, w in non_zeros:
        u_to_vw[u].append((v, w))
        
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        R = U.shape[1]
        # Check if sum_r U[u,r] V[v,r] W[w,r] == 1 for all (u,v,w) in non_zeros
        # and 0 otherwise? 
        # The problem statement says "exactly decompose the matrix multiplication tensor".
        # The tensor is 1 at (u,v,w) for valid (i,j,k) and 0 elsewhere.
        # So we must check all (u,v,w) in the support is 1, and ideally others are 0.
        # But usually, rank minimization for multiplication tensor only requires the support to be correct?
        # No, the tensor is defined as 1 on the support and 0 elsewhere.
        # So we must ensure the reconstruction is 0 elsewhere.
        
        # Efficient check:
        # Compute reconstruction for all (u,v,w) in non_zeros
        # And check that the sum of squares of the whole tensor is equal to the number of non-zeros?
        # Or just check the non-zeros are 1 and the rest are 0?
        # Checking the rest is O(size_u * size_v * size_w) = 12*20*15 = 3600. Very fast.
        
        recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        target = np.zeros((size_u, size_v, size_w))
        for u, v, w in non_zeros:
            target[u, v, w] = 1.0
            
        if not np.allclose(recon, target, atol=1e-9):
            return False
        return True

    def solve_als(rank, max_iter=100, tol=1e-12):
        # Initialize
        U = np.random.randn(size_u, rank)
        V = np.random.randn(size_v, rank)
        W = np.random.randn(size_w, rank)
        
        # Precompute VtV and WtW updates? No, they change every step.
        
        for it in range(max_iter):
            # Update U
            # G = (V^T V) o (W^T W)
            VtV = V.T @ V
            WtW = W.T @ W
            G = VtV * WtW
            
            # RHS for U: B_u = M_u_flat @ A
            # B_u[r] = sum_{(v,w) in support(u)} M[u,v,w] * V[v,r] * W[w,r]
            # Since M[u,v,w]=1 for support, B_u[r] = sum_{(v,w) in support(u)} V[v,r] * W[w,r]
            
            U_new = np.zeros((size_u, rank))
            for u in range(size_u):
                if not u_to_vw[u]:
                    continue
                # Accumulate sum of V[v,r] * W[w,r] for all (v,w) in u_to_vw[u]
                # This is a rank-1 update for each (v,w)? No.
                # It's a sum of outer products of V[v,:] and W[w,:]? No, element-wise product.
                # B[r] = sum_{(v,w)} V[v,r] W[w,r]
                # We can do this by iterating over the small list of (v,w)
                b = np.zeros(rank)
                for v, w in u_to_vw[u]:
                    b += V[v] * W[w]
                # Solve U_new[u, :] G = b
                # G is rank x rank
                try:
                    U_new[u] = np.linalg.solve(G, b)
                except np.linalg.LinAlgError:
                    # Use lstsq if singular
                    U_new[u] = np.linalg.lstsq(G, b, rcond=None)[0]
                    
            # Update V
            # G_v = (U^T U) o (W^T W)
            UtU = U_new.T @ U_new
            WtW = W.T @ W
            G_v = UtU * WtW
            
            V_new = np.zeros((size_v, rank))
            # For each v, find all (u,w) such that M[u,v,w]=1
            # M[u,v,w]=1 implies u=i*m+j, v=j*p+k, w=k*n+i
            # So for a fixed v=j*p+k, the pairs (u,w) are:
            # u = i*m+j, w = k*n+i for i=0..n-1
            # So for each v, we have n pairs.
            for v in range(size_v):
                if v % p == 0 and v // p >= m: # Invalid v? No, v is 0..m*p-1
                    pass
                j = v // p
                k = v % p
                b = np.zeros(rank)
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    # Check if this (u,v,w) is in the support? Yes, by construction.
                    b += U_new[u] * W[w]
                try:
                    V_new[v] = np.linalg.solve(G_v, b)
                except np.linalg.LinAlgError:
                    V_new[v] = np.linalg.lstsq(G_v, b, rcond=None)[0]
                    
            # Update W
            # G_w = (U^T U) o (V^T V)
            UtU = U_new.T @ U_new
            VtV = V_new.T @ V_new
            G_w = UtU * VtV
            
            W_new = np.zeros((size_w, rank))
            # For each w, find all (u,v) such that M[u,v,w]=1
            # w = k*n+i
            # u = i*m+j, v = j*p+k
            # So for fixed w=k*n+i, the pairs (u,v) are:
            # for j=0..m-1: u = i*m+j, v = j*p+k
            for w in range(size_w):
                k = w // n
                i = w % n
                b = np.zeros(rank)
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    b += U_new[u] * V_new[v]
                try:
                    W_new[w] = np.linalg.solve(G_w, b)
                except np.linalg.LinAlgError:
                    W_new[w] = np.linalg.lstsq(G_w, b, rcond=None)[0]
                    
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            # Compute residual on the support
            # recon_support = sum_r U[u,r] V[v,r] W[w,r]
            # We want this to be 1.
            # And off-support to be 0.
            
            # Fast residual check on support
            err = 0.0
            for u, v, w in non_zeros:
                val = np.dot(U[u], V[v] * W[w])
                err += (val - 1.0)**2
                
            # Also check off-support? 
            # If we only check support, we might get a solution that is 1 on support but non-zero elsewhere.
            # The rank of the tensor is defined by the whole tensor.
            # However, if the solution is integer/half-integer and exact on support, 
            # is it guaranteed to be 0 elsewhere?
            # Not necessarily. But for small ranks, it's likely.
            # Let's do a full check every 10 iterations.
            
            if it % 10 == 0 or it == max_iter - 1:
                recon = np.einsum('ur,vr,wr->uvw', U, V, W)
                # Check max error
                target = np.zeros((size_u, size_v, size_w))
                for u, v, w in non_zeros:
                    target[u, v, w] = 1.0
                full_err = np.linalg.norm(recon - target)
                if full_err < tol:
                    return U, V, W, full_err
            else:
                if np.sqrt(err) < tol:
                    # Quick check passed, do full check
                    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
                    target = np.zeros((size_u, size_v, size_w))
                    for u, v, w in non_zeros:
                        target[u, v, w] = 1.0
                    full_err = np.linalg.norm(recon - target)
                    if full_err < tol:
                        return U, V, W, full_err
                    else:
                        # Support error was low, but off-support is high.
                        # Continue iterating.
                        pass

            if time.time() > deadline:
                return U, V, W, full_err
                
        return U, V, W, full_err

    # Search for rank
    best_rank = rank_schoolbook
    best_solution = None
    
    # Try ranks from 45 to 59
    for rank in range(45, rank_schoolbook):
        # Try more restarts for lower ranks
        num_restarts = 10 if rank < 55 else 5
        for restart in range(num_restarts):
            U, V, W, res = solve_als(rank, max_iter=50, tol=1e-10)
            if U is None:
                continue
            if res < 1e-8:
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r):
                    best_rank = rank
                    best_solution = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    break
        if best_solution:
            break
        if time.time() > deadline:
            break
            
    if best_solution:
        return best_solution

    # Fallback to schoolbook
    rank = rank_schoolbook
    U = np.zeros((size_u, rank))
    V = np.zeros((size_v, rank))
    W = np.zeros((size_w, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
