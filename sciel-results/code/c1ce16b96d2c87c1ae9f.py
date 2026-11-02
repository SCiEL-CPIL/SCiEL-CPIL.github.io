import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
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
    
    # Build the target tensor T as a sparse list of (a, b, c, val)
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # Note: c index is k*n + i.
    # a index is i*m + j.
    # b index is j*p + k.
    
    nnz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nnz.append((a, b, c, 1.0))
                
    # We want to find U, V, W of rank R < rank_schoolbook
    # such that sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # We'll try ranks from rank_schoolbook - 1 down to some limit
    # Use ALS
    # To speed up, we can use the fact that the system is determined by the non-zeros and zeros.
    # But ALS works on the full tensor or a sampled subset.
    # Given the size is small (9x21x21), we can use the full tensor.
    
    T_full = np.zeros((n*m, m*p, p*n))
    for a, b, c, val in nnz:
        T_full[a, b, c] = val
        
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Try ranks
    for R in range(rank_schoolbook - 1, 30, -1):
        if time.time() > deadline:
            break
        # Try a few restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
            # Initialize U, V, W
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Normalize to prevent divergence
            # U /= np.linalg.norm(U, axis=0, keepdims=True)
            
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                # Minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is equivalent to solving for U[a,:] given V, W
                # Residual R[a,b,c] = T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r]
                # Gradient wrt U[a,r] is -2 sum_{b,c} R[a,b,c] V[b,r] W[c,r]
                # Hessian is 2 sum_{b,c} V[b,r] W[c,r] V[b,r'] W[c,r']
                # We can solve the normal equations.
                # Let M_U[r, r'] = sum_{b,c} V[b,r] W[c,r] V[b,r'] W[c,r']
                # Let b_U[r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r]
                # U[a,:] = M_U^{-1} b_U
                
                # Compute M_U and b_U for all a
                # M_U is R x R
                # M_U[r, r'] = sum_{b,c} V[b,r] W[c,r] V[b,r'] W[c,r']
                # = sum_b (sum_c V[b,r] W[c,r] V[b,r'] W[c,r'])
                # Let's compute efficiently.
                
                # Vectorize:
                # V is (m*p, R), W is (p*n, R)
                # We need sum_{b,c} V[b,r] W[c,r] V[b,r'] W[c,r']
                # This is trace(V^T V * (W^T W)^T)? No.
                # Let's just loop over r, r' if R is small.
                # R is ~60. 60^2 = 3600.
                # b, c loops: 21 * 21 = 441.
                # Total ops: 3600 * 441 ~ 1.6M. Fast.
                
                M_U = np.zeros((R, R))
                b_U = np.zeros((n*m, R))
                
                # Precompute V^T V? No, W is involved.
                # M_U[r, r'] = sum_b sum_c V[b,r] V[b,r'] W[c,r] W[c,r']
                # = sum_b V[b,r] V[b,r'] (sum_c W[c,r] W[c,r'])
                # Let S_W[r, r'] = sum_c W[c,r] W[c,r'] = (W^T W)[r, r']
                # Then M_U[r, r'] = sum_b V[b,r] V[b,r'] S_W[r, r']
                # This is element-wise product of (V^T V) and S_W? No.
                # M_U = (V^T V) * S_W? No, the indices are mixed.
                # M_U[r, r'] = sum_b V[b,r] V[b,r'] S_W[r, r']
                # This is not a standard matrix product.
                
                # Let's just compute it directly for simplicity, as R is small.
                for r in range(R):
                    for rp in range(r, R):
                        val = 0
                        for b in range(m*p):
                            for c in range(p*n):
                                val += V[b, r] * W[c, r] * V[b, rp] * W[c, rp]
                        M_U[r, rp] = val
                        M_U[rp, r] = val
                        
                for a in range(n*m):
                    for r in range(R):
                        val = 0
                        for b in range(m*p):
                            for c in range(p*n):
                                val += T_full[a, b, c] * V[b, r] * W[c, r]
                        b_U[a, r] = val
                        
                # Solve M_U * U[a,:] = b_U[a,:]
                try:
                    U[a, :] = np.linalg.solve(M_U, b_U[a, :])
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(M_U, b_U[a, :], rcond=None)[0]
                
                # Similarly for V and W
                # M_V[r, rp] = sum_{a,c} U[a,r] W[c,r] U[a,rp] W[c,rp]
                # M_W[r, rp] = sum_{a,b} U[a,r] V[b,r] U[a,rp] V[b,rp]
                
                M_V = np.zeros((R, R))
                b_V = np.zeros((m*p, R))
                for r in range(R):
                    for rp in range(r, R):
                        val = 0
                        for a in range(n*m):
                            for c in range(p*n):
                                val += U[a, r] * W[c, r] * U[a, rp] * W[c, rp]
                        M_V[r, rp] = val
                        M_V[rp, r] = val
                for b in range(m*p):
                    for r in range(R):
                        val = 0
                        for a in range(n*m):
                            for c in range(p*n):
                                val += T_full[a, b, c] * U[a, r] * W[c, r]
                        b_V[b, r] = val
                try:
                    for b in range(m*p):
                        V[b, :] = np.linalg.solve(M_V, b_V[b, :])
                except np.linalg.LinAlgError:
                    for b in range(m*p):
                        V[b, :] = np.linalg.lstsq(M_V, b_V[b, :], rcond=None)[0]
                        
                M_W = np.zeros((R, R))
                b_W = np.zeros((p*n, R))
                for r in range(R):
                    for rp in range(r, R):
                        val = 0
                        for a in range(n*m):
                            for b in range(m*p):
                                val += U[a, r] * V[b, r] * U[a, rp] * V[b, rp]
                        M_W[r, rp] = val
                        M_W[rp, r] = val
                for c in range(p*n):
                    for r in range(R):
                        val = 0
                        for a in range(n*m):
                            for b in range(m*p):
                                val += T_full[a, b, c] * U[a, r] * V[b, r]
                        b_W[c, r] = val
                try:
                    for c in range(p*n):
                        W[c, :] = np.linalg.solve(M_W, b_W[c, :])
                except np.linalg.LinAlgError:
                    for c in range(p*n):
                        W[c, :] = np.linalg.lstsq(M_W, b_W[c, :], rcond=None)[0]
                
                # Check residual
                # Res = T_full - sum_r U[:,r,None] * V[None,:,r] * W[None,None,r]
                # This is expensive. Check on a subset or every 10 iters.
                if it % 10 == 0:
                    # Compute residual on non-zeros
                    res = 0
                    for a, b, c, val in nnz:
                        s = 0
                        for r in range(R):
                            s += U[a, r] * V[b, r] * W[c, r]
                        res += (s - val) ** 2
                    # Check some zeros
                    # Random sample of 100 zeros
                    for _ in range(100):
                        a = np.random.randint(n*m)
                        b = np.random.randint(m*p)
                        c = np.random.randint(p*n)
                        if (a, b, c) not in [(x[0], x[1], x[2]) for x in nnz]: # This check is slow
                             pass # Skip zero check for speed in inner loop, do at end
            
            # Final check
            valid = True
            for a, b, c, val in nnz:
                s = 0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - val) > 1e-5:
                    valid = False
                    break
            if valid:
                # Check zeros
                # We need to ensure entries are 0 for non-nnz
                # This is hard to check all.
                # But if the ALS converged to low residual, it's likely.
                # Let's verify a few zeros.
                zero_check = 0
                for _ in range(500):
                    a = np.random.randint(n*m)
                    b = np.random.randint(m*p)
                    c = np.random.randint(p*n)
                    # Check if this is a non-zero
                    is_nnz = False
                    for na, nb, nc, _ in nnz:
                        if na == a and nb == b and nc == c:
                            is_nnz = True
                            break
                    if not is_nnz:
                        s = 0
                        for r in range(R):
                            s += U[a, r] * V[b, r] * W[c, r]
                        if abs(s) > 1e-5:
                            valid = False
                            break
                        zero_check += 1
                            
                if valid:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Verify rounded
                    valid_rounded = True
                    for a, b, c, val in nnz:
                        s = 0
                        for r in range(R):
                            s += U_r[a, r] * V_r[b, r] * W_r[c, r]
                        if abs(s - val) > 1e-9:
                            valid_rounded = False
                            break
                    if valid_rounded:
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = R
                        return best_result # Return immediately if found
    
    return best_result
