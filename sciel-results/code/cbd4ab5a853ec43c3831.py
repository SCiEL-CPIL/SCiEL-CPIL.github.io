import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16

    # Precompute indices for the non-zero entries of T
    # T[i*m+j, j*p+k, k*n+i] = 1
    idx_i = []
    idx_j = []
    idx_k = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_i.append(i*m+j)
                idx_j.append(j*p+k)
                idx_k.append(k*n+i)
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    num_nonzero = len(idx_i)

    def verify(U, V, W, R):
        # Check if U, V, W are half-integers
        # Check all tensor entries
        for ii in range(nm):
            for jj in range(mp):
                for kk in range(pn):
                    val = 0.0
                    for r in range(R):
                        val += U[ii, r] * V[jj, r] * W[kk, r]
                    
                    # Determine expected value
                    i = ii // m
                    j = ii % m
                    j2 = jj // p
                    k = jj % p
                    k2 = kk // n
                    i2 = kk % n
                    
                    expected = 0.0
                    if i == i2 and j == j2 and k == k2:
                        expected = 1.0
                    
                    if abs(val - expected) > 1e-9:
                        return False
        return True

    def objective(params, R):
        # params is a flat array of length R*(nm+mp+pn)
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        # T_recon[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # We need T_recon at specific indices (ii, jj, kk)
        # T_recon[ii, jj, kk] = sum_r U[ii, r] * V[jj, r] * W[kk, r]
        
        # Extract the non-zero entries of T_recon
        # U_idx = U[idx_i, :] -> (num_nonzero, R)
        # V_idx = V[idx_j, :] -> (num_nonzero, R)
        # W_idx = W[idx_k, :] -> (num_nonzero, R)
        # vals = sum_r U_idx[:,r] * V_idx[:,r] * W_idx[:,r]
        
        U_idx = U[idx_i, :]
        V_idx = V[idx_j, :]
        W_idx = W[idx_k, :]
        
        vals = np.sum(U_idx * V_idx * W_idx, axis=1)
        
        # Sum of squares of all entries of T_recon
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # This is expensive. 
        # Alternative: 
        # ||T_recon||^2_F = sum_{i,j,k} (sum_r U[i,r]V[j,r]W[k,r])^2
        # = sum_r sum_{i,j,k} U[i,r]^2 V[j,r]^2 W[k,r]^2
        # = sum_r (sum_i U[i,r]^2) (sum_j V[j,r]^2) (sum_k W[k,r]^2)
        
        sum_U_sq = np.sum(U**2, axis=0)
        sum_V_sq = np.sum(V**2, axis=0)
        sum_W_sq = np.sum(W**2, axis=0)
        
        norm_sq = np.sum(sum_U_sq * sum_V_sq * sum_W_sq)
        
        # Error = norm_sq + num_nonzero - 2 * sum(vals)
        err = norm_sq + num_nonzero - 2 * np.sum(vals)
        
        return err

    # Search strategy
    best_rank = 48
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    rank_configs = [
        (24, 31, 50),
        (31, 36, 30),
        (36, 41, 20),
        (41, 48, 10)
    ]
    
    for start_r, end_r, num_trials in rank_configs:
        for R in range(start_r, end_r):
            if time.time() > deadline:
                break
            
            found = False
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                
                np.random.seed(42 + R * 1000 + trial)
                # Random initialization
                scale = 1.0
                U = np.random.randn(nm, R) * scale
                V = np.random.randn(mp, R) * scale
                W = np.random.randn(pn, R) * scale
                
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                try:
                    res = opt.minimize(objective, params, args=(R,), method='L-BFGS-B', 
                                       options={'maxiter': 100, 'ftol': 1e-8})
                    
                    if res.success or res.fun < 1e-6:
                        U_opt = res.x[:R*nm].reshape(nm, R)
                        V_opt = res.x[R*nm:R*nm+R*mp].reshape(mp, R)
                        W_opt = res.x[R*nm+R*mp:].reshape(pn, R)
                        
                        # Round to half-integers
                        U_r = np.round(U_opt * 2) / 2
                        V_r = np.round(V_opt * 2) / 2
                        W_r = np.round(W_opt * 2) / 2
                        
                        if verify(U_r, V_r, W_r, R):
                            best_rank = R
                            best_U = U_r
                            best_V = V_r
                            best_W = W_r
                            found = True
                            break
                except Exception:
                    pass
            
            if found:
                break
        if found:
            break

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
