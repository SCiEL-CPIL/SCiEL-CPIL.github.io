import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Target tensor components
    # T[i, j, j', k, i', k'] = 1 if i==i', j==j', k==k' else 0
    # We want sum_alpha U[i,j,alpha] * V[j',k,alpha] * W[i',k',alpha] == T
    
    # Flattened indices
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r) -> Task says W[k*n+i, r], so W is (p*n, r) corresponding to k (0..p-1), i (0..n-1)
    # Note: The tensor equation uses W[i', k'] in standard math notation (C_{i',k'}).
    # The task defines W index as k*n+i. So W[k*n+i, alpha] corresponds to C_{i,k}.
    # Let's map carefully.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T_{A,B,C} = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
    # Decomposition: T = sum_alpha U_alpha \otimes V_alpha \otimes W_alpha
    # U_alpha is n x m. V_alpha is m x p. W_alpha is n x p.
    # Equation: sum_alpha U_{i,j}^{alpha} V_{j',k}^{alpha} W_{i',k'}^{alpha} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # Variables to optimize:
    # x_U: r * n * m
    # x_V: r * m * p
    # x_W: r * p * n  (flattened as k*n + i to match task's W layout? 
    # Task W is (p*n, r). So column alpha is a vector of length p*n.
    # Index in W vector: idx = k*n + i.
    # So W_alpha[k*n + i] corresponds to C_{i,k} component in the tensor product?
    # Yes, if we align W_alpha with the C space.
    
    def residual(x, r):
        # x is flat array of size r*(nm + mp + pn)
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U_flat = x[:r*size_u].reshape(r, size_u)
        V_flat = x[r*size_u:r*(size_u+size_v)].reshape(r, size_v)
        W_flat = x[r*(size_u+size_v):].reshape(r, size_w)
        
        # Residual is sum over all indices of (sum_alpha ... - T)^2
        # We can compute this efficiently using einsum or loops.
        # Since n,m,p are small, loops are fine.
        
        res = 0.0
        # Precompute U, V, W as 3D arrays for clarity?
        # U_arr[alpha, i, j]
        # V_arr[alpha, j, k]
        # W_arr[alpha, i, k]  (Note: W_flat index is k*n+i. So W_arr[alpha, i, k] = W_flat[alpha, k*n+i])
        
        U_arr = U_flat.reshape(r, n, m)
        V_arr = V_flat.reshape(r, m, p)
        # W_flat is (r, p*n). We need W_arr (r, n, p) where W_arr[a, i, k] = W_flat[a, k*n+i]
        # Let's just access W_flat directly in the sum.
        
        for i in range(n):
            for j in range(m):
                for jp in range(m): # j'
                    for k in range(p):
                        for ip in range(n): # i'
                            for kp in range(p): # k'
                                # Target
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                
                                # Sum over alpha
                                s = 0.0
                                for a in range(r):
                                    u_val = U_arr[a, i, j]
                                    v_val = V_arr[a, jp, k]
                                    # W index for (ip, kp) is kp*n + ip
                                    w_val = W_flat[a, kp * n + ip]
                                    s += u_val * v_val * w_val
                                res += (s - target) ** 2
                                
        return res

    def snap_and_verify(U, V, W):
        # Snap to half-integers
        U_sn = np.round(U * 2) / 2
        V_sn = np.round(V * 2) / 2
        W_sn = np.round(W * 2) / 2
        
        # Verify exactness
        # Check if sum_alpha U_sn[a,i,j] V_sn[a,jp,k] W_sn[a,kp*n+ip] == delta
        # We can do this with numpy for speed
        U_arr = U_sn.reshape(r, n, m)
        V_arr = V_sn.reshape(r, m, p)
        W_arr = W_sn.reshape(r, p, n) # W_sn is (r, p*n). Reshape to (r, p, n) so W_arr[a, k, i]
        
        # Compute tensor product sum
        # Res[a, i, j, jp, k, ip, kp] ? No.
        # We want S[i, jp, k, ip, kp] = sum_a U[a,i,jp] V[a,jp,k] W[a,kp,ip]
        # Wait, indices:
        # U: a, i, j
        # V: a, j, k
        # W: a, k, i (in W_arr)
        
        # Let's just loop for verification since it's small
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                s = 0.0
                                for a in range(r):
                                    s += U_arr[a, i, j] * V_arr[a, jp, k] * W_arr[a, kp, ip]
                                if abs(s - target) > 1e-9:
                                    return False, None, None, None
        return True, U_sn, V_sn, W_sn

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 11 up to schoolbook
    # Known lower bound for 2x3x4 is 11.
    for r in range(11, best_rank + 1):
        if time.time() > deadline:
            break
            
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_vars = r * (size_u + size_v + size_w)
        
        # Multiple random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            x0 = np.random.randn(total_vars) * 0.5
            # Try to optimize
            try:
                res = minimize(residual, x0, args=(r,), method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < 1e-6:
                    # Extract U, V, W
                    U_flat = res.x[:r*size_u].reshape(r, size_u)
                    V_flat = res.x[r*size_u:r*(size_u+size_v)].reshape(r, size_v)
                    W_flat = res.x[r*(size_u+size_v):].reshape(r, size_w)
                    
                    # Snap and verify
                    is_valid, U_sn, V_sn, W_sn = snap_and_verify(U_flat, V_flat, W_flat)
                    if is_valid:
                        # Found a valid decomposition of rank r
                        best_rank = r
                        best_U = U_sn
                        best_V = V_sn
                        best_W = W_sn
                        # Since we want minimal rank, we can stop if r is small?
                        # But maybe a lower rank exists? We are iterating up from 11.
                        # So the first valid one is the minimal in this range.
                        # We can break out of restarts and ranks?
                        # Yes, if we assume 11 is the global minimum.
                        # But to be safe, we could check if r > 11, maybe 10 exists?
                        # Lower bound is 11, so 11 is minimal possible.
                        if r == 11:
                            break
                        else:
                            # If we found 12, maybe 11 exists?
                            # But we started at 11. If 11 failed, we are at 12.
                            # If 12 works, it's better than 24.
                            # We should return the best found so far if time runs out.
                            pass
            except Exception as e:
                continue
        
        if r == 11 and best_U is not None:
            break

    # If no low rank found, fall back to schoolbook
    if best_U is None:
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
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
