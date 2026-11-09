import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor dimensions
    dim_u = n * m  # 8
    dim_v = m * p  # 16
    dim_w = p * n  # 8
    
    # Create the target tensor T as a list of (index_u, index_v, index_w, value)
    # T is sparse, only 32 non-zero entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target_entries.append((idx_u, idx_v, idx_w, 1.0))
    
    # We want to find U, V, W of rank R
    # We will try ranks from 28 down to 24
    
    def residual(x, R):
        # x is flattened U, V, W
        # Size: dim_u * R + dim_v * R + dim_w * R
        size_u = dim_u * R
        size_v = dim_v * R
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u + size_v]
        W_flat = x[size_u + size_v:]
        
        U = U_flat.reshape(dim_u, R)
        V = V_flat.reshape(dim_v, R)
        W = W_flat.reshape(dim_w, R)
        
        # Compute the tensor product sum
        # T_calc[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We only need to check the non-zero entries of the target
        res = []
        for idx_u, idx_v, idx_w, val in target_entries:
            calc_val = 0.0
            for r in range(R):
                calc_val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
            res.append(calc_val - val)
            
        # Also, we should ensure that the entries where T is 0 are 0?
        # The tensor is defined on the whole space.
        # However, if we only constrain the non-zero entries, we might get spurious non-zero entries elsewhere.
        # But the tensor for matrix multiplication is zero elsewhere.
        # So we must enforce T_calc[a, b, c] = 0 for all (a, b, c) not in target_entries.
        # This is a lot of constraints: 8 * 16 * 8 = 1024 entries.
        # 32 are 1, 992 are 0.
        # We should include the zero entries in the residual to ensure exactness.
        
        # Let's include all entries
        # This is 1024 residuals.
        # x has 32R variables.
        # For R=28, 896 variables, 1024 residuals. Overdetermined.
        
        # Let's re-evaluate. The residual function above only checks 32 entries.
        # We need to check all 1024 entries.
        
        res_full = []
        for a in range(dim_u):
            for b in range(dim_v):
                # Optimization: precompute U[a, :] * V[b, :]
                # This is a vector of length R
                # Then dot with W[c, :]
                u_a = U[a, :]
                v_b = V[b, :]
                prod = u_a * v_b
                for c in range(dim_w):
                    calc_val = np.dot(prod, W[c, :])
                    # Find target value
                    # We can use a lookup or just check if (a,b,c) is in target
                    # For speed, we can precompute the target tensor as a dense array
                    # But 1024 floats is small.
                    res_full.append(calc_val - target_dense[a, b, c])
        
        return res_full

    # Precompute target dense tensor
    target_dense = np.zeros((dim_u, dim_v, dim_w))
    for idx_u, idx_v, idx_w, val in target_entries:
        target_dense[idx_u, idx_v, idx_w] = val

    # Modify residual to use target_dense
    def residual(x, R):
        size_u = dim_u * R
        size_v = dim_v * R
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u + size_v]
        W_flat = x[size_u + size_v:]
        
        U = U_flat.reshape(dim_u, R)
        V = V_flat.reshape(dim_v, R)
        W = W_flat.reshape(dim_w, R)
        
        # T_calc = sum_r U[:, r] V[:, r]^T W[:, r]^T
        # This is a tensor contraction.
        # We can compute it as:
        # Temp[a, b, r] = U[a, r] V[b, r]
        # T_calc[a, b, c] = sum_r Temp[a, b, r] W[c, r]
        
        # Efficient computation:
        # U is (dim_u, R), V is (dim_v, R), W is (dim_w, R)
        # We want T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        
        # Reshape to (dim_u, R, 1), (dim_v, R, 1), (dim_w, R, 1)
        # This might be memory heavy if R is large, but R is small.
        
        # Let's just loop over r
        T_calc = np.zeros((dim_u, dim_v, dim_w))
        for r in range(R):
            T_calc += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            
        return (T_calc - target_dense).flatten()

    best_result = None
    best_rank = 32
    best_residual = float('inf')
    
    # Start with schoolbook
    U_school = np.zeros((dim_u, 32))
    V_school = np.zeros((dim_v, 32))
    W_school = np.zeros((dim_w, 32))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    x_school = np.concatenate([U_school.flatten(), V_school.flatten(), W_school.flatten()])
    
    # Try to reduce rank
    # We will try R = 28, 26, 24
    # We need a good initial guess.
    # We can take the schoolbook solution and try to project it or randomize it.
    # Random initialization is often better for local minima avoidance.
    
    deadline = time.time() + 140
    
    for R in [28, 26, 24]:
        if time.time() > deadline - 10:
            break
            
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale: entries around 1
            U_init = np.random.randn(dim_u, R)
            V_init = np.random.randn(dim_v, R)
            W_init = np.random.randn(dim_w, R)
            
            # Normalize columns of U to have norm 1 to help convergence
            for r in range(R):
                norm = np.linalg.norm(U_init[:, r])
                if norm > 1e-6:
                    U_init[:, r] /= norm
                else:
                    U_init[0, r] = 1.0
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residual, x0, args=(R,), method='trf', max_nfev=2000)
                if result.cost < 1e-10:
                    # Found a solution
                    best_rank = R
                    best_result = result.x
                    break # Break restart loop, try lower rank
            except Exception:
                pass
                
    if best_result is not None:
        R = best_rank
        size_u = dim_u * R
        size_v = dim_v * R
        U_flat = best_result[:size_u]
        V_flat = best_result[size_u:size_u + size_v]
        W_flat = best_result[size_u + size_v:]
        
        U = U_flat.reshape(dim_u, R)
        V = V_flat.reshape(dim_v, R)
        W = W_flat.reshape(dim_w, R)
        
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
