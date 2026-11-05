"""Program for 4x5 * 5x6 matrix multiplication tensor decomposition.
Uses numerical optimization to find a low-rank exact decomposition.
"""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np
import scipy.optimize as opt
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 140  # safety margin from 150s limit

    n, m, p = 4, 5, 6
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24

    def idxU(i, j): return i * m + j
    def idxV(j, k): return j * p + k
    def idxW(k, i): return k * n + i

    # Target tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1
    # We want sum_r U[:,r] V[:,r] W[:,r] = T
    
    # Precompute the non-zero entries of T as a list of (idxU, idxV, idxW)
    target_entries = []
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = idxU(i, j)
                v_idx = idxV(j, k)
                w_idx = idxW(k, i)
                target_entries.append((u_idx, v_idx, w_idx))
                target_set.add((u_idx, v_idx, w_idx))
                
    # Fallback: Schoolbook
    best_rank = n * m * p
    U_fb = np.zeros((num_U, best_rank))
    V_fb = np.zeros((num_V, best_rank))
    W_fb = np.zeros((num_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[idxU(i, j), col] = 1
                V_fb[idxV(j, k), col] = 1
                W_fb[idxW(k, i), col] = 1
                col += 1
    best_result = (U_fb, V_fb, W_fb)

    # Precompute indices for error calculation
    # We will compute the full tensor T and check errors.
    # T shape: (num_U, num_V, num_W) = (20, 30, 24) -> 14400 elements.
    
    def compute_tensor(U, V, W):
        """Compute tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]"""
        # Using einsum for efficiency
        return np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

    def error_func(params, R):
        size_U = num_U * R
        size_V = num_V * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(R, num_U).T # (num_U, R)
        V = V_flat.reshape(R, num_V).T
        W = W_flat.reshape(R, num_W).T
        
        T = compute_tensor(U, V, W)
        
        # Error on targets: (T - 1)^2
        # Error on non-targets: T^2
        # We can create a mask once, but here we just sum the squares.
        # To avoid creating a huge mask every time, we can subtract 1 from targets and square,
        # but we need to handle the mask.
        
        # Efficient way:
        # err = sum((T[target_indices] - 1)^2) + sum((T[non_target_indices])^2)
        
        # Since T is small, let's just use a precomputed mask array.
        # We'll pass the mask via closure or global? No, let's keep it local to construct scope.
        # We defined non_target_mask in the original, let's define it here.
        
        # Accessing non_target_mask from outer scope
        err_target = np.sum((T[target_indices] - 1.0)**2)
        err_non_target = np.sum((T * non_target_mask)**2)
        
        return err_target + err_non_target

    # Precompute indices and mask for fast error calculation
    target_indices = np.array(target_entries)
    non_target_mask = np.ones((num_U, num_V, num_W))
    non_target_mask[target_indices[:,0], target_indices[:,1], target_indices[:,2]] = 0

    def verify_decomposition(U, V, W, R):
        """
        Verifies if U, V, W of shape (rows, R) decompose the tensor.
        Returns True if valid.
        """
        T = compute_tensor(U, V, W)
        
        # Check target entries
        target_vals = T[target_indices[:,0], target_indices[:,1], target_indices[:,2]]
        if np.max(np.abs(target_vals - 1.0)) > 1e-6:
            return False
        
        # Check non-target entries
        non_target_vals = T * non_target_mask
        if np.max(np.abs(non_target_vals)) > 1e-6:
            return False
            
        return True

    # Try to find lower rank
    # We will try ranks from 115 down to 100 to save time.
    # Schoolbook is 120.
    
    for R in range(115, 99, -1):
        if time.time() - t0 > DEADLINE - 10:
            break
            
        # Try a few random seeds
        for seed in [42 + R, 100 + R, 200 + R]:
            if time.time() - t0 > DEADLINE - 10:
                break
                
            np.random.seed(seed)
            U_init = np.zeros((num_U, R))
            V_init = np.zeros((num_V, R))
            W_init = np.zeros((num_W, R))
            
            # Random initialization with sparse structure
            # Each column has a few non-zeros
            for r in range(R):
                # Pick random indices
                num_nonzero = 2
                u_idx = np.random.choice(num_U, num_nonzero, replace=False)
                v_idx = np.random.choice(num_V, num_nonzero, replace=False)
                w_idx = np.random.choice(num_W, num_nonzero, replace=False)
                
                U_init[u_idx, r] = 1.0
                V_init[v_idx, r] = 1.0
                W_init[w_idx, r] = 1.0
                
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            # Use L-BFGS-B for speed
            try:
                res = opt.minimize(error_func, x0, args=(R,), method='L-BFGS-B', 
                                   options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    # Recover matrices
                    size_U = num_U * R
                    size_V = num_V * R
                    U = res.x[:size_U].reshape(R, num_U).T
                    V = res.x[size_U:size_U+size_V].reshape(R, num_V).T
                    W = res.x[size_U+size_V:].reshape(R, num_W).T
                    
                    # Try to snap to integers/half-integers
                    U_snapped = np.round(U * 2) / 2.0
                    V_snapped = np.round(V * 2) / 2.0
                    W_snapped = np.round(W * 2) / 2.0
                    
                    # Verify snapped version
                    if verify_decomposition(U_snapped, V_snapped, W_snapped, R):
                        best_rank = R
                        best_result = (U_snapped, V_snapped, W_snapped)
                        break # Found a valid one, stop searching lower ranks
            except Exception as e:
                pass

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
