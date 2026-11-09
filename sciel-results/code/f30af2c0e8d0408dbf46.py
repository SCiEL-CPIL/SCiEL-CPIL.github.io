import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, rank_schoolbook))
    V0 = np.zeros((m*p, rank_schoolbook))
    W0 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
                
    def compute_error_and_grad(U, V, W, r):
        """
        Computes the error and the gradient for the continuous relaxation.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        """
        # Reshape to 3D for easier index handling
        # U[i, j, r] -> U.reshape(n, m, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # S[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Using einsum: 'ijr,jkr,kir->ijk'
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Target is 1
        diff = 1.0 - S
        error = np.sum(diff ** 2)
        
        # Gradients
        # dE/dU[i,j,r] = 2 * diff[i,j,k] * V[j,k,r] * W[k,i,r]
        # Note: k is summed over in the gradient calculation?
        # dS[i,j,k]/dU[i,j,r] = V[j,k,r] * W[k,i,r]
        # dE/dU[i,j,r] = sum_k 2 * diff[i,j,k] * V[j,k,r] * W[k,i,r]
        
        # Let's compute grad_U
        # diff: (n, m, p)
        # V_3d: (m, p, r)
        # W_3d: (p, n, r)
        
        # We need to sum over k (index 2 of diff, index 1 of V, index 0 of W)
        # grad_U[i, j, r] = 2 * sum_k diff[i, j, k] * V[j, k, r] * W[k, i, r]
        
        # Einsum for grad_U:
        # diff: 'ijk'
        # V: 'jkr'
        # W: 'kir'
        # Result: 'ijr'
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', diff, V_3d, W_3d)
        
        # Similarly for V:
        # grad_V[j, k, r] = 2 * sum_i diff[i, j, k] * U[i, j, r] * W[k, i, r]
        # diff: 'ijk'
        # U: 'ijr'
        # W: 'kir'
        # Result: 'jkr'
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', diff, U_3d, W_3d)
        
        # Similarly for W:
        # grad_W[k, i, r] = 2 * sum_j diff[i, j, k] * U[i, j, r] * V[j, k, r]
        # diff: 'ijk'
        # U: 'ijr'
        # V: 'jkr'
        # Result: 'kir'
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U_3d, V_3d)
        
        # Flatten gradients
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad_flat

    def unpack(x, r):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = x[:size_U].reshape(n, m, r)
        V = x[size_U:size_U+size_V].reshape(m, p, r)
        W = x[size_U+size_V:].reshape(p, n, r)
        
        return U, V, W

    def objective(x, r):
        U, V, W = unpack(x, r)
        err, _ = compute_error_and_grad(U, V, W, r)
        return err

    def jac(x, r):
        U, V, W = unpack(x, r)
        _, grad = compute_error_and_grad(U, V, W, r)
        return grad

    def try_continuous_rank(r, num_restarts=30):
        nonlocal best_U, best_V, best_W, best_rank, best_error
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            # Scale is important. Values around 1 are good.
            scale = 1.0
            x0 = np.random.normal(0, scale, size=(n*m + m*p + p*n) * r)
            
            # Sometimes initializing with 0.5 or 1.0 helps
            if restart % 3 == 0:
                 x0 = np.random.choice([0.5, 1.0, -0.5, -1.0], size=x0.shape)
            
            try:
                res = minimize(
                    objective, x0, args=(r,),
                    jac=lambda x: jac(x, r),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < best_error:
                    U, V, W = unpack(res.x, r)
                    err = res.fun
                    
                    if err < best_error:
                        best_error = err
                        best_U, best_V, best_W = U, V, W
                        best_rank = r
                        
                    # Try to snap to half-integers
                    if err < 1e-4: # Close enough to try snapping
                        U_snapped = np.round(U * 2) / 2
                        V_snapped = np.round(V * 2) / 2
                        W_snapped = np.round(W * 2) / 2
                        
                        # Verify exactness of snapped version
                        U_s_3d = U_snapped.reshape(n, m, r)
                        V_s_3d = V_snapped.reshape(m, p, r)
                        W_s_3d = W_snapped.reshape(p, n, r)
                        S_snap = np.einsum('ijr,jkr,kir->ijk', U_s_3d, V_s_3d, W_s_3d)
                        snap_err = np.sum((S_snap - 1.0)**2)
                        
                        if snap_err < 1e-9:
                            best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                            best_rank = r
                            best_error = 0.0
                            return # Found exact solution
                            
            except Exception:
                continue
        
        if best_rank == 16 and best_error < 1e-9:
            return

    # Check schoolbook first
    err0 = compute_error_and_grad(U0, V0, W0, rank_schoolbook)[0]
    if err0 < 1e-9:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook
        best_error = err0

    # Try lower ranks with continuous optimization
    # Rank 16 is the known minimum for 2x3x3
    for r in [16, 17, 18, 19, 20]:
        if time.time() > deadline - 10:
            break
        try_continuous_rank(r, num_restarts=50)
        if best_rank == r and best_error < 1e-9:
            break

    # If continuous search didn't find an exact solution, try discrete local search from the best continuous result
    if best_error > 1e-9 and best_U is not None:
        # Use the best continuous result as a starting point for discrete search
        # But we need to ensure we stay within time limits
        
        # We'll do a few steps of discrete refinement on the best candidate found so far
        # If we didn't find anything, we might have to fall back to random discrete search
        # But usually continuous finds the basin.
        
        # Let's try a quick discrete polish on the best_U, best_V, best_W if they exist
        if best_U is not None:
            U = best_U.copy()
            V = best_V.copy()
            W = best_W.copy()
            r = best_rank
            
            # Snap to half integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            err, _ = compute_error_and_grad(U, V, W, r)
            
            # Simple local search
            allowed_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
            max_steps = 1000
            for _ in range(max_steps):
                if time.time() > deadline - 2:
                    break
                if err < 1e-9:
                    break
                
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    rows, cols = n*m, r
                elif mat_idx == 1:
                    mat = V
                    rows, cols = m*p, r
                else:
                    mat = W
                    rows, cols = p*n, r
                
                i = random.randint(0, rows - 1)
                j = random.randint(0, cols - 1)
                
                old_val = mat[i, j]
                new_val = allowed_vals[random.randint(0, len(allowed_vals)-1)]
                
                if new_val == old_val:
                    continue
                
                mat[i, j] = new_val
                new_err, _ = compute_error_and_grad(U, V, W, r)
                
                if new_err < err:
                    err = new_err
                else:
                    mat[i, j] = old_val
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U, V, W
                best_rank = r

    # Ensure we have a valid fallback
    if best_U is None:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook
        best_error = err0

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                nearest = round(val * 2) / 2
                new_row.append(nearest)
            cleaned.append(new_row)
        return cleaned

    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
