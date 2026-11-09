"""Rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Target tensor T[i, j, k] = 1 if i + k == j else 0
    # Shape: (n, m, p)
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i + k == j:
                    T[i, j, k] = 1.0

    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6

    # Function to evaluate the error of a rank-r decomposition
    # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
    # Reconstruction: sum_c U[:,c] * V[:,c] * W[:,c] mapped back to tensor
    def compute_error(U, V, W):
        r = U.shape[1]
        # Efficient reconstruction using broadcasting
        # We need to map U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] to T[i,j,k]
        # This is tricky with simple broadcasting due to index mixing.
        # We can reshape and use einsum-like logic or explicit loops if r is small.
        # For speed, let's use a vectorized approach.
        
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # We want sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        # This is a contraction over j, k, c.
        # Let's build the full tensor R
        R = np.zeros((n, m, p))
        for c in range(r):
            # U_c[i,j], V_c[j,k], W_c[k,i]
            # R[i,j,k] += U_c[i,j] * V_c[j,k] * W_c[k,i]
            # We can do this with matrix multiplications or loops
            # Loop over i, j, k is too slow if r is large, but here dims are small.
            # Let's optimize:
            # For fixed i, j: R[i,j,:] += U_c[i,j] * (V_c[j,:] * W_c[:,i])
            for i in range(n):
                for j in range(m):
                    # V_c[j, :] is length p
                    # W_c[:, i] is length p
                    # product is length p
                    prod = V_reshaped[j, :, c] * W_reshaped[:, i, c]
                    R[i, j, :] += U_reshaped[i, j, c] * prod
        
        diff = R - T
        return np.linalg.norm(diff)

    def try_find_decomposition(rank, max_iterations=500, seed=0):
        """
        Attempt to find a rank-`rank` decomposition using random restarts and local search.
        Returns (U, V, W) if found, else None.
        """
        np.random.seed(seed)
        best_err = np.inf
        best_factors = None
        
        # Number of restarts depends on time budget
        # Estimate time per iteration
        # For small ranks, compute_error is fast.
        
        for restart in range(20): # 20 restarts per rank
            if time.time() > deadline - 5:
                break
            
            # Initialize with random Gaussian
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Simple gradient descent or Nelder-Mead?
            # Given the small dimensionality (e.g., 6*18 + 9*18 + 6*18 = 324 params for rank 18),
            # Nelder-Mead might be slow.
            # Let's use a simple stochastic gradient descent with noise.
            
            current_err = compute_error(U, V, W)
            if current_err < best_err:
                best_err = current_err
                best_factors = (U.copy(), V.copy(), W.copy())
                
            # Local optimization loop
            step_size = 0.1
            for it in range(max_iterations):
                if time.time() > deadline - 2:
                    break
                
                # Compute gradients numerically? Too slow.
                # Use random perturbations (simulated annealing style)
                # Or use the fact that the problem is smooth.
                # Let's try a simple coordinate descent or random walk.
                
                # Perturb one factor
                factor_idx = np.random.randint(3)
                if factor_idx == 0:
                    dU = np.random.randn(dim_u, rank) * step_size
                    new_U = U + dU
                    err = compute_error(new_U, V, W)
                    if err < current_err:
                        U = new_U
                        current_err = err
                        if current_err < best_err:
                            best_err = current_err
                            best_factors = (U.copy(), V.copy(), W.copy())
                elif factor_idx == 1:
                    dV = np.random.randn(dim_v, rank) * step_size
                    new_V = V + dV
                    err = compute_error(U, new_V, W)
                    if err < current_err:
                        V = new_V
                        current_err = err
                        if current_err < best_err:
                            best_err = current_err
                            best_factors = (U.copy(), V.copy(), W.copy())
                else:
                    dW = np.random.randn(dim_w, rank) * step_size
                    new_W = W + dW
                    err = compute_error(U, V, new_W)
                    if err < current_err:
                        W = new_W
                        current_err = err
                        if current_err < best_err:
                            best_err = current_err
                            best_factors = (U.copy(), V.copy(), W.copy())
                
                # Adaptive step size
                if it % 100 == 0:
                    step_size *= 0.95
                
                if current_err < 1e-8:
                    break

        if best_err < 1e-6:
            return best_factors
        return None

    # Schoolbook decomposition (Rank 18)
    def get_schoolbook():
        U_list = []
        V_list = []
        W_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n)
                    v = np.zeros(m)
                    w = np.zeros(p)
                    u[i] = 1.0
                    v[j] = 1.0
                    w[k] = 1.0
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
        
        rank = len(U_list)
        U_mat = np.zeros((dim_u, rank))
        V_mat = np.zeros((dim_v, rank))
        W_mat = np.zeros((dim_w, rank))
        
        for c in range(rank):
            u = U_list[c]
            v = V_list[c]
            w = W_list[c]
            for i in range(n):
                for j in range(m):
                    U_mat[i * m + j, c] = u[i] * v[j]
            for j in range(m):
                for k in range(p):
                    V_mat[j * p + k, c] = v[j] * w[k]
            for k in range(p):
                for i in range(n):
                    W_mat[k * n + i, c] = w[k] * u[i]
        return U_mat, V_mat, W_mat

    # Try to find lower rank
    found = None
    # Theoretical lower bound is 18 for 2x3x3? 
    # Actually, for 2x2x2 it's 7. For 2x3x3, it might be 18.
    # Let's try ranks 15, 16, 17, 18.
    # If we find a valid one for 17, we return it.
    
    for r in range(15, 19):
        if time.time() > deadline - 10:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            res = try_find_decomposition(r, max_iterations=200, seed=seed + r*100)
            if res is not None:
                found = res
                # Verify strictly
                U, V, W = res
                err = compute_error(U, V, W)
                if err < 1e-9:
                    # Found a valid decomposition of rank r
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                else:
                    # Close but not exact. Keep as candidate if better than schoolbook?
                    # No, must be exact.
                    pass
        
    # Fallback to schoolbook
    U_mat, V_mat, W_mat = get_schoolbook()
    
    # Final verification
    err = compute_error(U_mat, V_mat, W_mat)
    if err > 1e-9:
        # This should not happen
        raise ValueError("Schoolbook decomposition failed verification")

    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
