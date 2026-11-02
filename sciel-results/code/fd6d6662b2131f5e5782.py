import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 36
    
    # Build the target tensor T flattened
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We represent T as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Trivial solution as fallback
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, idx] = 1.0
                V_triv[j * p + k, idx] = 1.0
                W_triv[k * n + i, idx] = 1.0
                idx += 1

    deadline = time.time() + 110  # Safety margin
    
    # Pre-compute flattened T views
    T_flat_U = T.reshape(n * m, m * p * p * n)
    T_flat_V = T.reshape(m * p, n * m * p * n)
    T_flat_W = T.reshape(p * n, n * m * m * p)
    
    # Try to find a lower rank decomposition
    # We will try ranks from 35 down to 30
    for r in range(35, 29, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations based on the trivial solution
        for trial in range(100):
            if time.time() > deadline:
                break
                
            # Initialization: Take a random subset of trivial columns
            # This provides a good starting point close to the trivial solution
            perm = np.random.permutation(trivial_rank)
            idxs = perm[:r]
            
            U = U_triv[:, idxs].copy()
            V = V_triv[:, idxs].copy()
            W = W_triv[:, idxs].copy()
            
            # Add small random noise to escape local minima
            U += np.random.randn(n * m, r) * 0.01
            V += np.random.randn(m * p, r) * 0.01
            W += np.random.randn(p * n, r) * 0.01
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                # D_U[r, b_idx * p*n + c_idx] = V[b_idx, r] * W[c_idx, r]
                D_U = np.einsum('br,cr->rbc', V, W).reshape(r, m * p * p * n)
                try:
                    U_new = np.linalg.lstsq(D_U, T_flat_U.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue
                    
                # Update V
                # D_V[r, a_idx * p*n + c_idx] = U[a_idx, r] * W[c_idx, r]
                D_V = np.einsum('ar,cr->rac', U, W).reshape(r, n * m * p * n)
                try:
                    V_new = np.linalg.lstsq(D_V, T_flat_V.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue

                # Update W
                # D_W[r, a_idx * m*p + b_idx] = U[a_idx, r] * V[b_idx, r]
                D_W = np.einsum('ar,br->rab', U, V).reshape(r, n * m * m * p)
                try:
                    W_new = np.linalg.lstsq(D_W, T_flat_W.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue

                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.linalg.norm(T - T_approx)
                
                if residual < 1e-8:
                    # Found a solution! Now round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    residual_rounded = np.linalg.norm(T - T_approx_rounded)
                    
                    if residual_rounded < 1e-8:
                        # Valid integer/half-integer solution found
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }

    # Fallback to trivial
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }
