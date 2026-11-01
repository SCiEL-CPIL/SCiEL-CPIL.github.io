import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # Trivial decomposition as fallback
    U_triv = [[0] * trivial_rank for _ in range(n * m)]
    V_triv = [[0] * trivial_rank for _ in range(m * p)]
    W_triv = [[0] * trivial_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1

    best_rank = trivial_rank
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}

    # Precompute indices for fast evaluation
    # We need to map (i, j, k) to flat indices in U, V, W for each t
    # U shape: (n*m, R) -> index u_idx = i*m + j
    # V shape: (m*p, R) -> index v_idx = j*p + k
    # W shape: (p*n, R) -> index w_idx = k*n + i
    
    # For a given R, x is a vector of length R * (n*m + m*p + p*n)
    # Layout: [U_0...U_{R-1}, V_0...V_{R-1}, W_0...W_{R-1}]
    # U_t is a vector of length n*m
    # V_t is a vector of length m*p
    # W_t is a vector of length p*n
    
    def eval_and_grad(x, R):
        """
        Compute error and gradient for a given x and rank R.
        x is a flat vector of length R * (n*m + m*p + p*n).
        """
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = R * (size_u + size_v + size_w)
        
        if len(x) != total_size:
            return 1e10, np.zeros(total_size)

        # Reshape x into U, V, W
        # x is [U_0...U_{R-1}, V_0...V_{R-1}, W_0...W_{R-1}]
        u_flat = x[:R * size_u]
        v_flat = x[R * size_u : R * size_u + R * size_v]
        w_flat = x[R * size_u + R * size_v :]

        U = u_flat.reshape(R, size_u)
        V = v_flat.reshape(R, size_v)
        W = w_flat.reshape(R, size_w)

        # Compute S_ijk = sum_t U[t, i*m+j] * V[t, j*p+k] * W[t, k*n+i]
        # We need to sum over t for each (i,j,k).
        # This is a tensor contraction.
        # S is a tensor of shape (n, m, p)
        
        # Efficient way:
        # S_ijk = sum_t U[t, i*m+j] V[t, j*p+k] W[t, k*n+i]
        
        # Let's compute this by iterating over t and accumulating?
        # Or using einsum.
        # U is (R, n, m) if reshaped properly?
        # U_flat index is i*m + j. So U[t, i*m+j].
        # Let's reshape U to (R, n, m)
        U_3d = U.reshape(R, n, m)
        V_3d = V.reshape(R, m, p)
        W_3d = W.reshape(R, p, n)
        
        # S_ijk = sum_t U_3d[t, i, j] V_3d[t, j, k] W_3d[t, k, i]
        # Using einsum: 'tij,tjk,tki->ijk'
        S = np.einsum('tij,tjk,tki->ijk', U_3d, V_3d, W_3d)
        
        # Error
        error = np.sum((S - 1.0) ** 2)
        
        # Gradient
        # dE/dU[t, i, j] = 2 * (S[i, j, k] - 1) * V[t, j, k] * W[t, k, i]
        # Note: The sum over k is implicit in the gradient for a fixed U[t,i,j].
        # Wait, S[i,j,k] depends on U[t,i,j] for all k.
        # So dE/dU[t, i, j] = 2 * sum_k (S[i, j, k] - 1) * V[t, j, k] * W[t, k, i]
        
        # Let's compute this efficiently.
        # grad_U[t, i, j] = 2 * sum_k (S[i, j, k] - 1) * V[t, j, k] * W[t, k, i]
        
        # Reshape S to (n, m, p)
        S_minus_1 = S - 1.0
        
        # We need to contract S_minus_1 with V and W.
        # V is (R, m, p), W is (R, p, n).
        # For a fixed t, i, j:
        # sum_k S_minus_1[i, j, k] * V[t, j, k] * W[t, k, i]
        
        # Let's use einsum again.
        # term = einsum('ijk,tjk,tki->tij', S_minus_1, V_3d, W_3d)
        # This gives grad_U / 2.
        
        grad_U_3d = 2.0 * np.einsum('ijk,tjk,tki->tij', S_minus_1, V_3d, W_3d)
        
        # Similarly for V:
        # dE/dV[t, j, k] = 2 * sum_i (S[i, j, k] - 1) * U[t, i, j] * W[t, k, i]
        # term = einsum('ijk,tij,tki->tjk', S_minus_1, U_3d, W_3d)
        grad_V_3d = 2.0 * np.einsum('ijk,tij,tki->tjk', S_minus_1, U_3d, W_3d)
        
        # Similarly for W:
        # dE/dW[t, k, i] = 2 * sum_j (S[i, j, k] - 1) * U[t, i, j] * V[t, j, k]
        # term = einsum('ijk,tij,tjk->tki', S_minus_1, U_3d, V_3d)
        grad_W_3d = 2.0 * np.einsum('ijk,tij,tjk->tki', S_minus_1, U_3d, V_3d)
        
        # Flatten gradients
        grad_U = grad_U_3d.flatten()
        grad_V = grad_V_3d.flatten()
        grad_W = grad_W_3d.flatten()
        
        grad = np.concatenate([grad_U, grad_V, grad_W])
        
        return error, grad

    def check_decomposition(U, V, W, R):
        """Verify exact decomposition with integer/half-integer entries."""
        # U, V, W are lists of lists or numpy arrays
        # We assume they are already rounded to 0.5
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(R):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    # Search for lower ranks
    # Start from 41 down to 25
    for R in range(41, 24, -1):
        if time.time() > deadline - 10:
            break
        
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = R * (size_u + size_v + size_w)
        
        # Try a few random initializations
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random small integers
            # Or near trivial?
            # Let's try random first, it might find a global min faster if landscape is flat
            x0 = np.random.randint(-2, 3, total_size) * 0.5
            
            # Optimization
            try:
                res = minimize(
                    lambda x: eval_and_grad(x, R)[0],
                    x0,
                    jac=lambda x: eval_and_grad(x, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success:
                    x_opt = res.x
                    
                    # Round to nearest 0.5
                    x_rounded = np.round(x_opt * 2) / 2.0
                    
                    # Reshape to U, V, W for checking
                    u_flat = x_rounded[:R * size_u]
                    v_flat = x_rounded[R * size_u : R * size_u + R * size_v]
                    w_flat = x_rounded[R * size_u + R * size_v :]
                    
                    U_check = u_flat.reshape(size_u, R)
                    V_check = v_flat.reshape(size_v, R)
                    W_check = w_flat.reshape(size_w, R)
                    
                    # Convert to list of lists for the output format
                    U_list = U_check.tolist()
                    V_list = V_check.tolist()
                    W_list = W_check.tolist()
                    
                    if check_decomposition(U_list, V_list, W_list, R):
                        best_rank = R
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        break # Break attempt loop, try next lower R? Or return?
                        # We want the minimal rank, so we should try lower R next.
                        # But finding a valid R=41 is already an improvement.
                        # If we find R=41, we should check R=40.
                        
            except Exception:
                pass
        
        # If we found a solution for this R, we can continue to lower R.
        # If we didn't, we continue to lower R anyway (maybe a different init works).
        
    return best_result
