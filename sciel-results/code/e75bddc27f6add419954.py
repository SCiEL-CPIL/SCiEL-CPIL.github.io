import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 20 # Fallback
    
    # Tensor dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T is a 3D array of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def verify(U, V, W):
        # Check if U, V, W decompose T exactly
        # Compute sum of rank 1 tensors
        # U: (dim_u, R), V: (dim_v, R), W: (dim_w, R)
        R = U.shape[1]
        # Efficiently compute the tensor product sum
        # This is a 3-way sum
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can do this via matrix multiplications or broadcasting
        # Broadcasting: U[:, :, None] * V[None, :, :] * W[None, None, :] -> (dim_u, R, dim_v, dim_w) ?? No
        # U[:, :, None] is (dim_u, R, 1)
        # V[None, :, :] is (1, R, dim_v)
        # Product is (dim_u, R, dim_v)
        # Multiply by W[None, None, :] (1, 1, dim_w) -> (dim_u, R, dim_v, dim_w)
        # Sum over R -> (dim_u, dim_v, dim_w)
        
        # Memory might be an issue if R is large, but R~20 is small.
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def snap_to_half_int(x):
        # Snap to nearest 0.5
        return np.round(2 * x) / 2.0

    def try_rank(R):
        best_U, best_V, best_W = None, None, None
        start_time = time.time()
        timeout = 30 # seconds per rank attempt
        
        for restart in range(50):
            if time.time() - start_time > timeout:
                break
                
            # Initialize
            # Use small random integers
            U = np.random.randint(-2, 3, (dim_u, R)).astype(float)
            V = np.random.randint(-2, 3, (dim_v, R)).astype(float)
            W = np.random.randint(-2, 3, (dim_w, R)).astype(float)
            
            # Scale up?
            # The target tensor has 1s. The products U*V*W should sum to 1.
            # If U, V, W are small, the sum might be small.
            # Let's initialize with values that might sum to 1.
            # Actually, gradient descent will handle scaling.
            
            # Gradient Descent
            lr = 0.01
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                
                # Compute residual
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                residual = recon - T
                
                # Compute gradients
                # dL/dU[a, r] = sum_{b,c} residual[a, b, c] V[b, r] W[c, r]
                # dL/dV[b, r] = sum_{a,c} residual[a, b, c] U[a, r] W[c, r]
                # dL/dW[c, r] = sum_{a,b} residual[a, b, c] U[a, r] V[b, r]
                
                grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
                grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
                grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: Snap to half-integers periodically?
                # No, snap only at the end.
                
            # Snap to half-integers
            U_s = snap_to_half_int(U)
            V_s = snap_to_half_int(V)
            W_s = snap_to_half_int(W)
            
            if verify(U_s, V_s, W_s):
                return U_s, V_s, W_s
                
            # Try snapping to integers only?
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if verify(U_i, V_i, W_i):
                return U_i, V_i, W_i

        return None

    # Try ranks from 17 to 19
    for R in [17, 18, 19]:
        res = try_rank(R)
        if res:
            U, V, W = res
            # Convert to lists of lists for the output format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
