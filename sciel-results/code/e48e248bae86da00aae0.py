import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    num_a = n * m
    num_b = m * p
    num_c = p * n
    
    # Target tensor T[i, j, k] = 1
    # Equation: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    
    def verify(U, V, W, tol=1e-9):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        r = U.shape[1]
        # Reconstruct sum
        # We need to check for all i, j, k
        # Sum_l U[i*m+j, l] V[j*p+k, l] W[k*n+i, l]
        # This is a batched operation.
        # Create indices
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        # Gather
        U_vals = U[u_idx, :] # (n, m, p, r)
        V_vals = V[v_idx, :]
        W_vals = W[w_idx, :]
        
        prod = U_vals * V_vals * W_vals
        sum_l = np.sum(prod, axis=3)
        
        if np.allclose(sum_l, 1.0, atol=tol):
            return True
        return False

    def try_rank(r):
        # Initialize with random values
        # Use a seed for reproducibility if needed, but random is fine for search
        np.random.seed(42 + r)
        U = np.random.randn(num_a, r)
        V = np.random.randn(num_b, r)
        W = np.random.randn(num_c, r)
        
        # Optimize using simple gradient descent or ALS
        # Let's use ALS (Alternating Least Squares)
        # Fix V, W, solve for U.
        # U[i*m+j, l] * (V[j*p+k, l] * W[k*n+i, l]) = 1
        # For a fixed (i,j), we have:
        # sum_l U[i*m+j, l] * X_l(i,j) = 1
        # where X_l(i,j) = sum_k V[j*p+k, l] * W[k*n+i, l]
        
        # This is a linear system for each (i,j).
        # Actually, it's easier to just do gradient descent on the error surface.
        
        error = 1e10
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 10:
                break
            
            # Compute gradients
            # E = sum_{i,j,k} (sum_l U_{ij}^l V_{jk}^l W_{ki}^l - 1)^2
            # dE/dU_{ij}^l = 2 * (sum ... - 1) * V_{jk}^l W_{ki}^l
            
            # We can vectorize this.
            i_idx, j_idx, k_idx = np.indices((n, m, p))
            u_idx = i_idx * m + j_idx
            v_idx = j_idx * p + k_idx
            w_idx = k_idx * n + i_idx
            
            U_vals = U[u_idx, :] # (n, m, p, r)
            V_vals = V[v_idx, :]
            W_vals = W[w_idx, :]
            
            prod = U_vals * V_vals * W_vals
            sum_l = np.sum(prod, axis=3)
            diff = sum_l - 1.0
            
            # Grad U
            grad_U = np.zeros_like(U)
            # grad_U[u_idx, l] += diff * V_vals * W_vals
            # We need to accumulate over k for fixed i,j
            # grad_U_flat = np.zeros((n*m, r))
            # For each i,j,k: grad_U[i*m+j, l] += diff[i,j,k] * V[j*p+k, l] * W[k*n+i, l]
            
            # Vectorized accumulation
            # Create a tensor of gradients for U
            gU_tensor = diff[..., np.newaxis] * V_vals * W_vals
            # Sum over k axis (axis 2)
            gU_2d = np.sum(gU_tensor, axis=2) # (n, m, r)
            # Reshape to (n*m, r)
            gU = gU_2d.reshape(num_a, r)
            
            # Grad V
            gV_tensor = diff[..., np.newaxis] * U_vals * W_vals
            # Sum over i axis (axis 0) for fixed j,k?
            # V index is j*p+k. So for fixed j,k, we sum over i.
            # gV_2d = np.sum(gV_tensor, axis=0) # (m, p, r)
            gV_2d = np.sum(gV_tensor, axis=0)
            gV = gV_2d.reshape(num_b, r)
            
            # Grad W
            gW_tensor = diff[..., np.newaxis] * U_vals * V_vals
            # Sum over j axis (axis 1) for fixed i,k?
            # W index is k*n+i. So for fixed i,k, we sum over j.
            # gW_2d = np.sum(gW_tensor, axis=1) # (n, p, r)
            gW_2d = np.sum(gW_tensor, axis=1)
            # Reshape to (p*n, r)
            # gW_2d is (n, p, r). We need (p, n, r) then flatten?
            # W index is k*n+i.
            # gW_2d[i, k, l] corresponds to W[k*n+i, l]
            gW_2d_perm = np.transpose(gW_2d, (1, 0, 2)) # (p, n, r)
            gW = gW_2d_perm.reshape(num_c, r)
            
            # Update
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
            error = np.sum(diff**2)
            if error < 1e-12:
                break
        
        if verify(U, V, W):
            return U, V, W
        return None

    # Try ranks
    for r in [30, 32, 34, 36, 38, 40]:
        if time.time() > deadline - 20:
            break
        res = try_rank(r)
        if res is not None:
            U, V, W = res
            # Convert to list of lists of floats
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook rank 40
    rank = 40
    U = [[0] * rank for _ in range(num_a)]
    V = [[0] * rank for _ in range(num_b)]
    W = [[0] * rank for _ in range(num_c)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
