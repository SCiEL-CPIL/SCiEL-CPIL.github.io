import time
import random
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Target tensor T[i, j, k] = 1
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        # U, V, W are lists of lists or numpy arrays
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check shapes
        if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
            return False
            
        # Compute tensor
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can vectorize this.
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U_np.reshape(n, m, -1)
        V_3d = V_np.reshape(m, p, -1)
        W_3d = W_np.reshape(p, n, -1)
        
        # Sum over rank axis
        # T[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
        # This is a batched dot product?
        # We can do:
        # Temp[i, j, k] = sum_c (U_3d[i, j, c] * V_3d[j, k, c]) * W_3d[k, i, c]
        
        # Efficient way:
        # Create a tensor of shape (n, m, p, rank)
        # T[i, j, k, c] = U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
        # This is expensive to build explicitly.
        
        # Instead, iterate over c?
        # Or use einsum:
        # T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        try:
            T_calc = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        except Exception:
            return False
            
        if not np.allclose(T_calc, 1.0, atol=1e-9):
            return False
            
        # Check if entries are close to half-integers
        # The problem says "integer/half-integer".
        # We should check if 2 * entry is close to an integer.
        for mat in [U_np, V_np, W_np]:
            if not np.allclose(2 * mat, np.round(2 * mat), atol=1e-6):
                return False
                
        return True

    def run_als_search(rank, max_iters=1000, lr=0.01):
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute Residual
            T_calc = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            Residual = 1.0 - T_calc
            
            if np.allclose(Residual, 0, atol=1e-9):
                # Check half-integer constraint
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    return U, V, W
                
            # Compute Gradients
            # dE/dU[i,j,c] = -2 * sum_k Residual[i,j,k] * V[j,k,c] * W[k,i,c]
            # dE/dV[j,k,c] = -2 * sum_i Residual[i,j,k] * U[i,j,c] * W[k,i,c]
            # dE/dW[k,i,c] = -2 * sum_j Residual[i,j,k] * U[i,j,c] * V[j,k,c]
            
            # Use einsum for gradients
            # G_U[i,j,c] = -2 * einsum('ijk,jkc,kic->ijc', Residual, V_3d, W_3d)
            G_U = -2.0 * np.einsum('ijk,jkc,kic->ijc', Residual, V_3d, W_3d)
            G_V = -2.0 * np.einsum('ijk,ijc,kic->jkc', Residual, U_3d, W_3d)
            G_W = -2.0 * np.einsum('ijk,ijc,jkc->kic', Residual, U_3d, V_3d)
            
            # Update
            U -= lr * G_U
            V -= lr * G_V
            W -= lr * G_W
            
            # Reshape
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            # Periodic Rounding and Check
            if it % 10 == 0:
                # Round to nearest half-integer
                U_r = np.round(2 * U) / 2.0
                V_r = np.round(2 * V) / 2.0
                W_r = np.round(2 * W) / 2.0
                
                if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return U_r, V_r, W_r
                    
            # Decay learning rate
            lr *= 0.999

        # Final check
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return U, V, W
        return None

    best_decomp = get_schoolbook()
    
    # Try ranks from 28 up to 29
    for rank in [28, 29]:
        if time.time() > deadline - 10:
            break
            
        # Run multiple restarts
        found = False
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            result = run_als_search(rank, max_iters=500, lr=0.05)
            if result is not None:
                U, V, W = result
                best_decomp = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
                
        if found:
            break

    return best_decomp
