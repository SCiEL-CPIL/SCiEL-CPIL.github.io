import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook solution as fallback
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
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Lower bound estimation
    # Simple bound: max(n*m, m*p, p*n) is too low.
    # Coppersmith-Winograd style lower bound is complex.
    # Let's just try from a reasonable lower bound upwards.
    # For 2x5x6, rank is likely > 50.
    lower_bound = 50
    upper_bound = n * m * p
    
    random.seed(42)
    
    # Pre-compute indices for efficiency
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def compute_loss_and_grads(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # T_ijk = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        
        # Residual R_ijk = T_ijk - 1
        # We want to minimize sum(R_ijk^2)
        
        # Efficient computation:
        # Let A = U (n*m, r), B = V (m*p, r), C = W (p*n, r)
        # We can compute T by iterating over l or using tensor products.
        # Given small dimensions, explicit loops or vectorized numpy ops are fine.
        
        # Vectorized approach:
        # T is a tensor of shape (n, m, p)
        # T[i, j, k] = dot(U[i*m+j], V[j*p+k] * W[k*n+i]) -- No, this is not right.
        # T[i, j, k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        
        # Let's compute T using einsum or loops.
        # T = np.einsum('il,jl,kl->ijk', U_reshaped, V_reshaped, W_reshaped)
        # But U, V, W are 2D. We need to reshape them appropriately.
        
        # U_flat[i, j, l] where i in n, j in m, l in r
        # V_flat[j, k, l] where j in m, k in p, l in r
        # W_flat[k, i, l] where k in p, i in n, l in r
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # T[i, j, k] = sum_l U_3d[i, j, l] * V_3d[j, k, l] * W_3d[k, i, l]
        # This is a contraction over l.
        # We can do:
        # Temp1[i, j, k] = sum_l U_3d[i, j, l] * W_3d[k, i, l]
        # Then T[i, j, k] = sum_l Temp1[i, j, k] * V_3d[j, k, l] -- No, l is summed in both.
        
        # Correct einsum:
        # 'ijl,jkl,kil->ijk'
        T = np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d)
        
        R = T - 1.0
        loss = np.sum(R**2)
        
        # Gradients:
        # dL/dU_3d[i, j, l] = 2 * sum_{k} R[i, j, k] * V_3d[j, k, l] * W_3d[k, i, l]
        # dL/dV_3d[j, k, l] = 2 * sum_{i} R[i, j, k] * U_3d[i, j, l] * W_3d[k, i, l]
        # dL/dW_3d[k, i, l] = 2 * sum_{j} R[i, j, k] * U_3d[i, j, l] * V_3d[j, k, l]
        
        # Compute gradients using einsum
        # dU: 'ijk,jkl,kil->ijl'
        dU = 2 * np.einsum('ijk,jkl,kil->ijl', R, V_3d, W_3d)
        # dV: 'ijk,ijl,kil->jkl'
        dV = 2 * np.einsum('ijk,ijl,kil->jkl', R, U_3d, W_3d)
        # dW: 'ijk,ijl,jkl->kil'
        dW = 2 * np.einsum('ijk,ijl,jkl->kil', R, U_3d, V_3d)
        
        # Reshape back to 2D
        dU_2d = dU.reshape(n * m, r)
        dV_2d = dV.reshape(m * p, r)
        dW_2d = dW.reshape(p * n, r)
        
        return loss, dU_2d, dV_2d, dW_2d

    def try_rank(r, attempts=20, max_steps=500):
        for attempt in range(attempts):
            if time.time() > deadline:
                return False
            
            # Initialize
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            lr = 0.1
            prev_loss = np.inf
            
            for step in range(max_steps):
                if time.time() > deadline:
                    return False
                
                loss, dU, dV, dW = compute_loss_and_grads(U, V, W, r)
                
                # Check convergence
                if loss < 1e-6:
                    # Round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify
                    U_r_3d = U_rounded.reshape(n, m, r)
                    V_r_3d = V_rounded.reshape(m, p, r)
                    W_r_3d = W_rounded.reshape(p, n, r)
                    T = np.einsum('ijl,jkl,kil->ijk', U_r_3d, V_r_3d, W_r_3d)
                    if np.allclose(T, 1.0, atol=1e-9):
                        return True, U_rounded, V_rounded, W_rounded
                    else:
                        # If rounding failed, continue optimizing with regularization
                        pass

                # Add regularization to push towards half-integers
                # Reg = sum (round(2*x)/2 - x)^2
                # Gradient of Reg:
                # Let f(x) = (round(2x)/2 - x)^2. This is not differentiable at boundaries.
                # Approximate gradient: if x is close to half-integer, pull it there.
                # Simpler: just use the loss gradient for now, and rely on rounding at the end.
                # If we want to encourage half-integers, we can add a penalty.
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if loss < prev_loss:
                    prev_loss = loss
                    lr *= 1.05
                else:
                    lr *= 0.8
                    # Undo step
                    U += lr * dU
                    V += lr * dV
                    W += lr * dW
                    
                if lr < 1e-5:
                    break
        
        return False

    # Try ranks from lower bound to schoolbook
    for r in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break
            
        result = try_rank(r, attempts=10, max_steps=300)
        
        if isinstance(result, tuple) and len(result) == 4:
            _, U_sol, V_sol, W_sol = result
            best_rank = r
            best_U = U_sol
            best_V = V_sol
            best_W = W_sol
            break
            
        # If we fail, we might try harder for this rank or move to next
        # Given time constraints, we move to next rank if simple attempts fail.
        
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return get_schoolbook()
