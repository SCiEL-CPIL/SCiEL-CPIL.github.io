"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Try to find a rank-24 decomposition via optimization
    # Fallback to rank-28 Strassen if not found in time
    
    # First, let's try a structured rank-24 approach
    # The key idea: use 6 terms per column-pair with cross-block sharing
    
    # For a 2x2x2 block, Strassen uses 7 terms.
    # If we allow the U-factors to be shared (i.e., the same linear combination
    # of A-entries is used for adjacent blocks), we can potentially save terms.
    
    # Let's try: for blocks 0,1,2,3, use terms where:
    # - Block q uses terms 6q..6q+5 (6 terms each)
    # - But some terms span two blocks in their V or W factors
    
    # Actually, let me try a direct numerical search for rank 24
    rank = 24
    dim_u = n * m  # 4
    dim_v = m * p  # 16
    dim_w = p * n  # 16
    
    # Initialize with random values
    np.random.seed(42)
    
    best_score = None
    best_result = None
    
    # Try multiple random initializations
    max_iters = 5000
    lr = 0.01
    
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
            
        # Random initialization
        U = np.random.randn(dim_u, rank) * 0.5
        V = np.random.randn(dim_v, rank) * 0.5
        W = np.random.randn(dim_w, rank) * 0.5
        
        # The target tensor T[i,j,k] where:
        # i = row index of A (0,1)
        # j = col index of A / row index of B (0,1)  
        # k = col index of B (0..7)
        # T[i,j,k] = 1 if this is a valid matmul entry, else 0
        # Specifically: C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor has T[i,j,k] = 1 for all valid (i,j,k)
        
        # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
        # Wait, let me re-read the convention.
        
        # U has shape (n*m, rank) = (4, rank)
        # V has shape (m*p, rank) = (16, rank)
        # W has shape (p*n, rank) = (16, rank)
        
        # The decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is a vector in R^(n*m), V_r in R^(m*p), W_r in R^(p*n)
        
        # The tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a indexes A entries (i*m+j), b indexes B entries (j*p+k), c indexes C entries (k*n+i)
        
        # For matrix multiplication C = A*B:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[A_idx, B_idx, C_idx] = 1 iff A_idx=(i,j), B_idx=(j,k), C_idx=(i,k)
        
        # Let's define the target tensor
        T = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i  # trace convention: W[k*n+i]
                    T[a_idx, b_idx, c_idx] = 1.0
        
        # Optimize using gradient descent on the non-convex objective
        # Objective: ||sum_r U_r ⊗ V_r ⊗ W_r - T||^2
        
        def compute_loss_and_grad(U, V, W, T):
            # Compute the tensor from current U, V, W
            # T_current[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Use einsum for efficiency
            T_current = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_current - T
            loss = np.sum(diff**2)
            
            # Gradients:
            # dL/dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
            
            dU = np.einsum('abc,br,cr->ar', diff, V, W)
            dV = np.einsum('abc,ar,cr->br', diff, U, W)
            dW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            return loss, dU, dV, dW
        
        # Adam optimizer
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
                
            t += 1
            loss, dU, dV, dW = compute_loss_and_grad(U, V, W, T)
            
            if loss < 1e-10:
                # Found exact solution!
                # Convert to list format
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify
                T_check = np.einsum('ar,br,cr->abc', np.array(U_list), np.array(V_list), np.array(W_list))
                if np.allclose(T_check, T, atol=1e-9):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                best_score = loss
                best_result = (U_list, V_list, W_list)
                break
            
            # Adam updates
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU**2
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV**2
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW**2
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if it % 500 == 0:
                # Check convergence
                pass
        
        # Check final loss
        loss, _, _, _ = compute_loss_and_grad(U, V, W, T)
        if loss < 1e-8:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            T_check = np.einsum('ar,br,cr->abc', np.array(U_list), np.array(V_list), np.array(W_list))
            if np.allclose(T_check, T, atol=1e-8):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        if best_score is None or loss < best_score:
            best_score = loss
            best_result = (U.tolist(), V.tolist(), W.tolist())
    
    # Fallback to rank-28 Strassen
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: 1}
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
