import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 140.0
    
    # Try ranks from 20 upwards
    for r in range(20, 30):
        if time.time() - start_time > max_time:
            break
            
        found = False
        # Increase attempts due to speedup
        for attempt in range(200):
            if time.time() - start_time > max_time:
                break
                
            # Initialize U, V, W
            # Shapes: U (n, m, r), V (m, p, r), W (p, n, r)
            # Note: The problem defines U as (n*m, r), V as (m*p, r), W as (p*n, r).
            # We will use 3D tensors for calculation and reshape for output.
            
            U = np.random.randn(n, m, r)
            V = np.random.randn(m, p, r)
            W = np.random.randn(p, n, r)
            
            # ALS Iterations
            for it in range(500): # More iterations possible now
                if time.time() - start_time > max_time:
                    break
                    
                # Update W
                # A_{i, k, c} = sum_j U[i, j, c] * V[j, k, c]
                A = np.einsum('ijc,jkc->ikc', U, V)
                # Norm squared along c
                norm_sq_A = np.sum(A * A, axis=2, keepdims=True)
                # Avoid division by zero
                norm_sq_A[norm_sq_A < 1e-9] = 1e-9
                W = A / norm_sq_A
                
                # Update U
                # B_{i, j, c} = sum_k V[j, k, c] * W[k, i, c]
                # W is (p, n, r). We need W[k, i, c].
                # Swap axes of W to get (n, p, r) then transpose?
                # W[k, i, c] -> W.T (n, p, r) is W[i, k, c]? No.
                # W shape (p, n, r).
                # We want index k (0..p-1) as first dim of sum, i (0..n-1) as second?
                # B_{i, j, c} = sum_k V[j, k, c] * W[k, i, c]
                # V[j, k, c]
                # W[k, i, c]
                # Result B[i, j, c]
                # Einsum: 'jkc,kic->ijc'
                B = np.einsum('jkc,kic->ijc', V, W)
                norm_sq_B = np.sum(B * B, axis=2, keepdims=True)
                norm_sq_B[norm_sq_B < 1e-9] = 1e-9
                U = B / norm_sq_B
                
                # Update V
                # C_{j, k, c} = sum_i U[i, j, c] * W[k, i, c]
                # Einsum: 'ijc,kic->jkc'
                C = np.einsum('ijc,kic->jkc', U, W)
                norm_sq_C = np.sum(C * C, axis=2, keepdims=True)
                norm_sq_C[norm_sq_C < 1e-9] = 1e-9
                V = C / norm_sq_C
                
            # Check for exact solution
            # Reshape to 2D for output format check
            U_2d = U.reshape(n*m, r)
            V_2d = V.reshape(m*p, r)
            W_2d = W.reshape(p*n, r)
            
            U_rounded = np.round(U_2d * 2) / 2
            V_rounded = np.round(V_2d * 2) / 2
            W_rounded = np.round(W_2d * 2) / 2
            
            # Verify
            valid = True
            # Use tensor contraction for speed
            # T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # U (n, m, r), V (m, p, r), W (p, n, r)
            # Reshape rounded to 3D
            Ur = U_rounded.reshape(n, m, r)
            Vr = V_rounded.reshape(m, p, r)
            Wr = W_rounded.reshape(p, n, r)
            
            # Calculate T
            # T_{i,j,k} = sum_c Ur[i,j,c] Vr[j,k,c] Wr[k,i,c]
            # Einsum: 'ijc,jkc,kic->ijk'
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            
            if np.allclose(T, 1.0, atol=1e-9):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                
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
