import numpy as np
import random
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Generate the tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try ranks from rank_school-1 down to some limit
    for rank in range(rank_school - 1, 38, -1):
        # Try a few random initializations
        for trial in range(10):
            # Initialize U, V, W randomly
            U = np.random.rand(n*m, rank)
            V = np.random.rand(m*p, rank)
            W = np.random.rand(p*n, rank)
            
            # ALS iterations
            for it in range(50):
                # Fix U, V, solve for W
                # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
                # For each c, we have T_{a,b,c} = sum_r (U_{a,r} V_{b,r}) W_{c,r}
                # Let M_{a,b} = T_{a,b,:} (shape n*m x m*p x p*n)
                # This is not straightforward.
                
                # Reshape T to (n*m, m*p, p*n)
                # We want to solve for W (p*n x rank)
                # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
                # Let A_{a,b,r} = U_{a,r} V_{b,r}
                # Then T_{a,b,c} = sum_r A_{a,b,r} W_{c,r}
                # This is a linear system for W.
                # We can vectorize:
                # T_vec (n*m * m*p x p*n) = A_vec (n*m * m*p x rank) @ W^T (rank x p*n)
                
                T_2d = T.reshape(n*m * m*p, p*n)
                A_2d = (U[:, :, None] * V[None, :, :]).reshape(n*m * m*p, rank)
                # Solve A_2d @ W^T = T_2d
                # W^T = A_2d^+ @ T_2d
                W_new = np.linalg.lstsq(A_2d, T_2d, rcond=None)[0].T
                
                # Fix V, W, solve for U
                # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
                # Let B_{b,c,r} = V_{b,r} W_{c,r}
                # T_{a,b,c} = sum_r U_{a,r} B_{b,c,r}
                # T_2d (n*m x m*p * p*n) = U (n*m x rank) @ B_2d (rank x m*p * p*n)
                T_2d = T.reshape(n*m, m*p * p*n)
                B_2d = (V[:, :, None] * W[None, :, :]).reshape(rank, m*p * p*n)
                U_new = np.linalg.lstsq(B_2d.T, T_2d.T, rcond=None)[0].T
                
                # Fix U, W, solve for V
                # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
                # Let C_{a,c,r} = U_{a,r} W_{c,r}
                # T_{a,b,c} = sum_r C_{a,c,r} V_{b,r}
                # T_2d (m*p x n*m * p*n) = V (m*p x rank) @ C_2d (rank x n*m * p*n)
                T_2d = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
                C_2d = (U[:, :, None] * W[None, :, :]).reshape(rank, n*m * p*n)
                V_new = np.linalg.lstsq(C_2d.T, T_2d.T, rcond=None)[0].T
                
                U, V, W = U_new, V_new, W_new
                
                # Check error
                error = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
                if error < 1e-6:
                    # Found a solution
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    # Fallback to schoolbook
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}
