import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if (i,j) in A and (j,k) in B?
    # T is defined by the multiplication: C_ik = sum_j A_ij B_jk
    # The tensor T has indices i (0..n-1), j (0..m-1), k (0..p-1)
    # T[i, j, k] = 1
    
    # Flattened indices for U, V, W
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # Construct target tensor in flattened form for checking
    # T_flat[i*m+j, j*p+k, k*n+i] = 1
    
    def check_exact(U, V, W):
        R = U.shape[1]
        # Reconstruct tensor
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need this to be 1 for all i,j,k
        
        # Efficient check using einsum or loops
        # Given small dims, loops are fine
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for c in range(R):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def round_to_half(x):
        return np.round(x * 2) / 2

    # Try ranks from 32 to 36
    # We prioritize lower ranks
    for R in range(32, 37):
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Simple ALS
        # Minimize || T - sum U_c V_c W_c ||_F
        # For fixed V, W, update U:
        # U[:, c] = (T * V_c * W_c) / (||V_c * W_c||^2)
        
        # Precompute slices?
        # T is sparse (all 1s at valid indices).
        
        # We'll do a few iterations of ALS
        # But ALS might get stuck in local minima or converge slowly.
        # Given the time limit, we can try a few random restarts for R=32, 33, 34.
        
        found = False
        for restart in range(5):
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Scale to avoid overflow/underflow
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-9
            V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-9
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
            
            for it in range(20):
                # Update U
                for c in range(R):
                    # Term for U[:, c]
                    # Sum over j, k of T[i,j,k] V[j*p+k, c] W[k*n+i, c]
                    # This is a linear system for U[:, c]
                    # Actually, since T is 1, it's just summing V and W products weighted by T.
                    
                    # Vectorized update might be faster
                    # But for small dims, loops are okay.
                    
                    # Let's just do gradient descent step?
                    # Or simple projection.
                    
                    # ALS step for U[:, c]:
                    # U[:, c] = (sum_{j,k} T[i,j,k] V[j*p+k, c] W[k*n+i, c]) / (sum_{j,k} V[j*p+k, c]^2 W[k*n+i, c]^2)
                    
                    num = np.zeros(n*m)
                    den = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                v_val = V[j*p+k, c]
                                w_val = W[k*n+i, c]
                                # T[i,j,k] is 1
                                num[i*m+j] += v_val * w_val
                                den += v_val * v_val * w_val * w_val
                    if den > 1e-9:
                        U[:, c] = num / den
                    else:
                        U[:, c] = 0.0
                
                # Update V
                for c in range(R):
                    num = np.zeros(m*p)
                    den = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_val = U[i*m+j, c]
                                w_val = W[k*n+i, c]
                                num[j*p+k] += u_val * w_val
                                den += u_val * u_val * w_val * w_val
                    if den > 1e-9:
                        V[:, c] = num / den
                    else:
                        V[:, c] = 0.0
                
                # Update W
                for c in range(R):
                    num = np.zeros(p*n)
                    den = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_val = U[i*m+j, c]
                                v_val = V[j*p+k, c]
                                num[k*n+i] += u_val * v_val
                                den += u_val * u_val * v_val * v_val
                    if den > 1e-9:
                        W[:, c] = num / den
                    else:
                        W[:, c] = 0.0
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0)**2
                if err < 1e-6:
                    # Round and check
                    Ur = round_to_half(U)
                    Vr = round_to_half(V)
                    Wr = round_to_half(W)
                    if check_exact(Ur, Vr, Wr):
                        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    # If rounding failed, try without rounding? 
                    # The task requires integer/half-integer.
                    # If numerical solution is close to half-integers, rounding should work.
                    
        # If not found for this R, try next R

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
