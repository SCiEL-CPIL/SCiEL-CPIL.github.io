import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140
    
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def check_solution(U, V, W):
        for i, j, k in indices:
            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            if abs(val - 1) > 1e-6:
                return False
        return True
    
    def round_to_half(x):
        return np.round(2*x) / 2
    
    # Try ranks from 53 down
    for rank in range(53, 45, -1):
        if time.time() > deadline:
            break
        for attempt in range(5):
            if time.time() > deadline:
                break
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Gradient descent
            err = float('inf')
            for step in range(30):
                if time.time() > deadline:
                    break
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                for i, j, k in indices:
                    u = U[i*m+j, :]
                    v = V[j*p+k, :]
                    w = W[k*n+i, :]
                    val = np.dot(u, v * w)
                    d = val - 1
                    err = d*d  # Just use the last error for now
                    grad_U[i*m+j, :] += 2*d*v*w
                    grad_V[j*p+k, :] += 2*d*u*w
                    grad_W[k*n+i, :] += 2*d*u*v
                
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            if check_solution(U_r, V_r, W_r):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Also check the unrounded solution
            if check_solution(U, V, W):
                # Try to find a nearby half-integer solution
                for scale in [0.5, 1.0, 2.0]:
                    U_s = U * scale
                    V_s = V * scale
                    W_s = W / scale
                    U_r = round_to_half(U_s)
                    V_r = round_to_half(V_s)
                    W_r = round_to_half(W_s)
                    if check_solution(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fall back to schoolbook
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
