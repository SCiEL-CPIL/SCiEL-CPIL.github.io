import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p  # 45
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    target = np.ones((n, m, p), dtype=np.float64)

    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, target, atol=1e-9, rtol=1e-9)

    def try_snap_and_verify(U, V, W):
        # Try snapping to nearest integer
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i):
            return U_i, V_i, W_i
        
        # Try snapping to nearest half-integer
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        if verify(U_h, V_h, W_h):
            return U_h, V_h, W_h
            
        return None

    def solve_rank_adam(R, max_iters=3000, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        # Initialize with random normal
        U = np.random.randn(size_u, R) * 0.5
        V = np.random.randn(size_v, R) * 0.5
        W = np.random.randn(size_w, R) * 0.5
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        t = 0
        best_error = np.inf
        
        for i in range(max_iters):
            if time.time() > deadline:
                return None
            
            t += 1
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            diff = result - target
            error = np.sum(diff**2)
            
            if error < best_error:
                best_error = error
                if error < 1e-10:
                    snapped = try_snap_and_verify(U, V, W)
                    if snapped:
                        return snapped
            
            # Compute gradients
            grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
            grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
            grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
            
            # Adam update
            # U
            mU = beta1 * mU + (1 - beta1) * grad_U
            vU = beta2 * vU + (1 - beta2) * (grad_U ** 2)
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            # V
            mV = beta1 * mV + (1 - beta1) * grad_V
            vV = beta2 * vV + (1 - beta2) * (grad_V ** 2)
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            # W
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * (grad_W ** 2)
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            # Periodic check for snapping
            if i % 50 == 0 and error < 1e-4:
                snapped = try_snap_and_verify(U, V, W)
                if snapped:
                    return snapped
        
        return None

    def solve_rank(R, max_restarts=30):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            result = solve_rank_adam(R)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        return None

    # Try to find lower ranks
    # Based on R(3,3,2)=13 and R(3,3,3)=23, R(3,3,5) <= 36.
    # Try 36 first, then 37, 38...
    for R in range(36, 45):
        if time.time() > deadline - 10:
            break
        
        result = solve_rank(R, max_restarts=30)
        if result is not None:
            return result

    # Return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
