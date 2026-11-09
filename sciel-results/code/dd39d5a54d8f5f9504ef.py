import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Target Tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def compute_residual(U, V, W, R):
        S = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return S - T

    def compute_error(U, V, W, R):
        R_res = compute_residual(U, V, W, R)
        return np.sum(R_res**2)

    def snap_to_half_int(M):
        return np.round(M * 2) / 2

    def verify(U, V, W, R):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        # Check exact decomposition
        S = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=1e-9)

    def to_list(U, V, W, R):
        return {
            "U": [[float(U[i, c]) for c in range(R)] for i in range(n * m)],
            "V": [[float(V[j, c]) for c in range(R)] for j in range(m * p)],
            "W": [[float(W[k, c]) for c in range(R)] for k in range(p * n)]
        }

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return to_list(U, V, W, rank)

    best_result = schoolbook()
    best_rank = n * m * p

    # Try to find better rank
    # We try ranks from 44 down to 35
    for R in range(44, 34, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Multiple random restarts
        for attempt in range(20): # Increased restarts
            if time.time() > deadline - 2:
                break
                
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Alternating Least Squares (ALS) is often more stable for tensor decomposition
            # Minimize ||T - sum_r u_r v_r w_r||^2
            # Fix V, W -> solve for U
            # U[:, r] = (V[:, r] * W[:, r] flattened) projected? No.
            # Let's stick to gradient but with momentum or better step size.
            
            # Use a simple gradient descent with adaptive learning rate
            lr = 0.05
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for step in range(2000):
                if time.time() > deadline - 2:
                    break
                    
                # Compute gradients
                S = np.zeros((n*m, m*p, p*n))
                for r in range(R):
                    S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res = S - T
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(R):
                    grad_U[:, r] = np.einsum('abc,b,c->a', Res, V[:, r], W[:, r])
                    grad_V[:, r] = np.einsum('abc,a,c->b', Res, U[:, r], W[:, r])
                    grad_W[:, r] = np.einsum('abc,a,b->c', Res, U[:, r], V[:, r])
                
                # Update with momentum
                vU = momentum * vU - lr * grad_U
                vV = momentum * vV - lr * grad_V
                vW = momentum * vW - lr * grad_W
                
                U += vU
                V += vV
                W += vW
                
                err = np.sum(Res**2)
                
                if err < 1e-10:
                    break
                
                # Adaptive LR
                if step % 100 == 0:
                    if err < 1e-4:
                        lr = 0.005
                    elif err < 1e-2:
                        lr = 0.02
                    else:
                        lr = 0.05

            err = compute_error(U, V, W, R)
            if err < 1e-6:
                # Snap to half integers
                U_s = snap_to_half_int(U)
                V_s = snap_to_half_int(V)
                W_s = snap_to_half_int(W)
                
                if verify(U_s, V_s, W_s, R):
                    best_result = to_list(U_s, V_s, W_s, R)
                    best_rank = R
                    found = True
                    break
        if found:
            break

    return best_result
