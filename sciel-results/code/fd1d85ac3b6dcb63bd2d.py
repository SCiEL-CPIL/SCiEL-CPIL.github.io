import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _error(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err += (T[a, b, c] - 1) ** 2
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                i2 = c % n
                k2 = c // n
                if not (i == i2 and j == j2 and k == k2):
                    err += T[a, b, c] ** 2
    return err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try random search for lower rank
    for r in range(40, 54):
        if time.time() > deadline - 10:
            break
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            U_r = np.random.randn(n * m, r)
            V_r = np.random.randn(m * p, r)
            W_r = np.random.randn(p * n, r)
            # Scale to get reasonable magnitudes
            U_r /= np.linalg.norm(U_r, axis=0)
            V_r /= np.linalg.norm(V_r, axis=0)
            W_r /= np.linalg.norm(W_r, axis=0)
            
            # Simple local search
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                err = _error(U_r, V_r, W_r, n, m, p)
                if err < 1e-8:
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
                # Gradient descent (approximate)
                # This is a placeholder; in reality, we'd compute gradients
                U_r += lr * np.random.randn(*U_r.shape)
                V_r += lr * np.random.randn(*V_r.shape)
                W_r += lr * np.random.randn(*W_r.shape)
    
    if best_rank < rank:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    else:
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
