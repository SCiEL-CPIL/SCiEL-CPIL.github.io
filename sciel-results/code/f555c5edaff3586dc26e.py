import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    T = np.ones((n, m, p))
    
    # Try to find a lower-rank decomposition
    for target_rank in range(n * m * p - 1, 30, -1):
        if time.time() > deadline - 20:
            break
        
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(seed * 7 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                pred = np.zeros((n, m, p))
                for r in range(target_rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    pred += u[:, :, None] * v[None, :, :] * w.T[None, None, :]
                
                err = T - pred
                loss = np.sum(err ** 2)
                if loss < 1e-8:
                    # Try rounding
                    Ur = np.round(U * 2) / 2
                    Vr = np.round(V * 2) / 2
                    Wr = np.round(W * 2) / 2
                    pred_r = np.zeros((n, m, p))
                    for r in range(target_rank):
                        u = Ur[:, r].reshape(n, m)
                        v = Vr[:, r].reshape(m, p)
                        w = Wr[:, r].reshape(p, n)
                        pred_r += u[:, :, None] * v[None, :, :] * w.T[None, None, :]
                    if np.allclose(pred_r, T, atol=1e-10):
                        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    break
                
                lr = 0.001
                for r in range(target_rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    grad_u = 2 * np.einsum('ijk,jk,ki->ij', err, v, w)
                    grad_v = 2 * np.einsum('ijk,ij,ki->jk', err, u, w)
                    grad_w = 2 * np.einsum('ijk,ij,jk->ki', err, u, v)
                    U[:, r] -= lr * grad_u.ravel()
                    V[:, r] -= lr * grad_v.ravel()
                    W[:, r] -= lr * grad_w.ravel()
    
    # Schoolbook fallback
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
