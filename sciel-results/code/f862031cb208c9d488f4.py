import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    def energy(X):
        X2 = np.sum(X**2, axis=1)
        dist2 = X2[:, None] + X2[None, :] - 2 * (X @ X.T)
        np.fill_diagonal(dist2, 1.0)
        dist2 = np.maximum(dist2, 1e-12)
        dist = np.sqrt(dist2)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        E = -np.sum(np.log(dist[mask]))
        return E

    def gradient(X):
        X2 = np.sum(X**2, axis=1)
        dist2 = X2[:, None] + X2[None, :] - 2 * (X @ X.T)
        np.fill_diagonal(dist2, 1.0)
        dist2 = np.maximum(dist2, 1e-12)
        
        inv_dist2 = 1.0 / dist2
        np.fill_diagonal(inv_dist2, 0.0)
        
        sum_inv = np.sum(inv_dist2, axis=1)
        sum_x_inv = X.T @ inv_dist2
        
        grad = sum_x_inv - X * sum_inv[:, None]
        
        dot = np.sum(grad * X, axis=1)
        grad_tan = grad - dot[:, None] * X
        return grad_tan

    best_X = X.copy()
    best_E = energy(X)
    
    eta = 0.1
    prev_grad = None
    prev_X = None
    
    max_iter = 5000
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        E_curr = energy(X)
        if E_curr < best_E:
            best_E = E_curr
            best_X = X.copy()
            
        g = gradient(X)
        
        if prev_grad is not None and prev_X is not None:
            s = X - prev_X
            y = g - prev_grad
            ss = np.sum(s**2)
            sy = np.sum(s * y)
            if sy > 1e-12:
                eta_bb = ss / sy
                eta = np.clip(eta_bb, 1e-6, 1.0)
            else:
                eta *= 0.9
        else:
            eta = 0.1
            
        X_new = X - eta * g
        X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
        
        if not np.all(np.isfinite(X_new)):
            break
            
        E_new = energy(X_new)
        if E_new > E_curr + 1e-6:
            eta *= 0.5
            X_new = X - eta * g
            X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
            E_new = energy(X_new)
            if E_new > E_curr + 1e-6:
                continue
                
        prev_grad = g
        prev_X = X
        X = X_new
        
        if k > 100 and abs(E_new - E_curr) < 1e-10:
            break

    # Final validity check
    # Check min distance
    X2 = np.sum(best_X**2, axis=1)
    dist2 = X2[:, None] + X2[None, :] - 2 * (best_X @ best_X.T)
    np.fill_diagonal(dist2, 1.0)
    min_dist = np.sqrt(np.min(dist2))
    
    if min_dist < 1e-9:
        # Fallback to initial if something went wrong
        return X
        
    return best_X
