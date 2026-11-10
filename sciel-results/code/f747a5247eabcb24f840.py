import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X.copy()
    best_E = np.inf
    
    def energy(X):
        # Compute pairwise distances
        # X is N x 3
        # D2[i,j] = ||X[i] - X[j]||^2
        # D2 = sum(X^2) + sum(X^2) - 2 * X X^T
        XX = X @ X.T
        D2 = 2.0 - 2.0 * XX
        np.fill_diagonal(D2, 1.0) # Avoid division by zero
        D = np.sqrt(D2)
        # Energy is sum_{i<j} log(1/D) = -sum_{i<j} log(D)
        # Use upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        E = -np.sum(np.log(D[mask]))
        return E
    
    def gradient(X):
        XX = X @ X.T
        D2 = 2.0 - 2.0 * XX
        np.fill_diagonal(D2, 1.0)
        # Force on i from j is (X[i] - X[j]) / D2[i,j]
        # Sum over j
        # G[i] = sum_j (X[i] - X[j]) / D2[i,j]
        # G = X * sum(1/D2, axis=1) - sum(X / D2, axis=1)
        inv_D2 = 1.0 / D2
        np.fill_diagonal(inv_D2, 0.0)
        sum_inv = np.sum(inv_D2, axis=1)
        # X * sum_inv is N x 3
        term1 = X * sum_inv[:, np.newaxis]
        # sum(X / D2, axis=1) is N x 3
        # X is N x 3, inv_D2 is N x N
        # We want sum_j X[j] * inv_D2[i,j]
        # This is inv_D2 @ X
        term2 = inv_D2 @ X
        G = term1 - term2
        return G
    
    def project_tangent(G, X):
        # Project gradient G onto tangent space of sphere at X
        # v = G - (G . X) X
        dot = np.sum(G * X, axis=1)
        return G - dot[:, np.newaxis] * X
    
    # Gradient descent
    step_size = 0.01
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        G = gradient(X)
        G_tan = project_tangent(G, X)
        
        # Adaptive step size: try to reduce energy
        # Simple backtracking
        alpha = step_size
        E_curr = energy(X)
        
        # Update
        X_new = X + alpha * G_tan
        # Normalize
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        E_new = energy(X_new)
        
        if E_new < E_curr:
            X = X_new
            step_size = min(step_size * 1.05, 0.1) # Increase step if successful
            if E_new < best_E:
                best_E = E_new
                best_X = X.copy()
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        
    # Try a few random restarts if time permits
    while time.time() < deadline - 5.0:
        # Random start
        X_rand = np.random.randn(N, 3)
        X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
        
        # Short gradient descent
        step_size = 0.01
        for _ in range(500):
            if time.time() > deadline:
                break
            G = gradient(X_rand)
            G_tan = project_tangent(G, X_rand)
            
            alpha = step_size
            E_curr = energy(X_rand)
            
            X_new = X_rand + alpha * G_tan
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            E_new = energy(X_new)
            
            if E_new < E_curr:
                X_rand = X_new
                step_size = min(step_size * 1.05, 0.1)
                if E_new < best_E:
                    best_E = E_new
                    best_X = X_rand.copy()
            else:
                step_size *= 0.5
                if step_size < 1e-8:
                    break

    return best_X
