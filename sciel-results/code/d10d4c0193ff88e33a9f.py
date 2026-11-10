import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Energy function
    def energy(X):
        # Pairwise distances
        # X is (N, 3)
        # dist[i, j] = ||X[i] - X[j]||
        # We need sum_{i<j} log(1/d) = -sum_{i<j} log(d)
        # Using broadcasting
        # diff[i, j, k] = X[i, k] - X[j, k]
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid log(0)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        # Sum over i < j
        # Mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy_val = -np.sum(np.log(dist[mask]))
        return energy_val

    # Gradient function
    def gradient(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        # grad_k = - sum_{j != k} (x_k - x_j) / dist_kj^2
        # Note: dist_kj^2 is dist_sq[k, j]
        # We want to sum over j for each k.
        # term[i, j] = (X[i] - X[j]) / dist_sq[i, j]
        # We need to handle division by zero on diagonal, but mask will remove it.
        # Let's compute the matrix of vectors divided by dist_sq
        # diff is (N, N, 3)
        # dist_sq is (N, N)
        # We want to divide each vector in diff[i, j] by dist_sq[i, j]
        inv_dist_sq = 1.0 / dist_sq
        # Set diagonal to 0 to avoid inf/nan
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # grad[i] = - sum_j (X[i] - X[j]) * inv_dist_sq[i, j]
        # This is a matrix multiplication if we view diff as a tensor
        # grad = - np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        return grad

    # Optimization loop
    eta = 0.01
    decay = 0.9999
    best_energy = energy(X)
    best_X = X.copy()
    
    for t in range(100000):
        if time.time() > deadline:
            break
        
        g = gradient(X)
        
        # Update
        X = X - eta * g
        
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms
        
        # Compute energy
        e = energy(X)
        
        if e < best_energy:
            best_energy = e
            best_X = X.copy()
            
        # Decay learning rate
        eta *= decay
        
        # Convergence check
        if t > 100 and abs(best_energy - e) < 1e-10:
             # We might break early if stuck, but let's continue for a bit to be sure
             pass

    return best_X
