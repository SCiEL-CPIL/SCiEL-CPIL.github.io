"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # 1. MDS Embedding of locations
    # b is distance matrix. We need squared distances for MDS.
    D = b ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    M = -0.5 * J @ D @ J
    # Symmetrize M to handle numerical errors
    M = (M + M.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Take top d eigenvectors (use full rank d=n for simplicity and accuracy in relaxation)
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure eigenvalues are non-negative (clip small negative values)
    eigenvalues = np.maximum(eigenvalues, 0)
    Y = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    # Gram matrix of locations
    G = Y.T @ Y
    
    # 2. Dynamics Simulation
    # Initialize X (facility coordinates) deterministically
    # Start with X = Y (identity assignment)
    X = Y.copy()
    V = np.zeros_like(X)
    
    # Symplectic Integrator: Velocity Verlet
    # Potential V(X) = -tr(A X G X^T)
    # Gradient = -2 A X G
    # We want to minimize Cost = tr(A P B P^T) approx -tr(A X G X^T)
    # So we follow the gradient of the potential which is the negative of the cost gradient?
    # Cost ~ -tr(A X G X^T).
    # We want to minimize Cost.
    # So we want to increase tr(A X G X^T).
    # Let's define Energy E(X) = -tr(A X G X^T).
    # Force F = -grad E = 2 A X G.
    # Acceleration a = F / m (m=1).
    
    dt = 0.01
    num_steps = 500
    sample_freq = 10
    
    candidates = []
    
    for step in range(num_steps):
        # Compute force at current X
        F = 2 * (a @ X) @ G
        
        # Half step position
        X_half = X + 0.5 * dt * V
        # Project X_half to Stiefel manifold (orthogonalize)
        U, s, Vt = np.linalg.svd(X_half, full_matrices=False)
        X_half_proj = U @ Vt
        
        # Compute force at X_half
        F_half = 2 * (a @ X_half_proj) @ G
        
        # Update velocity
        V = V + 0.5 * dt * (F + F_half)
        
        # Full step position
        X = X_half_proj + 0.5 * dt * V
        # Project X to Stiefel manifold
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X = U @ Vt
        
        # Sample state
        if step % sample_freq == 0:
            # Discretize X to permutation
            # Cost matrix C[i, k] = ||X[i] - Y[k]||^2
            # ||x - y||^2 = x.x + y.y - 2 x.y
            # C = sum(X^2, axis=1)[:, None] + sum(Y^2, axis=1)[None, :] - 2 * X @ Y.T
            C = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * (X @ Y.T)
            row_ind, col_ind = linear_sum_assignment(C)
            p = col_ind.copy()
            # Verify p is a permutation
            # Compute cost
            D_p = b[p[:, None], p[None, :]]
            cost = np.sum(a * D_p)
            candidates.append((cost, p))
    
    # 3. Local Search on best candidates
    # Sort candidates by cost
    candidates.sort(key=lambda x: x[0])
    
    # Take top 5 unique permutations
    best_p = None
    best_cost = float('inf')
    seen = set()
    
    start_time = time.time()
    time_limit = 110
    
    for cost, p in candidates:
        if (time.time() - start_time) > time_limit - 10:
            break
        p_tuple = tuple(p)
        if p_tuple in seen:
            continue
        seen.add(p_tuple)
        
        # Run 2-opt local search
        p = p.copy()
        current_cost = cost
        
        improved = True
        while improved and (time.time() - start_time) < time_limit - 5:
            improved = False
            for i in range(n):
                if (time.time() - start_time) >= time_limit - 5:
                    break
                x = p[i]
                best_j = -1
                best_delta = 0
                for j in range(i + 1, n):
                    y = p[j]
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    pk = p[mask]
                    S1 = np.sum(a[i, mask] * (b[y, pk] - b[x, pk]))
                    S2 = np.sum(a[mask, i] * (b[pk, y] - b[pk, x]))
                    S3 = np.sum(a[j, mask] * (b[x, pk] - b[y, pk]))
                    S4 = np.sum(a[mask, j] * (b[pk, x] - b[pk, y]))
                    S5 = a[i, j] * (b[y, x] - b[x, y])
                    S6 = a[j, i] * (b[x, y] - b[y, x])
                    S_self = a[i, i] * (b[y, y] - b[x, x]) + a[j, j] * (b[x, x] - b[y, y])
                    
                    delta = S1 + S2 + S3 + S4 + S5 + S6 + S_self
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1 and best_delta < 0:
                    p[i], p[best_j] = p[best_j], p[i]
                    current_cost += best_delta
                    improved = True
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p
    
    if best_p is None:
        # Fallback to initial heuristic if dynamics failed
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
