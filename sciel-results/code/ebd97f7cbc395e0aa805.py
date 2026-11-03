"""Minimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import time

N = 100

def _log_energy(X):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    import math
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-15:
                return np.inf
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _gradient(X):
    """Compute gradient of log energy w.r.t. each point (projected to sphere)."""
    grad = np.zeros_like(X)
    for i in range(N):
        for j in range(i + 1, N):
            diff = X[i] - X[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-30:
                continue
            d = np.sqrt(d2)
            # d/dx_i log(1/|x_i-x_j|) = -(x_i - x_j)/|x_i-x_j|^2
            force = -diff / d2
            grad[i] += force
            grad[j] -= force
    # Project to tangent space: remove radial component
    for i in range(N):
        r = np.dot(X[i], grad[i])
        grad[i] -= r * X[i]
    return grad

def _normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return X / norms

def construct():
    deadline = time.time() + 130.0  # 10s safety margin before 140s kill
    
    # Initial: golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X.copy()
    best_E = _log_energy(best_X)
    
    # Adaptive gradient descent with backtracking
    lr = 0.01
    min_lr = 1e-8
    max_lr = 0.5
    max_backtracks = 20
    
    while time.time() < deadline:
        # Check deadline more frequently in later stages
        if time.time() > deadline - 5.0:
            break
        
        grad = _gradient(X)
        gnorm = np.linalg.norm(grad)
        
        if gnorm < 1e-12:
            break
        
        # Try step with backtracking
        step = lr
        accepted = False
        for _ in range(max_backtracks):
            X_new = _normalize(X - step * grad)
            E_new = _log_energy(X_new)
            if E_new < best_E:
                accepted = True
                break
            step *= 0.5
            if step < min_lr:
                break
        
        if not accepted:
            # If no step improves, try smaller base lr
            lr *= 0.5
            if lr < min_lr:
                break
            continue
        
        X = X_new
        best_X = X.copy()
        best_E = E_new
        
        # Increase lr if we used full step
        if step >= lr * 0.99:
            lr = min(lr * 1.2, max_lr)
    
    # Final: ensure valid output
    best_X = _normalize(best_X)
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_X[i] - best_X[j]) < 1e-9:
                # Nudge apart
                best_X[j] += 1e-8 * np.random.randn(3)
                best_X = _normalize(best_X)
    
    return best_X
