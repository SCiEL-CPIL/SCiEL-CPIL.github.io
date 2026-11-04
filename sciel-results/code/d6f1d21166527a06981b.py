import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad(x_flat):
    """
    Calculate the logarithmic energy and its gradient.
    x_flat is a flattened array of shape (N*3,).
    Returns (energy, gradient_flat).
    """
    X = x_flat.reshape(N, 3)
    # Normalize to ensure points are on the sphere
    norms = np.linalg.norm(X, axis=1)
    # Handle zero norms
    norms[norms < 1e-12] = 1e-12
    P = X / norms[:, np.newaxis]
    
    # Pairwise distances
    # d2_ij = 2 - 2 * P_i . P_j
    dot_prod = P @ P.T
    d2 = 2.0 - 2.0 * dot_prod
    np.fill_diagonal(d2, 1.0) # Avoid self interaction
    d2 = np.clip(d2, 1e-12, None)
    d = np.sqrt(d2)
    
    # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # Sum over upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    log_d = np.log(d)
    energy = -np.sum(log_d[mask])
    
    # Gradient
    # grad_E_k = - sum_{j != k} (P_k - P_j) / d_kj^2
    # diff[k, j] = P_k - P_j
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    
    # d2_inv = 1 / d2, with diagonal 0
    d2_inv = np.zeros_like(d2)
    mask_non_diag = ~np.eye(N, dtype=bool)
    d2_inv[mask_non_diag] = 1.0 / d2[mask_non_diag]
    
    # weighted_diff[k, j] = diff[k, j] * d2_inv[k, j]
    weighted_diff = diff * d2_inv[:, :, np.newaxis]
    
    # grad_P[k] = - sum_j weighted_diff[k, j]
    grad_P = -np.sum(weighted_diff, axis=1)
    
    # Project gradient to tangent space
    # grad_X = grad_P - (grad_P . P) P
    dot_grad_P = np.sum(grad_P * P, axis=1)
    grad_X = grad_P - dot_grad_P[:, np.newaxis] * P
    
    return energy, grad_X.flatten()

def _local_minimize(P, maxiter=100, gtol=1e-8):
    """
    Run L-BFGS-B to find a local minimum.
    """
    x0 = P.flatten()
    try:
        res = minimize(
            _energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        # Ensure points are on sphere
        X = res.x.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1)
        norms[norms < 1e-12] = 1e-12
        X = X / norms[:, np.newaxis]
        return X, res.fun
    except Exception:
        return P, float('inf')

def _fibonacci_sphere(n):
    """Generate n points on a sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    v = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1)
    return v / norms[:, np.newaxis]

def _perturb(P, strength=0.1):
    """
    Perturb the points by adding Gaussian noise and projecting back to sphere.
    """
    noise = np.random.normal(0, strength, size=P.shape)
    P_new = P + noise
    norms = np.linalg.norm(P_new, axis=1)
    norms[norms < 1e-12] = 1e-12
    P_new = P_new / norms[:, np.newaxis]
    return P_new

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initializations
    inits = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
        _random_sphere(N, seed=4),
        _random_sphere(N, seed=5)
    ]
    
    best_P = None
    best_E = float('inf')
    
    # Phase 1: Minimize initializations
    for i, init_P in enumerate(inits):
        if time.time() > deadline - 40: break
        P_min, E_min = _local_minimize(init_P, maxiter=100)
        if E_min < best_E:
            best_E = E_min
            best_P = P_min.copy()
            
    # Initialize current state with the best found so far
    current_P = best_P.copy()
    current_E = best_E
    
    # Phase 2: Basin Hopping
    T = 1.0
    T_min = 1e-6
    cooling = 0.98
    no_improve_count = 0
    max_no_improve = 30 # Restart if stuck
    
    # We will use a fixed perturbation strength for simplicity, 
    # or slightly adapt it.
    perturb_strength = 0.05
    
    while time.time() < deadline - 20.0:
        # Perturb
        P_trial = _perturb(current_P, strength=perturb_strength)
        
        # Minimize the perturbed state
        # Use fewer iterations for the inner loop to allow more global steps
        P_min, E_min = _local_minimize(P_trial, maxiter=30, gtol=1e-6)
        
        delta_E = E_min - current_E
        
        # Metropolis criterion
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            current_P = P_min
            current_E = E_min
            
            if current_E < best_E:
                best_E = current_E
                best_P = current_P.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        T *= cooling
        if T < T_min:
            T = T_min
            
        # If stuck, restart from a different initialization
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Try a random restart
            if time.time() < deadline - 20.0:
                P_restart = _random_sphere(N)
                P_min, E_min = _local_minimize(P_restart, maxiter=100)
                if E_min < best_E:
                    best_E = E_min
                    best_P = P_min.copy()
                current_P = P_min
                current_E = E_min
                T = 1.0 # Reset temperature

    # Phase 3: Final Polish
    if best_P is not None:
        # High precision polish
        if time.time() < deadline - 5.0:
            P_final, E_final = _local_minimize(best_P, maxiter=1000, gtol=1e-12)
            if E_final < best_E:
                best_E = E_final
                best_P = P_final
        
        # One more polish if time permits
        if time.time() < deadline - 1.0:
            P_final2, E_final2 = _local_minimize(best_P, maxiter=1000, gtol=1e-14)
            if E_final2 < best_E:
                best_E = E_final2
                best_P = P_final2

    # If best_P is still None (should not happen), use Fibonacci
    if best_P is None:
        best_P = _fibonacci_sphere(N)
        
    # Normalize final result
    norms = np.linalg.norm(best_P, axis=1)
    norms[norms < 1e-12] = 1e-12
    best_P = best_P / norms[:, np.newaxis]
    
    return best_P
