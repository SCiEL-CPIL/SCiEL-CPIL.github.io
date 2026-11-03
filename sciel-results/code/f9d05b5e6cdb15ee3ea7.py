import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _fibonacci_sphere(n):
    """Generate n points on a sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(x_flat):
    """
    Calculate the logarithmic energy and its gradient.
    x_flat is a flattened array of shape (N*3,).
    Returns (energy, gradient_flat).
    """
    X = x_flat.reshape(N, 3)
    # Normalize to ensure points are on the sphere
    norms = np.linalg.norm(X, axis=1)
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

def _energy_only(x_flat):
    """Calculate only the energy, for SA."""
    E, _ = _energy_and_grad(x_flat)
    return E

def _sa_step(P, T, step_size):
    """
    Perform one step of simulated annealing.
    Returns new P, delta_E, and accepted flag.
    """
    n_points = P.shape[0]
    idx = np.random.randint(n_points)
    
    # Generate a random rotation
    # Axis
    axis = np.random.normal(size=3)
    norm_axis = np.linalg.norm(axis)
    if norm_axis < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / norm_axis
        
    # Angle
    angle = np.random.uniform(-step_size, step_size)
    
    # Rotation matrix
    c = np.cos(angle)
    s = np.sin(angle)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    
    # Apply rotation to point idx
    P_new = P.copy()
    P_new[idx] = R @ P[idx]
    # Re-normalize to counter drift
    P_new[idx] /= np.linalg.norm(P_new[idx])
    
    # Calculate delta E
    # We can calculate full energy difference or just affected pairs.
    # For N=150, full recalc is O(N^2) = 22500 ops. 
    # Affected pairs are N-1 = 149. 
    # Let's do full recalc for simplicity first, it's fast enough in numpy.
    # E_old = _energy_only(P.flatten())
    # E_new = _energy_only(P_new.flatten())
    # This is 2 * 22500 log ops. ~45k ops. 
    # 100k steps * 45k ops = 4.5e9 ops. Might be too slow in Python/NumPy overhead.
    # Let's optimize delta E calculation.
    
    # E = - sum_{i<j} log(d_ij)
    # Delta E = - (sum_{j != idx} log(d_idx_j_new) - log(d_idx_j_old))
    
    d_old = np.linalg.norm(P[idx] - P, axis=1)
    d_new = np.linalg.norm(P_new[idx] - P_new, axis=1)
    # Exclude self
    d_old[idx] = 1.0
    d_new[idx] = 1.0
    d_old = np.clip(d_old, 1e-12, None)
    d_new = np.clip(d_new, 1e-12, None)
    
    delta_E = - (np.sum(np.log(d_new)) - np.sum(np.log(d_old)))
    
    if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
        return P_new, delta_E, True
    else:
        return P, delta_E, False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial points: Fibonacci spiral
    X_init = _fibonacci_sphere(N)
    P = X_init.copy()
    
    best_P = P.copy()
    best_E = _energy_only(P.flatten())
    
    # Simulated Annealing
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling = 0.999  # Cooling factor per step
    steps_per_temp = 1000 # Steps at each temperature
    step_size = 0.1 # Initial step size
    
    current_E = best_E
    
    # We need to be careful with time.
    # Let's estimate time per step.
    
    try:
        while time.time() < deadline - 20.0: # Leave 20s for polishing
        
            # Calculate energy for current state if not known
            # We track current_E, but P might have changed.
            # In the loop below, we update P and current_E.
            
            for _ in range(steps_per_temp):
                if time.time() > deadline - 20.0:
                    break
                    
                P_new, delta_E, accepted = _sa_step(P, T, step_size)
                if accepted:
                    P = P_new
                    current_E += delta_E
                    
                    if current_E < best_E:
                        best_E = current_E
                        best_P = P.copy()
                else:
                    # If rejected, P didn't change, current_E didn't change
                    pass
                
                # Adaptive step size?
                # If acceptance rate is too low, increase step size?
                # If too high, decrease?
                # Let's keep it simple for now.
                
            T *= cooling
            if T < T_min:
                T = T_min
                
    except Exception as e:
        print(f"SA Error: {e}")
    
    # Final Polish with L-BFGS-B
    # Use best_P found by SA
    x0 = best_P.flatten()
    
    try:
        # Use the gradient-based optimizer to find the local minimum precisely
        res = minimize(
            _energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 1000, 'gtol': 1e-12, 'maxfun': 5000}
        )
        
        if res.success or res.fun < best_E:
            best_E = res.fun
            best_P = res.x.reshape(N, 3)
    except Exception as e:
        print(f"Polish Error: {e}")
        
    # One more polish if time permits and we improved
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(
                _energy_and_grad, 
                best_P.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-14, 'maxfun': 5000}
            )
            if res2.fun < best_E:
                best_E = res2.fun
                best_P = res2.x.reshape(N, 3)
        except Exception as e:
            print(f"Polish 2 Error: {e}")

    # Normalize final result
    norms = np.linalg.norm(best_P, axis=1)
    norms[norms < 1e-12] = 1e-12
    best_P = best_P / norms[:, np.newaxis]
    
    return best_P
