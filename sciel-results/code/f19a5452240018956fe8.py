import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute pair indices for efficient energy calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def log_energy_cartesian(pts):
        """Compute log energy from Cartesian points (N,3)."""
        diff = pts[i_idx] - pts[j_idx]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist = np.maximum(dist, 1e-15)
        return -np.sum(np.log(dist))
    
    def objective(v):
        """Objective function: v is 2N array of (theta, phi) pairs."""
        if not np.all(np.isfinite(v)):
            return 1e100
        th = v[0::2]
        ph = v[1::2]
        
        st = np.sin(th)
        x = st * np.cos(ph)
        y = st * np.sin(ph)
        z = np.cos(th)
        pts = np.column_stack([x, y, z])
        
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        
        return log_energy_cartesian(pts)
    
    def cartesian_from_params(v):
        """Convert parameter vector to Cartesian points."""
        th = v[0::2]
        ph = v[1::2]
        st = np.sin(th)
        x = st * np.cos(ph)
        y = st * np.sin(ph)
        z = np.cos(th)
        pts = np.column_stack([x, y, z])
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        return pts
    
    def fibonacci_params():
        """Generate Fibonacci spiral initial parameters."""
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        theta = np.arccos(np.clip(z, -1, 1))
        phi_sph = np.arctan2(y, x)
        phi_sph = phi_sph % (2 * np.pi)
        
        return np.column_stack([theta, phi_sph]).flatten()
    
    def random_params(rng):
        """Generate random initial parameters."""
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi = rng.uniform(0, 2 * np.pi, N)
        return np.column_stack([theta, phi]).flatten()
    
    def jittered_fibonacci_params(rng, sigma=0.02):
        """Generate jittered Fibonacci initial parameters."""
        base = fibonacci_params()
        jitter = rng.normal(0, sigma, 2 * N)
        perturbed = base + jitter
        th = perturbed[0::2]
        ph = perturbed[1::2]
        th = np.clip(th, 1e-4, np.pi - 1e-4)
        ph = ph % (2 * np.pi)
        return np.column_stack([th, ph]).flatten()
    
    best_energy = np.inf
    best_params = None
    
    rng = np.random.default_rng(42)
    
    initializations = []
    initializations.append(fibonacci_params())
    initializations.append(jittered_fibonacci_params(rng, 0.02))
    initializations.append(jittered_fibonacci_params(rng, 0.05))
    initializations.append(random_params(rng))
    initializations.append(random_params(rng))
    
    for x0 in initializations:
        if time.time() > deadline - 15.0:
            break
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if np.all(np.isfinite(res.x)) and res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res_polish = minimize(objective, best_params, method='L-BFGS-B',
                                 options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
            if np.all(np.isfinite(res_polish.x)) and res_polish.fun < best_energy:
                best_energy = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = fibonacci_params()
    
    pts = cartesian_from_params(best_params)
    
    if not np.all(np.isfinite(pts)):
        pts = cartesian_from_params(fibonacci_params())
    
    diff = pts[i_idx] - pts[j_idx]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    if np.min(dist) < 1e-9:
        pts = cartesian_from_params(fibonacci_params())
    
    return pts
