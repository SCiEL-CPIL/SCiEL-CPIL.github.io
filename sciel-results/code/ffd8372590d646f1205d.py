"""Dzugutov cluster global minima search using vectorized energy/gradient, L-BFGS, and perturbation restarts."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for n atoms. Vectorized pair computation."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    # Use upper triangle indices for efficiency
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = positions[i_idx] - positions[j_idx]
    r = np.sqrt(np.sum(diff**2, axis=1))
    
    # Avoid division by zero
    r_safe = np.maximum(r, 1e-10)
    
    # Region 1: r < a
    mask_a = r < a
    energy_a = np.zeros_like(r)
    if np.any(mask_a):
        r_a = r_safe[mask_a]
        energy_a[mask_a] = (A * (r_a**(-m) - B)) * np.exp(c / (r_a - a))
    
    # Region 2: a <= r < b
    mask_b = (r >= a) & (r < b)
    energy_b = np.zeros_like(r)
    if np.any(mask_b):
        r_b = r_safe[mask_b]
        energy_b[mask_b] = B * np.exp(d / (r_b - b))
    
    # Region 3: r >= b
    # energy = 0
    
    return np.sum(energy_a + energy_b)

def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy. Vectorized."""
    n = positions.shape[0]
    if n < 2:
        return np.zeros_like(positions)
    
    grad = np.zeros_like(positions)
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = positions[i_idx] - positions[j_idx]
    r = np.sqrt(np.sum(diff**2, axis=1))
    r_safe = np.maximum(r, 1e-10)
    
    # Compute force magnitudes (negative gradient of energy w.r.t. r)
    # dV/dr for each region
    dVdr = np.zeros_like(r)
    
    # Region 1: r < a
    mask_a = r < a
    if np.any(mask_a):
        r_a = r_safe[mask_a]
        # V_a = A*(r^-m - B)*exp(c/(r-a))
        # dV_a/dr = A * (-m*r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2))
        exp_term = np.exp(c / (r_a - a))
        dVdr[mask_a] = A * (-m * r_a**(-m-1) * exp_term + (r_a**(-m) - B) * exp_term * (-c / (r_a - a)**2))
    
    # Region 2: a <= r < b
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        r_b = r_safe[mask_b]
        # V_b = B*exp(d/(r-b))
        # dV_b/dr = B*exp(d/(r-b)) * (-d/(r-b)^2)
        dVdr[mask_b] = B * np.exp(d / (r_b - b)) * (-d / (r_b - b)**2)
    
    # Region 3: r >= b -> dV/dr = 0
    
    # Gradient with respect to positions: grad_i = sum_{j != i} dV/dr * (r_i - r_j)/r
    # For pair (i,j), contribution to grad[i] = -dVdr * diff/r, grad[j] = +dVdr * diff/r
    # where diff = pos[i] - pos[j]
    
    # diff/r gives unit vector from j to i
    unit_vec = diff / r_safe[:, np.newaxis]
    
    # Force on i due to j: -dVdr * unit_vec (since grad = -force, and force = -dV/dr * unit_vec_from_i_to_j)
    # Actually: dE/dpos_i = sum_j dV/dr * (pos_i - pos_j)/r = sum_j dVdr * unit_vec
    # dE/dpos_j = sum_i dV/dr * (pos_j - pos_i)/r = sum_i dVdr * (-unit_vec)
    
    # Accumulate using np.add.at
    force_magnitudes = dVdr  # This is dV/dr
    # Contribution to grad[i_idx] += dVdr * unit_vec
    # Contribution to grad[j_idx] -= dVdr * unit_vec
    
    np.add.at(grad, i_idx, force_magnitudes[:, np.newaxis] * unit_vec)
    np.add.at(grad, j_idx, -force_magnitudes[:, np.newaxis] * unit_vec)
    
    return grad

def _make_initial_positions(n, rng, spacing=1.1):
    """Generate initial positions for n atoms in a compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def construct(instance):
    """Find low-energy Dzugutov cluster structures using L-BFGS with perturbation restarts."""
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    rng = np.random.default_rng(42 + n)
    
    best_energy = np.inf
    best_positions = None
    
    # Try multiple initial configurations with L-BFGS
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial positions
        if restart == 0:
            # First restart: more structured initial guess
            positions = _make_initial_positions(n, rng, spacing=1.1)
        else:
            # Subsequent restarts: perturb the best found so far, or random
            if best_positions is not None and restart < n_restarts // 2:
                # Perturb best
                sigma = 0.12
                positions = best_positions + rng.normal(0, sigma, size=best_positions.shape)
            else:
                # Random new start
                positions = _make_initial_positions(n, rng, spacing=1.1)
        
        # Optimize using L-BFGS
        try:
            result = minimize(
                _dzugutov_energy,
                positions,
                jac=_dzugutov_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_positions = result.x.copy()
        except Exception:
            continue
    
    # Final polish: high-precision optimization from best
    if best_positions is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                _dzugutov_energy,
                best_positions,
                jac=_dzugutov_gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_positions = result.x.copy()
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: return random positions
        best_positions = _make_initial_positions(n, rng, spacing=1.1)
    
    return best_positions
# EVOLVE-BLOCK-END
