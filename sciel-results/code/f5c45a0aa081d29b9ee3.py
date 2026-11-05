"""Dzugutov cluster global minima via continuous annealing of softness parameters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Dzugutov potential constants
_A = 5.82
_B = 1.28
_a = 1.87
_b = 1.94
_c_real = 1.10
_d_real = 0.27
_m = 16

def _energy_and_gradient(pos, c_val, d_val):
    """
    Vectorized computation of Dzugutov energy and gradient.
    V(r) = A*(r^-m - B)*exp(c/(r-a))  for r < a
    V(r) = B*exp(d/(r-b))              for a <= r < b
    V(r) = 0                            for r >= b
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Pairwise differences and distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    r = np.sqrt(np.maximum(r2, 1e-20))
    
    # Mask out diagonal
    np.fill_diagonal(r, np.inf)
    np.fill_diagonal(r2, np.inf)
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Region 1: r < a
    mask_a = r < _a
    if np.any(mask_a):
        r_a = r[mask_a]
        # V = A*(r^-m - B)*exp(c/(r-a))
        exp_term = np.exp(c_val / (r_a - _a))
        V = _A * (r_a**(-_m) - _B) * exp_term
        energy += np.sum(V)
        
        # dV/dr = A*[-m*r^(-m-1)*exp(c/(r-a)) + (r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)]
        dV_dr = _A * (-_m * r_a**(-_m - 1) * exp_term +
                      (r_a**(-_m) - _B) * exp_term * (-c_val / (r_a - _a)**2))
        
        # Scatter gradient contributions
        # For pair (i,j), grad[i] += dV_dr/r * diff[i,j], grad[j] -= dV_dr/r * diff[i,j]
        factor = dV_dr / r_a
        # Get indices
        idx_a = np.where(mask_a)
        for k in range(len(r_a)):
            i, j = idx_a[0][k], idx_a[1][k]
            f = factor[k]
            grad[i] += f * diff[i, j]
            grad[j] -= f * diff[i, j]
    
    # Region 2: a <= r < b
    mask_b = (r >= _a) & (r < _b)
    if np.any(mask_b):
        r_b = r[mask_b]
        # V = B*exp(d/(r-b))
        exp_term = np.exp(d_val / (r_b - _b))
        V = _B * exp_term
        energy += np.sum(V)
        
        # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
        dV_dr = _B * exp_term * (-d_val / (r_b - _b)**2)
        
        factor = dV_dr / r_b
        idx_b = np.where(mask_b)
        for k in range(len(r_b)):
            i, j = idx_b[0][k], idx_b[1][k]
            f = factor[k]
            grad[i] += f * diff[i, j]
            grad[j] -= f * diff[i, j]
    
    return float(energy), grad

def _energy(pos):
    """Compute Dzugutov energy with real parameters."""
    e, _ = _energy_and_gradient(pos, _c_real, _d_real)
    return e

def _gradient(pos):
    """Compute Dzugutov gradient with real parameters."""
    _, g = _energy_and_gradient(pos, _c_real, _d_real)
    return g

def _build_seed(n_atoms, rng):
    """
    Build a polytetrahedral-like seed structure.
    Start with a regular tetrahedron and add atoms in shells
    favoring tetrahedral coordination.
    """
    edge = 1.95  # near-minimum distance for Dzugutov
    
    # Regular tetrahedron centered at origin
    # Vertices of regular tetrahedron with edge length s:
    # (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) scaled appropriately
    # Distance between (1,1,1) and (1,-1,-1) = sqrt(0+4+4) = 2*sqrt(2)
    # So scale = edge / (2*sqrt(2))
    scale = edge / (2 * np.sqrt(2))
    v0 = np.array([1, 1, 1]) * scale
    v1 = np.array([1, -1, -1]) * scale
    v2 = np.array([-1, 1, -1]) * scale
    v3 = np.array([-1, -1, 1]) * scale
    
    points = [v0, v1, v2, v3]
    
    min_dist = 0.52  # slightly above 0.5 for safety
    
    while len(points) < n_atoms:
        added = False
        for attempt in range(300):
            # Random point on sphere of radius ~edge
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            r = edge * rng.uniform(0.85, 1.25)
            p = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            
            # Check minimum distance to all existing points
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            
            if valid:
                points.append(p)
                added = True
                break
        
        if not added:
            # Force add at larger radius
            r = edge * rng.uniform(1.3, 2.0)
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            points.append(p)
    
    return np.array(points[:n_atoms])

def _enforce_min_dist(pos, min_d=0.5, max_iter=20):
    """Iteratively enforce minimum pairwise distance."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        max_violation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_d:
                    if r > 1e-10:
                        push = (min_d - r) * d / r
                        pos[i] += 0.5 * push
                        pos[j] -= 0.5 * push
                    else:
                        pos[i] += np.random.uniform(-0.01, 0.01, 3)
                    max_violation = max(max_violation, min_d - r)
        if max_violation < 1e-8:
            break
    return pos

def _anneal(pos, n_steps=1500, c_start=4.0, d_start=1.2, lr_init=0.03):
    """
    Continuous annealing: gradually reduce c and d from high values to real values.
    Uses adaptive gradient descent with line-search-like step control.
    """
    pos = pos.copy()
    c_end = _c_real
    d_end = _d_real
    
    best_energy = _energy(pos)
    best_pos = pos.copy()
    
    for step in range(n_steps):
        t = step / max(n_steps - 1, 1)
        
        # Smooth annealing schedule: use cosine easing for smoother transition
        # cos(t*pi/2) goes from 1 to 0 as t goes from 0 to 1
        ease = np.cos(t * np.pi / 2)
        c_val = c_end + (c_start - c_end) * ease
        d_val = d_end + (d_start - d_end) * ease
        
        # Compute gradient
        e_curr, grad = _energy_and_gradient(pos, c_val, d_val)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-14:
            break
        
        # Adaptive learning rate: larger early, smaller late
        # Also scale inversely with gradient magnitude for stability
        lr = lr_init * (0.3 + 0.7 * ease) / (1.0 + 0.5 * grad_norm)
        lr = min(lr, 0.08)
        
        # Take step
        pos_new = pos - lr * grad
        
        # Enforce minimum distance
        pos_new = _enforce_min_dist(pos_new, min_d=0.5, max_iter=10)
        
        # Check if energy improved (with real potential for tracking)
        e_new_real = _energy(pos_new)
        
        if e_new_real < best_energy:
            best_energy = e_new_real
            best_pos = pos_new.copy()
        
        pos = pos_new
    
    return best_pos

def construct(instance):
    """Construct a low-energy Dzugutov cluster using continuous annealing."""
    n_atoms = instance["n"]
    rng = np.random.default_rng(n_atoms * 7919)
    
    start_time = time.time()
    time_budget = 42.0  # seconds per instance (6 instances * 42 = 252 < 300)
    
    # Build initial seed
    pos = _build_seed(n_atoms, rng)
    pos = _enforce_min_dist(pos, min_d=0.55)
    
    # Phase 1: Continuous annealing from soft to hard potential
    annealed = _anneal(pos, n_steps=2000, c_start=4.5, d_start=1.5, lr_init=0.025)
    
    # Phase 2: Final local refinement with L-BFGS-B
    def obj(x):
        p = x.reshape(n_atoms, 3)
        return _energy(p)
    
    def jac(x):
        p = x.reshape(n_atoms, 3)
        return _gradient(p).flatten()
    
    result = minimize(obj, annealed.flatten(), jac=jac, method='L-BFGS-B',
                      options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
    
    best_pos = result.x.reshape(n_atoms, 3)
    best_e = result.fun
    
    # Phase 3: If time remains, try a second annealing run from a perturbed seed
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.55:
        pos2 = _build_seed(n_atoms, rng)
        pos2 = _enforce_min_dist(pos2, min_d=0.55)
        annealed2 = _anneal(pos2, n_steps=1500, c_start=3.5, d_start=1.0, lr_init=0.02)
        
        result2 = minimize(obj, annealed2.flatten(), jac=jac, method='L-BFGS-B',
                          options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-12})
        
        if result2.fun < best_e:
            best_pos = result2.x.reshape(n_atoms, 3)
            best_e = result2.fun
    
    # Final distance enforcement
    best_pos = _enforce_min_dist(best_pos, min_d=0.5, max_iter=30)
    
    return best_pos
# EVOLVE-BLOCK-END
