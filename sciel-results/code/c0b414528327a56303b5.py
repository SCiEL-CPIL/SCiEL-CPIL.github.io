"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|) using fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return 1e10
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _repulsion_forces(pts):
    """Compute repulsion forces (gradient of negative energy) for all points."""
    n = len(pts)
    forces = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            # Force is proportional to d / dist^2 * (1/dist) for log potential
            # Actually, gradient of log(1/|x-y|) = -1/|x-y|^2 * (x-y)
            # So repulsion force (minimizing energy) is +1/|x-y|^2 * (x-y)
            # Wait, we want to minimize sum log(1/d) = -sum log(d)
            # Gradient w.r.t. x_i of -log(|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
            # So we move in direction of negative gradient = +(x_i - x_j)/|x_i - x_j|^2
            # That's the repulsion direction
            forces[i] += d / (dist * dist)
    return forces

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral().copy()
    current_energy = _energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Simulated annealing with decreasing temperature
    T = 0.01
    T_min = 1e-10
    cooling = 0.999
    
    # Number of iterations per temperature step
    iters_per_step = 50
    
    step_size = 0.05
    
    while time.time() < deadline:
        # Run several iterations at current temperature
        for _ in range(iters_per_step):
            if time.time() > deadline:
                break
            
            # Pick a random point to perturb
            idx = np.random.randint(N)
            orig = pts[idx].copy()
            
            # Random perturbation
            perturbation = np.random.randn(3) * step_size
            new_pt = pts[idx] + perturbation
            new_pt = _normalize(new_pt)
            
            # Compute energy change (approximate: only terms involving idx change)
            # For efficiency, compute old and new energy contributions
            old_contrib = 0.0
            new_contrib = 0.0
            for j in range(N):
                if j == idx:
                    continue
                d_old = np.linalg.norm(orig - pts[j])
                d_new = np.linalg.norm(new_pt - pts[j])
                if d_old < 1e-12 or d_new < 1e-12:
                    continue
                old_contrib += math.log(1.0 / d_old)
                new_contrib += math.log(1.0 / d_new)
            
            delta_e = new_contrib - old_contrib
            
            # Accept if better, or with probability exp(-delta_e / T)
            if delta_e < 0 or np.random.random() < math.exp(-delta_e / T):
                pts[idx] = new_pt
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
            
            # Update step size based on acceptance (simple adaptation)
            # Keep step size roughly constant but decrease slowly
            step_size *= 0.9995
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Occasionally re-heat if stuck (optional)
        # Reduce step size as we cool
        step_size = max(0.001, 0.05 * T / 0.01)
    
    return best_pts

# EVOLVE-BLOCK-END
