"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def morse_energy_and_gradient(pos, rho):
    """Compute Morse energy and gradient for a cluster.
    
    E = sum_{i<j} exp(rho*(1-r_ij))*(exp(rho*(1-r_ij)) - 2)
    """
    n = pos.shape[0]
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Compute all pair distances
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-8:
                continue
            x = rho * (1.0 - r)
            ex = np.exp(x)
            # Energy term
            e_ij = ex * (ex - 2.0)
            energy += e_ij
            # Gradient: dE/dr_ij = rho * ex * ex * sign... 
            # Actually: d/dr [exp(rho(1-r))(exp(rho(1-r))-2)]
            # Let u = rho(1-r), du/dr = -rho
            # dE/du = exp(u)(exp(u) - 2) + exp(u)*exp(u) = exp(u)(2exp(u) - 2) = 2exp(u)(exp(u)-1)
            # dE/dr = dE/du * du/dr = -rho * 2*exp(u)*(exp(u)-1)
            de_dr = -rho * 2.0 * ex * (ex - 1.0)
            # dE/dpos_i = de_dr * (pos_i - pos_j)/r
            # dE/dpos_j = -de_dr * (pos_i - pos_j)/r
            factor = de_dr / r
            grad[i] += factor * rij
            grad[j] -= factor * rij
    
    return energy, grad


def morse_energy_only(pos, rho):
    """Compute Morse energy only (faster for SA steps)."""
    n = pos.shape[0]
    energy = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-8:
                continue
            x = rho * (1.0 - r)
            ex = np.exp(x)
            energy += ex * (ex - 2.0)
    
    return energy


def basin_hopping(pos, rho, n_atoms, max_iters=200, step_size=0.1, 
                   inner_maxiter=100, time_budget=30.0):
    """Basin hopping with local minimization via L-BFGS-B.
    
    Uses gradient-based local minimization within each basin hop.
    """
    best_pos = pos.copy()
    best_energy = morse_energy_and_gradient(best_pos, rho)[0]
    
    rng = np.random.default_rng(42)
    start_time = time.time()
    
    for it in range(max_iters):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Generate perturbation
        perturbation = rng.normal(0, step_size, size=pos.shape)
        trial_pos = pos + perturbation
        
        # Enforce minimum distance constraint (soft)
        # If any pair is too close, push them apart
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = trial_pos[i] - trial_pos[j]
                dist = np.linalg.norm(d)
                if dist < 0.5:
                    # Push apart
                    if dist < 1e-8:
                        d = rng.normal(0, 0.1, 3)
                        dist = np.linalg.norm(d)
                    push = (0.5 - dist) / dist * 0.5
                    trial_pos[i] += push * d
                    trial_pos[j] -= push * d
        
        # Local minimization from trial position
        def objective(x):
            e, g = morse_energy_and_gradient(x, rho)
            return e, g
        
        try:
            result = minimize(objective, trial_pos, method='L-BFGS-B', 
                             jac=True, options={'maxiter': inner_maxiter, 'gtol': 1e-6})
            trial_energy = result.fun
            trial_pos = result.x
        except Exception:
            trial_energy = morse_energy_only(trial_pos, rho)
        
        # Metropolis acceptance
        delta_e = trial_energy - best_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
            pos = trial_pos.copy()
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_pos = trial_pos.copy()
        
        # Gradually reduce step size
        step_size *= 0.995
    
    return best_pos, best_energy


def construct(instance):
    """Construct a low-energy Morse cluster using basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time budget: 300s total for 6 instances, so ~45s per instance
    # Leave some margin: use 40s per instance
    time_budget_per_instance = 40.0
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Initial radius based on expected cluster size
    # For n atoms with unit spacing, radius ~ n^(1/3)
    base_radius = 1.0 * n ** (1.0 / 3.0) * 0.8
    
    best_pos = None
    best_energy = np.inf
    
    # Try multiple starting points with basin hopping
    n_starts = min(8, max(3, int(time_budget_per_instance / 5)))
    
    start_time = time.time()
    for start_idx in range(n_starts):
        # Check time
        if time.time() - start_time > time_budget_per_instance:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            # Roughly spherical shell
            points = []
            attempts = 0
            while len(points) < n and attempts < n * 100:
                # Random point in sphere
                theta = rng.uniform(0, 2 * np.pi)
                phi = np.arccos(rng.uniform(-1, 1))
                r = base_radius * rng.uniform(0.3, 1.0)
                p = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
                # Check distance to existing points
                ok = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.5:
                        ok = False
                        break
                if ok:
                    points.append(p)
                attempts += 1
            
            # If we couldn't place all points, fill remaining randomly
            while len(points) < n:
                p = rng.uniform(-base_radius, base_radius, 3)
                points.append(p)
            
            pos = np.array(points)
        else:
            # Random compact configuration
            pos = rng.uniform(-base_radius, base_radius, size=(n, 3))
            # Center of mass to origin
            pos -= np.mean(pos, axis=0)
        
        # Run basin hopping
        bh_pos, bh_energy = basin_hopping(
            pos, rho, n, 
            max_iters=50, 
            step_size=0.15,
            inner_maxiter=80,
            time_budget=time_budget_per_instance / max(1, n_starts - start_idx)
        )
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Final polish: run a few more local optimizations from best position
    if best_pos is not None:
        def objective(x):
            e, g = morse_energy_and_gradient(x, rho)
            return e, g
        
        try:
            result = minimize(objective, best_pos, method='L-BFGS-B', 
                             jac=True, options={'maxiter': 200, 'gtol': 1e-8})
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: random configuration
        best_pos = rng.uniform(-base_radius, base_radius, size=(n, 3))
        best_pos -= np.mean(best_pos, axis=0)
    
    return best_pos
# EVOLVE-BLOCK-END
