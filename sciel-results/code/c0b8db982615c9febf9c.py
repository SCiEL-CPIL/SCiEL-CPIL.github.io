"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def morse_energy_and_gradient(positions, rho):
    """Compute total Morse energy and gradient for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij_vec = positions[i] - positions[j]
            rij = np.linalg.norm(rij_vec)
            if rij < 1e-10:
                rij = 1e-10
                rij_vec = np.array([1e-10, 0.0, 0.0])
            
            x = rho * (1.0 - rij)
            exp_x = np.exp(x)
            # V = exp(x) * (exp(x) - 2)
            v = exp_x * (exp_x - 2.0)
            energy += v
            
            # dV/drij = rho * exp(x) * (2*exp(x) - 1)
            dv_dr = rho * exp_x * (2.0 * exp_x - 1.0)
            # dV/dpos_i = dv_dr * (pos_i - pos_j) / rij
            grad_pair = dv_dr * rij_vec / rij
            gradient[i] += grad_pair
            gradient[j] -= grad_pair
    
    return energy, gradient


def morse_energy(positions, rho):
    """Compute total Morse energy only."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = np.linalg.norm(positions[i] - positions[j])
            if rij < 1e-10:
                rij = 1e-10
            x = rho * (1.0 - rij)
            exp_x = np.exp(x)
            energy += exp_x * (exp_x - 2.0)
    return energy


def generate_initial_cluster(n, rng, spacing=1.0):
    """Generate a random compact cluster with minimum distance constraints."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.5 * spacing
    
    # Try to place points with reasonable spacing
    max_attempts = n * 200
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all points with the constraint, just add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def basin_hopping(positions, rho, n, max_iterations=100, step_size=0.3, 
                  rng=None, time_budget=45.0):
    """
    Basin-hopping with local minimization using scipy's L-BFGS-B.
    Each iteration: perturb, then minimize locally.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, _ = morse_energy_and_gradient(positions, rho)
    best_positions = positions.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    for it in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Perturb
        perturbation = rng.normal(0, step_size, size=positions.shape)
        perturbed = positions + perturbation
        
        # Ensure minimum distance constraint (soft penalty via projection)
        # For simplicity, we'll just let the minimizer handle it
        
        # Local minimization
        try:
            result = minimize(
                morse_energy,
                perturbed,
                args=(rho,),
                method='L-BFGS-B',
                jac=lambda x, r: morse_energy_and_gradient(x, r)[1],
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            new_positions = result.x
            new_energy = result.fun
        except Exception:
            continue
        
        # Accept if lower energy
        if new_energy < current_energy:
            current_energy = new_energy
            positions = new_positions
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_positions = new_positions.copy()
        
        # Adapt step size
        if new_energy < best_energy:
            step_size = max(0.1, step_size * 0.95)
        else:
            step_size = min(1.0, step_size * 1.05)
    
    return best_positions, best_energy


def construct(instance):
    """
    Find low-energy arrangements using basin-hopping with multiple restarts.
    """
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Time budget per instance: total 300s / 6 instances = 50s each
    # Use 45s to be safe
    time_budget = 45.0
    
    rng = np.random.default_rng(hash(instance) % (2**32))
    
    best_positions = None
    best_energy = np.inf
    
    start_time = time.time()
    
    # Multiple restarts with different strategies
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Generate initial cluster
        if restart == 0:
            # More compact initial guess
            positions = generate_initial_cluster(size, rng, spacing=1.0)
        elif restart == 1:
            # Even more compact
            positions = generate_initial_cluster(size, rng, spacing=0.9)
        else:
            # Random with varying compactness
            spacing = 1.0 + 0.2 * rng.random()
            positions = generate_initial_cluster(size, rng, spacing=spacing)
        
        # Run basin-hopping with remaining time
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        bh_time = remaining_time * 0.9 / (n_restarts - restart)
        bh_positions, bh_energy = basin_hopping(
            positions, rho, size,
            max_iterations=50,
            step_size=0.3,
            rng=rng,
            time_budget=bh_time
        )
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()
    
    if best_positions is None:
        # Fallback: just return a random valid cluster
        best_positions = generate_initial_cluster(size, rng, spacing=1.0)
    
    return best_positions
# EVOLVE-BLOCK-END
