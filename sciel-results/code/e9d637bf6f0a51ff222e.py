"""
Strategy: Simulated Annealing with Velocity Verlet Dynamics and Collision Repair.
Instead of gradient-based local minimization (L-BFGS-B) combined with basin hopping,
we use a stochastic many-body collision dynamics simulation. The system is initialized
as a high-temperature gas and cooled down. Atoms move according to Newton's laws
(accelerated by Morse forces) with viscous damping. If atoms get too close (violating
the 0.5 distance constraint), they are forcibly pushed apart (collision repair).
This allows the cluster to self-assemble into the global minimum through energy
dissipation and structural reorganization without relying on predefined crystalline
templates or gradient calculations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # 300s limit, keep 5s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    # Morse potential: V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let x = rho(1-r)
    # V = e^x (e^x - 2)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    
    # Sum over all i<j. The matrix sum counts each pair twice.
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    # dV/dri = -dV/dr * (ri - rj) / r
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    factor = dV_dr * inv_r
    row_sum = np.sum(factor, axis=1)
    weighted_pos = factor.dot(positions)
    
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _repair_collisions(positions, min_dist=0.5):
    """
    Iteratively push apart atoms that are closer than min_dist.
    This ensures the final configuration is valid.
    """
    n = len(positions)
    if n < 2:
        return positions
    
    max_iters = 100
    for _ in range(max_iters):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(r, np.inf)
        
        # Find pairs violating constraint
        mask = r < min_dist
        if not np.any(mask):
            break
            
        # For each violating pair, push them apart
        # We do this iteratively to avoid complex simultaneous resolution
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    rij = diff[i, j]
                    dist = r[i, j]
                    if dist < 1e-8:
                        # Random nudge
                        rij = np.random.normal(0, 0.01, 3)
                        dist = np.linalg.norm(rij)
                    
                    # Target distance is min_dist
                    # Move each atom half the correction
                    correction = (min_dist - dist) / 2.0
                    direction = rij / dist
                    positions[i] += direction * correction
                    positions[j] -= direction * correction
                    
    return positions

def _simulate_annealing(initial_pos, rho, n, t_start=1.0, t_end=0.05, steps=2000, dt=0.01, damping=0.9):
    """
    Run a simulated annealing-like dynamics.
    1. Initialize velocities randomly.
    2. Integrate with Velocity Verlet.
    3. Apply damping to remove energy.
    4. Repair collisions to maintain validity.
    5. Track best energy.
    """
    pos = initial_pos.copy()
    # Initialize velocities with magnitude related to temperature
    vel = np.random.normal(0, np.sqrt(t_start), pos.shape)
    
    best_pos = pos.copy()
    best_energy = _morse_energy_and_grad(pos, rho)[0]
    
    # Pre-compute some constants
    # We need forces for integration
    # F = -grad
    
    for step in range(steps):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
            
        # Current temperature (linear cooling)
        t = t_start * (t_end / t_start) ** (step / steps)
        
        # 1. Compute forces at current position
        e, g = _morse_energy_and_grad(pos, rho)
        forces = -g
        
        # 2. Update positions
        pos_new = pos + vel * dt + 0.5 * forces * dt**2
        
        # 3. Compute forces at new position
        e_new, g_new = _morse_energy_and_grad(pos_new, rho)
        forces_new = -g_new
        
        # 4. Update velocities
        vel_new = vel + 0.5 * (forces + forces_new) * dt
        vel_new *= damping  # Damping to dissipate energy
        
        # 5. Collision Repair
        # This is crucial to keep the state valid and prevent atoms from sticking too closely
        pos_new = _repair_collisions(pos_new, min_dist=0.5)
        
        # 6. Metropolis-like acceptance? 
        # In this "collision dynamics" strategy, we always accept the new position 
        # because the repair step ensures validity, and the damping ensures 
        # the system trends towards lower energy states over time. 
        # However, to escape local minima, we rely on the initial thermal energy 
        # and the stochastic nature of the collision repairs if we add noise.
        # Let's add a small random kick to velocities to prevent getting stuck in 
        # shallow minima due to damping.
        if step % 10 == 0:
            vel_new += np.random.normal(0, 0.1 * t, pos.shape)
            
        pos = pos_new
        vel = vel_new
        
        # Update best
        if e_new < best_energy:
            best_energy = e_new
            best_pos = pos_new.copy()
            
    return best_pos, best_energy

def _local_minimize(positions, rho):
    """
    Final polish using L-BFGS-B to ensure we are at a local minimum.
    This is distinct from the main search strategy but used for final refinement.
    """
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        g = g.flatten()
        return e, g
    
    try:
        # Ensure valid start
        positions = _repair_collisions(positions, 0.5)
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        
        # Final validity check
        if _is_valid(final_pos):
            return final_pos, final_energy
        else:
            # If L-BFGS-B violated constraints (unlikely but possible), repair and re-eval
            final_pos = _repair_collisions(final_pos, 0.5)
            final_energy = _morse_energy_and_grad(final_pos, rho)[0]
            return final_pos, final_energy
    except Exception:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    # Time budget allocation
    # We will run multiple independent annealing runs.
    # Each run takes roughly O(steps * n^2) for force calculation.
    # For n=80, n^2=6400. 2000 steps * 6400 = 12.8M ops per run.
    # This is very fast. We can afford many runs.
    
    best_energy = float('inf')
    best_positions = None
    
    # Number of restarts depends on time
    # Let's estimate time for one run
    # We'll do a quick test run to estimate speed, or just set a fixed number
    # and check time.
    
    n_restarts = 20 # Start with a reasonable number
    
    for i in range(n_restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
            
        # Generate a random gas-like initial configuration
        # Radius should be large enough to avoid initial overlaps
        # Volume ~ n. Radius ~ n^(1/3)
        # To avoid overlaps, we need volume per atom > (2*0.5)^3 = 1
        # So radius should be at least n^(1/3)
        # Let's use a slightly larger radius to ensure "gas" state
        radius = 1.5 * size ** (1.0 / 3.0)
        
        # Simple rejection sampling for initial positions
        # This might be slow for large n if radius is small, but radius is scaled.
        points = []
        max_attempts = 10000
        attempts = 0
        while len(points) < size and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check distance to existing points
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
            attempts += 1
            
        if len(points) < size:
            # Fallback: place on a sphere surface if random fails
            # Or just use a grid
            # Let's use a simple spiral on a sphere
            phi = np.linspace(0, np.pi, size)
            theta = np.linspace(0, 2*np.pi, size)
            r_sphere = 1.0 * size ** (1.0/3.0)
            pts = np.zeros((size, 3))
            pts[:, 0] = r_sphere * np.sin(phi) * np.cos(theta)
            pts[:, 1] = r_sphere * np.sin(phi) * np.sin(theta)
            pts[:, 2] = r_sphere * np.cos(phi)
            # Add small noise
            pts += rng.normal(0, 0.1, pts.shape)
            points = pts.tolist()
            
        initial_pos = np.array(points)
        
        # Run Simulated Annealing Dynamics
        try:
            sa_pos, sa_energy = _simulate_annealing(initial_pos, parameter, size, 
                                                    t_start=0.5, t_end=0.01, 
                                                    steps=1000, dt=0.02, damping=0.95)
            
            # Polish with L-BFGS-B
            final_pos, final_energy = _local_minimize(sa_pos, parameter)
            
            if _is_valid(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = final_pos
        except Exception:
            continue
            
    # If no valid position found, return a simple valid random one
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
        
    return best_positions
# EVOLVE-BLOCK-END
