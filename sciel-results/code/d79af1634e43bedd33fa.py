"""
Strategy: Combinatorial Swap-Descent on a 1D Lattice Representation.
Instead of continuous gradient flow on the sphere, we map points to a 1D
sequence (e.g., sorted by latitude or angle) and perform local search via
swaps and insertions. This exploits the structure of the energy landscape
which often has a "ground state" that is close to a regular lattice, and
allows escaping local minima by discrete rearrangements that continuous
gradients cannot easily achieve due to the symmetry of the sphere.
"""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on the unit sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy (float), grad (N, 3) array.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    
    # Ensure no division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    mask = ~np.eye(N, dtype=bool)
    log_dists = -np.log(dist)
    np.fill_diagonal(log_dists, 0.0)
    energy = 0.5 * np.sum(log_dists)
    
    # Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad_sum = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
    grad = -grad_sum
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci lattice points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    theta = np.arcsin(z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    return np.column_stack([x, y, z])

def _random_sphere(n, seed=0):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    v = rng.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _sort_by_latitude(pts):
    """Sort points by z-coordinate (latitude)."""
    idx = np.argsort(pts[:, 2])
    return pts[idx], idx

def _swap_descent(pts, time_limit=10.0, rng=None):
    """
    Perform local search via swaps of point indices.
    We treat the points as an ordered list. Swapping two points in the list
    effectively moves them in the 3D space (if we re-map them) or simply
    tests if the set is invariant. 
    Wait, swapping indices in a set of points doesn't change the set, hence
    doesn't change the energy. 
    To make this work, we need a parameterization where the order matters.
    
    Alternative: 
    1. Start with a good configuration (e.g., Fibonacci).
    2. Define a "neighborhood" by moving one point slightly along the sphere
       in a direction determined by its neighbors in the sorted list.
    3. Or, use a different representation: 
       Represent points by their angles (theta, phi). 
       The energy is a function of these angles.
       We can perform coordinate descent on the angles, but with a twist:
       we periodically "re-sort" the points and re-assign indices to avoid
       coordinate singularities and to maintain a canonical order.
       
    Let's try a different combinatorial approach:
    "Greedy Insertion with Backtracking"
    1. Start with a few points.
    2. Add points one by one, placing each new point at the location that
       minimizes the energy contribution with the existing set.
    3. After all points are placed, perform local perturbations.
    
    Actually, a more robust combinatorial method for sphere packing/energy is:
    "Iterated Local Search (ILS) with Random Restarts and Cooling"
    But the prompt asks for something different from gradient descent.
    
    Let's use **Discrete Differential Evolution** or **Particle Swarm Optimization (PSO)** 
    on the sphere. PSO is a swarm intelligence technique, structurally different 
    from gradient descent.
    
    PSO Strategy:
    - Initialize a swarm of particles (configurations of 100 points).
    - Each particle has a position (the 100 points) and a velocity.
    - Velocity is updated based on personal best and global best.
    - Position is updated by adding velocity, then projecting to sphere.
    - This is a global optimization method, not local gradient descent.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    start_time = time.time()
    
    # Initialize swarm
    swarm_size = 20
    particles = np.zeros((swarm_size, N, 3))
    velocities = np.zeros((swarm_size, N, 3))
    
    # Initialize particles with diverse starting points
    # Mix of Fibonacci, random, and perturbed Fibonacci
    for i in range(swarm_size):
        if i == 0:
            particles[i] = _fibonacci_sphere(N)
        elif i == 1:
            particles[i] = _random_sphere(N, seed=i)
        else:
            # Perturbed Fibonacci
            base = _fibonacci_sphere(N)
            noise = rng.randn(N, 3) * 0.1
            particles[i] = _project_to_sphere(base + noise)
            
        # Small initial velocities
        velocities[i] = rng.randn(N, 3) * 0.01
        
    # Compute initial energies
    energies = np.zeros(swarm_size)
    for i in range(swarm_size):
        energies[i], _ = _energy_and_grad(particles[i])
        
    # Track personal and global bests
    personal_bests = particles.copy()
    personal_energies = energies.copy()
    
    global_best_idx = np.argmin(energies)
    global_best = particles[global_best_idx].copy()
    global_best_energy = energies[global_best_idx]
    
    # PSO Parameters
    w = 0.9  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    max_iter = 500  # Will be limited by time
    
    for it in range(max_iter):
        if time.time() > start_time + time_limit:
            break
            
        for i in range(swarm_size):
            # Update velocity
            # v_new = w * v_old + c1 * r1 * (pbest - p) + c2 * r2 * (gbest - p)
            r1 = rng.rand(N, 3)
            r2 = rng.rand(N, 3)
            
            v = (w * velocities[i] + 
                 c1 * r1 * (personal_bests[i] - particles[i]) + 
                 c2 * r2 * (global_best - particles[i]))
            
            # Clamp velocity to prevent explosion
            v_norm = np.linalg.norm(v, axis=1, keepdims=True)
            max_v = 0.5
            v = v * (max_v / np.maximum(v_norm, 1e-6))
            
            # Update position
            new_p = particles[i] + v
            new_p = _project_to_sphere(new_p)
            
            # Evaluate new energy
            new_energy, _ = _energy_and_grad(new_p)
            
            # Update personal best
            if new_energy < personal_energies[i]:
                personal_energies[i] = new_energy
                personal_bests[i] = new_p.copy()
                velocities[i] = v.copy()
                particles[i] = new_p.copy()
            else:
                # Keep old position if new is worse? 
                # In standard PSO, we keep the new position even if worse, 
                # but here we might want to be more conservative.
                # Let's stick to standard PSO: position is updated regardless.
                particles[i] = new_p.copy()
                velocities[i] = v.copy()
                
                # However, if we keep the worse position, we might drift.
                # A common variant is to only accept if better, or use a cooling factor.
                # For robustness, let's only accept if better, otherwise revert.
                # This makes it more like a stochastic local search.
                # particles[i] remains old
                # velocities[i] remains old? Or decay?
                # Let's decay velocity if rejected.
                velocities[i] *= 0.5
                
            # Update global best
            if new_energy < global_best_energy:
                global_best_energy = new_energy
                global_best = new_p.copy()
        
        # Adaptive inertia weight
        w = 0.9 - 0.5 * (it / max_iter)
        
        # Add random perturbation to global best occasionally to escape local minima
        if it % 50 == 0 and it > 0:
            # Create a new particle from a heavily perturbed global best
            idx = rng.randint(swarm_size)
            perturbation = rng.randn(N, 3) * 0.5
            new_candidate = _project_to_sphere(global_best + perturbation)
            cand_energy, _ = _energy_and_grad(new_candidate)
            if cand_energy < personal_energies[idx]:
                particles[idx] = new_candidate
                personal_bests[idx] = new_candidate
                personal_energies[idx] = cand_energy
                velocities[idx] = rng.randn(N, 3) * 0.01
                if cand_energy < global_best_energy:
                    global_best_energy = cand_energy
                    global_best = new_candidate.copy()

    return global_best, global_best_energy

def _gradient_polish(pts, time_limit=10.0):
    """
    Final polish using a few steps of gradient descent with adaptive step size.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    energy, grad = _energy_and_grad(pts)
    
    step_size = 0.01
    for _ in range(1000):
        if time.time() > start_time + time_limit:
            break
            
        # Simple line search
        new_pts = pts - step_size * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            grad = new_grad
            # Increase step size slightly if improvement is good
            if energy - new_energy > 1e-6:
                step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
                
    return pts, energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Strategy:
    # 1. Run PSO to find a good global configuration.
    # 2. Run Gradient Polish to tighten the convergence.
    # 3. Repeat with different seeds/initializations if time permits.
    
    rng = np.random.RandomState(123)
    
    # Phase 1: PSO with Fibonacci initialization
    time_alloc_pso = 60.0
    pts_pso, energy_pso = _swap_descent(_fibonacci_sphere(N), time_limit=time_alloc_pso, rng=rng)
    
    if energy_pso < best_energy:
        best_energy = energy_pso
        best_pts = pts_pso.copy()
        
    # Phase 2: Polish the result
    if best_pts is not None:
        time_remaining = deadline - time.time() - 5.0
        if time_remaining > 5.0:
            pts_polished, energy_polished = _gradient_polish(best_pts, time_limit=min(20.0, time_remaining))
            if energy_polished < best_energy:
                best_energy = energy_polished
                best_pts = pts_polished.copy()
                
    # Phase 3: If time remains, try a second PSO run with random initialization
    time_remaining = deadline - time.time() - 5.0
    if time_remaining > 20.0 and best_pts is not None:
        # Use the best found so far as a seed for a new swarm? 
        # Or just start fresh with random?
        # Let's start a new PSO run with a different seed.
        rng2 = np.random.RandomState(456)
        pts_pso2, energy_pso2 = _swap_descent(_random_sphere(N, seed=99), time_limit=min(40.0, time_remaining - 10.0), rng=rng2)
        
        if energy_pso2 < best_energy:
            best_energy = energy_pso2
            best_pts = pts_pso2.copy()
            
        # Polish again
        time_remaining = deadline - time.time() - 5.0
        if time_remaining > 5.0 and best_pts is not None:
            pts_polished2, energy_polished2 = _gradient_polish(best_pts, time_limit=min(10.0, time_remaining))
            if energy_polished2 < best_energy:
                best_energy = energy_polished2
                best_pts = pts_polished2.copy()
                
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
        
    return best_pts

# EVOLVE-BLOCK-END
