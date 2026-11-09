"""
Morse Cluster Global Minimizer using Simulated Annealing with Gradient Updates.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
TOTAL_TIME_BUDGET = 295.0 # Leave 5s margin for overhead
INSTANCE_TIME_BUDGET = TOTAL_TIME_BUDGET / 6.0
start_time_global = time.time()
_instance_count = 0

def _generate_initial_positions(n, rng, mode='fcc'):
    """
    Generate initial positions for n atoms.
    Modes: 'fcc', 'random', 'icosahedral_approx'
    """
    if mode == 'fcc':
        # Generate a dense FCC-like structure
        # For n atoms, estimate side length L ~ n^(1/3)
        L = int(np.ceil(n ** (1/3.0))) + 2
        coords = []
        # Use a simple cubic grid with FCC offsets to ensure density
        # FCC lattice constant a such that nearest neighbor distance is 1.
        # a = sqrt(2)
        a = np.sqrt(2.0)
        
        points = []
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                for k in range(-L, L+1):
                    base = np.array([i, j, k]) * a
                    points.append(base)
                    points.append(base + a * np.array([0, 0.5, 0.5]))
                    points.append(base + a * np.array([0.5, 0, 0.5]))
                    points.append(base + a * np.array([0.5, 0.5, 0]))
                    
        points = np.array(points)
        # Center of mass
        com = points.mean(axis=0)
        points -= com
        # Sort by distance to center
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        selected = points[indices[:n]]
        return selected

    elif mode == 'random':
        # Random points in a sphere
        radius = (n * 0.5) ** (1/3.0) * 1.5 # Heuristic radius
        points = []
        attempts = 0
        while len(points) < n and attempts < n * 50:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            # Check min distance 0.5
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
            attempts += 1
        if len(points) < n:
            # Fallback: just add random points, relax will fix
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
        return np.array(points)
    else:
        # Default to random
        return _generate_initial_positions(n, rng, 'random')

def construct(instance):
    global start_time_global, _instance_count
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Check global time
    elapsed_global = time.time() - start_time_global
    if elapsed_global > TOTAL_TIME_BUDGET - 5.0:
        # Emergency exit
        return _generate_initial_positions(size, np.random.default_rng(0), 'fcc')
    
    # Determine time budget for this instance
    # We assume instances are called sequentially.
    # We don't know the order, so we use a dynamic budget.
    # If we've used X seconds, and we have 6 instances total,
    # assume remaining instances = 6 - _instance_count.
    # But _instance_count is incremented after.
    # Let's just use a fixed fraction of remaining time, capped.
    remaining_total = TOTAL_TIME_BUDGET - elapsed_global
    # Assume this is one of the remaining 6 instances.
    # To be safe, cap at 45s per instance.
    inst_budget = min(INSTANCE_TIME_BUDGET, remaining_total * 0.45)
    
    start_time_inst = time.time()
    rng = np.random.default_rng(size * 12345 + _instance_count)
    
    best_energy = np.inf
    best_pos = None
    
    # Initial structure: FCC is usually a good starting point for these clusters
    pos = _generate_initial_positions(size, rng, 'fcc')
    
    # Initial relaxation to ensure validity and get to a local min
    try:
        e, pos = relax(pos, potential, rho)
        best_energy = e
        best_pos = pos.copy()
    except Exception:
        # If relax fails, try random
        pos = _generate_initial_positions(size, rng, 'random')
        try:
            e, pos = relax(pos, potential, rho)
            best_energy = e
            best_pos = pos.copy()
        except Exception:
            # Last resort
            return pos

    # Simulated Annealing parameters
    # T0: Initial temperature. Needs to be high enough to explore.
    # Energy scale is roughly N * 1.0 (well depth).
    # T0 ~ N * 0.5 ?
    T0 = size * 0.5
    T_end = 0.01
    alpha = 0.999 # Cooling factor per step
    
    # We will run SA for a fixed number of steps or until time runs out.
    # Each step involves a gradient calculation (fast) and maybe a relax (slow).
    # Strategy:
    # 1. Small perturbations (noise) + Gradient descent step.
    # 2. Periodically (every K steps) do a full relax to find the basin bottom.
    # 3. If energy improves, update best.
    
    # Let's estimate cost.
    # Gradient is O(N^2). Relax is O(N^2 * iter).
    # Let's do 5000 SA steps.
    # Every 50 steps, do a relax.
    
    step_count = 0
    max_steps = 10000 # Upper bound, will break on time
    relax_freq = 50
    noise_scale = 0.1
    lr = 0.01 # Learning rate for gradient step
    
    current_energy = best_energy
    current_pos = best_pos.copy()
    
    while time.time() - start_time_inst < inst_budget:
        if time.time() - start_time_global > TOTAL_TIME_BUDGET - 2.0:
            break
            
        step_count += 1
        
        # Cooling
        T = T0 * (T_end / T0) ** (step_count / max_steps)
        
        # 1. Perturbation
        # Mix of random noise and gradient step
        # Calculate gradient
        try:
            grad = morse_energy_gradient(current_pos, rho)
        except Exception:
            grad = np.zeros_like(current_pos)
            
        # Gradient descent step
        # Normalize gradient to prevent huge jumps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            grad = grad / grad_norm * lr
        else:
            grad = 0.0
            
        # Random noise
        noise = rng.normal(0, noise_scale, size=current_pos.shape)
        
        # Move
        new_pos = current_pos - grad + noise
        
        # 2. Check validity (min distance)
        # For SA, we can tolerate some invalidity if T is high, but let's try to keep it valid.
        # A simple check: if any pair < 0.5, push them apart slightly or reject.
        # For speed, we might skip explicit check and let energy be high (penalty).
        # But relax requires valid input? The task says "atoms must be at least 0.5 apart".
        # If we pass invalid to relax, it might fail.
        # Let's do a cheap validity check.
        # N=80, N^2=6400. np.triu_indices is fast.
        
        # Optimization: Only check validity every few steps? No, better to ensure.
        # Use a vectorized check.
        dists = np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=2)
        # Fill diagonal with inf
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5:
            # Reject or fix?
            # Simple rejection for now to maintain validity.
            # Or, we can just proceed and let the high energy (if potential is defined) handle it?
            # Morse potential is defined for r>0. If r < 0.5, energy is very high (repulsive).
            # The gradient will push them apart.
            # However, relax() might crash or behave badly on invalid inputs.
            # Let's assume relax handles it or we should ensure validity.
            # To be safe, if invalid, we skip the move (reject).
            continue
            
        # 3. Compute new energy
        # We don't have a direct energy function in the signature, but we can compute it.
        # E = sum exp(rho(1-r))(exp(rho(1-r))-2)
        # This is O(N^2).
        # We already computed dists.
        r = dists
        # Avoid division by zero or log issues? Morse is fine.
        # Only sum i<j
        iu = np.triu_indices(n, 1)
        r_pairs = r[iu]
        x = rho * (1 - r_pairs)
        # exp(x) can overflow if x is large?
        # rho is 3, 6, 10. r can be small.
        # If r=0.5, x = rho*0.5. For rho=10, x=5. exp(5)=148. Fine.
        # If r is very small, x is large.
        # But we ensured r >= 0.5.
        
        ex = np.exp(x)
        e_new = np.sum(ex * (ex - 2))
        
        # 4. Metropolis Acceptance
        if e_new < current_energy:
            current_energy = e_new
            current_pos = new_pos
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            # Accept with probability exp(-dE / T)
            dE = e_new - current_energy
            # If T is small, probability is ~0
            if T > 1e-6:
                p = np.exp(-dE / T)
                if rng.random() < p:
                    current_energy = e_new
                    current_pos = new_pos
            
        # 5. Periodic Relaxation
        # To escape shallow local minima and find deeper basins, we relax periodically.
        # This is the key to global search.
        if step_count % relax_freq == 0:
            try:
                e_relaxed, pos_relaxed = relax(current_pos, potential, rho)
                if e_relaxed < current_energy:
                    current_energy = e_relaxed
                    current_pos = pos_relaxed
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pos = current_pos.copy()
                else:
                    # If relaxation didn't improve, maybe we are stuck.
                    # Add a larger kick?
                    pass
            except Exception:
                # Relax failed, try a bigger kick
                noise = rng.normal(0, 0.5, size=current_pos.shape)
                current_pos = current_pos + noise
                
        # Adaptive noise
        # Reduce noise as T drops
        noise_scale = 0.1 * (T / T0 + 0.1)
        # Keep it from getting too small
        noise_scale = max(noise_scale, 0.01)

    # Final relaxation on the best position found
    try:
        e_final, pos_final = relax(best_pos, potential, rho)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final
    except Exception:
        pass

    _instance_count += 1
    return best_pos

# EVOLVE-BLOCK-END
