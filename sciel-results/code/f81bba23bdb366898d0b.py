import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 140.0  # Safety margin before 150s kill

def _energy_and_grad(pos):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    
    pos: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the unconstrained gradient w.r.t. pos.
    Energy is sum_{i<j} log(1 / |x_i - x_j|).
    """
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    
    # Avoid division by zero for self-pairs
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Clip to avoid log(0) if points are extremely close
    d = np.clip(d, 1e-12, None)
    energy = -np.sum(np.log(d))
    
    # Gradient computation:
    # grad_i = sum_{j != i} (p_j - p_i) / |p_i - p_j|^2
    #        = sum_{j != i} p_j / d_ij^2 - p_i * sum_{j != i} 1/d_ij^2
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # sum_j p_j / d_ij^2 for each i
    sum_pj_over_d2 = inv_d2 @ pos
    # sum_j 1/d_ij^2 for each i
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    grad = sum_pj_over_d2 - pos * sum_inv_d2[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(pos):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pos / norms

def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts

def _penalty_objective(pos, lambda_pen=1e5):
    """
    Objective function with penalty for sphere constraint.
    pos: (N, 3) array.
    """
    pos_flat = pos.reshape(-1)
    pos_reshaped = pos_flat.reshape(N, 3)
    
    energy, grad_energy = _energy_and_grad(pos_reshaped)
    
    # Penalty term: sum (||x_i||^2 - 1)^2
    norms_sq = np.sum(pos_reshaped * pos_reshaped, axis=1)
    penalty = np.sum((norms_sq - 1.0) ** 2)
    
    # Gradient of penalty: d/dx_i (||x_i||^2 - 1)^2 = 4 x_i (||x_i||^2 - 1)
    grad_penalty = 4.0 * pos_reshaped * (norms_sq - 1.0)[:, np.newaxis]
    
    total_energy = energy + lambda_pen * penalty
    total_grad = grad_energy + lambda_pen * grad_penalty
    
    return total_energy, total_grad.flatten()

def _optimize_lbfgs(pos_init, time_budget):
    """
    Optimize a single starting configuration using L-BFGS-B with penalty method.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos_init.copy())
    
    # Initial energy check
    current_energy, _ = _energy_and_grad(pos)
    best_pos = pos.copy()
    best_energy = current_energy
    
    # We use L-BFGS-B on the flat vector
    x0 = pos.flatten()
    
    # Callback to check time and update best
    def callback(xk):
        nonlocal best_pos, best_energy
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return True # Stop optimization
        
        pos_k = _project_to_sphere(xk.reshape(N, 3))
        e_k, _ = _energy_and_grad(pos_k)
        if e_k < best_energy:
            best_energy = e_k
            best_pos = pos_k.copy()
        return False

    try:
        res = minimize(
            _penalty_objective, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
        )
        
        # Final check after optimization
        if res.x is not None:
            pos_final = _project_to_sphere(res.x.reshape(N, 3))
            e_final, _ = _energy_and_grad(pos_final)
            if e_final < best_energy:
                best_energy = e_final
                best_pos = pos_final.copy()
                
    except Exception:
        pass
        
    return _project_to_sphere(best_pos), best_energy

def _simulated_annealing(pts, max_time, rng, T_start=0.1, T_min=1e-6, cooling=0.99):
    """
    Simulated annealing on the sphere.
    Uses tangent-space moves.
    """
    start_time = time.time()
    current = _project_to_sphere(pts.copy())
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    T = T_start
    
    while time.time() - start_time < max_time and T > T_min:
        # Check global deadline
        if time.time() > DEADLINE - 5:
            break
            
        # Move a small batch of points
        batch_size = rng.integers(1, 4)
        indices = rng.choice(N, size=batch_size, replace=False)
        
        new_pts = current.copy()
        for idx in indices:
            # Random direction in tangent space
            rand_dir = rng.standard_normal(3)
            rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-15:
                rand_dir /= norm
            
            # Move along random direction
            # Step size scales with T
            step_size = T * 0.5
            new_pt = current[idx].copy()
            new_pt += step_size * rand_dir
            new_pt = new_pt / np.linalg.norm(new_pt)
            new_pts[idx] = new_pt
        
        new_e, _ = _energy_and_grad(new_pts)
        delta_e = new_e - current_e
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-12)):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        
        T *= cooling
        
    return best, best_e

def _parallel_tempering(starts, time_budget, rng):
    """
    Parallel tempering (Replica Exchange Monte Carlo) combined with local optimization.
    """
    n_replicas = len(starts)
    replicas = [s.copy() for s in starts]
    # Sort replicas by energy initially
    energies = []
    for r in replicas:
        e, _ = _energy_and_grad(r)
        energies.append(e)
    
    # Temperatures: Higher T for "worse" replicas to help them explore
    # Base T
    T_base = 0.05
    # T_i = T_base * (1 + 0.5 * i) maybe?
    # Let's use a geometric progression of temperatures
    temps = [T_base * (1.5 ** i) for i in range(n_replicas)]
    # Sort replicas so best has lowest T
    sort_idx = np.argsort(energies)
    replicas = [replicas[i] for i in sort_idx]
    energies = [energies[i] for i in sort_idx]
    temps = [temps[i] for i in sort_idx]
    
    best_overall = replicas[0].copy()
    best_e_overall = energies[0]
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE - 5:
            break
            
        # 1. Local optimization (L-BFGS) for a short time on each replica
        # To save time, only optimize every few steps or if time permits
        # Let's optimize replica 0 (best) and replica 1 (second best) more often
        # Actually, just optimize all briefly? No, L-BFGS is expensive.
        # Let's do a single L-BFGS step on the best replica if time is low, or just SA.
        # The prompt suggests combining. Let's do SA moves first, then occasional L-BFGS.
        
        # SA Step for all replicas
        for i in range(n_replicas):
            # SA move
            current = replicas[i]
            T_i = temps[i]
            
            # Small batch move
            batch_size = rng.integers(1, 3)
            indices = rng.choice(N, size=batch_size, replace=False)
            
            new_pts = current.copy()
            for idx in indices:
                rand_dir = rng.standard_normal(3)
                rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
                norm = np.linalg.norm(rand_dir)
                if norm > 1e-15:
                    rand_dir /= norm
                
                step_size = T_i * 0.5
                new_pt = current[idx].copy()
                new_pt += step_size * rand_dir
                new_pt = new_pt / np.linalg.norm(new_pt)
                new_pts[idx] = new_pt
            
            new_e, _ = _energy_and_grad(new_pts)
            delta_e = new_e - energies[i]
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T_i, 1e-12)):
                replicas[i] = new_pts
                energies[i] = new_e
                if new_e < best_e_overall:
                    best_e_overall = new_e
                    best_overall = new_pts.copy()
            
            # Cool down
            temps[i] = max(temps[i] * 0.995, T_base * 0.5)
            # Reheat if too cold?
            if temps[i] < T_base * 0.5:
                temps[i] = T_base
        
        # 2. Exchange proposals (Replica Exchange)
        # Propose exchange between adjacent replicas i and i+1
        # Accept if (E_i T_{i+1} + E_{i+1} T_i) < (E_{i+1} T_i + E_i T_{i+1})?
        # Delta = E_i/T_i + E_{i+1}/T_{i+1} - (E_{i+1}/T_i + E_i/T_{i+1})
        # If Delta < 0 or accept with prob exp(-Delta)
        
        for i in range(n_replicas - 1):
            T1 = temps[i]
            T2 = temps[i+1]
            E1 = energies[i]
            E2 = energies[i+1]
            
            # Delta = (E1/T1 + E2/T2) - (E2/T1 + E1/T2)
            # = E1(1/T1 - 1/T2) + E2(1/T2 - 1/T1)
            # = (1/T1 - 1/T2) * (E1 - E2)
            # Note: T1 < T2 usually, so 1/T1 > 1/T2, so (1/T1 - 1/T2) > 0
            # If E1 < E2, Delta > 0 -> Unlikely to accept swap (Good config stays at low T)
            # If E1 > E2, Delta < 0 -> Likely to accept swap (Bad config moves to high T)
            
            delta = (1.0/T1 - 1.0/T2) * (E1 - E2)
            
            if delta < 0 or rng.random() < math.exp(-delta):
                # Swap
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                energies[i], energies[i+1] = energies[i+1], energies[i]
                temps[i], temps[i+1] = temps[i+1], temps[i]
                
                if energies[0] < best_e_overall:
                    best_e_overall = energies[0]
                    best_overall = replicas[0].copy()

    return best_overall, best_e_overall

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate starting configurations
    starts = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(_project_to_sphere(pos_fib))
    
    # 2. Perturbed Fibonacci
    pos_fib_pert = pos_fib + rng.normal(0, 0.01, size=(N, 3))
    starts.append(_project_to_sphere(pos_fib_pert))
    
    # 3. Random
    for _ in range(2):
        pos_rand = rng.normal(0, 1, size=(N, 3))
        starts.append(_project_to_sphere(pos_rand))

    best_pos = starts[0]
    best_energy = float('inf')
    
    # Phase 1: L-BFGS optimization on diverse starts
    # Allocate time
    total_opt_time = 60.0
    time_per_start = [
        total_opt_time * 0.40, # Fib
        total_opt_time * 0.30, # Fib Pert
        total_opt_time * 0.15, # Rand 1
        total_opt_time * 0.15, # Rand 2
    ]
    
    for idx, start_pos in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30: # Reserve 30s for PT
            break
            
        alloc_time = min(time_per_start[idx], DEADLINE - 30 - elapsed)
        if alloc_time < 2.0:
            continue
            
        pos, energy = _optimize_lbfgs(start_pos, alloc_time)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
    # Phase 2: Parallel Tempering / Simulated Annealing
    # Use the best result from L-BFGS and some other good starts
    # Create a set of replicas for PT
    pt_starts = [best_pos]
    
    # Add some perturbed versions of the best
    for _ in range(3):
        perturbed = best_pos + rng.normal(0, 0.02, size=(N, 3))
        pt_starts.append(_project_to_sphere(perturbed))
        
    # Add the original fib if it wasn't the best
    e_fib, _ = _energy_and_grad(starts[0])
    if e_fib < best_energy + 1.0:
        pt_starts.append(starts[0])
        
    # Run PT
    pt_time_budget = min(30.0, DEADLINE - 10 - (time.time() - start_time))
    if pt_time_budget > 2.0:
        pt_pos, pt_energy = _parallel_tempering(pt_starts, pt_time_budget, rng)
        if pt_energy < best_energy:
            best_energy = pt_energy
            best_pos = pt_pos.copy()
            
    # Phase 3: Final Polish
    # Use remaining time for high-precision L-BFGS on the best solution
    elapsed = time.time() - start_time
    remaining = DEADLINE - 5 - elapsed
    if remaining > 1.0:
        # Final polish
        final_pos, final_energy = _optimize_lbfgs(best_pos, remaining)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos.copy()

    # Final validation
    best_pos = _project_to_sphere(best_pos)
    
    if not np.all(np.isfinite(best_pos)):
        # Fallback
        best_pos = starts[0]
        
    # Check minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Perturb and retry quickly
        perturbed = best_pos + rng.normal(0, 0.01, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        pos, energy = _optimize_lbfgs(perturbed, 2.0)
        if energy < best_energy:
            best_pos = pos
            best_energy = energy
            best_pos = _project_to_sphere(best_pos)

    return best_pos
