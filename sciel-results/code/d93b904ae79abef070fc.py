import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # Safety margin

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_energy_and_grad(positions):
    """
    Compute Dzugutov potential energy and gradient.
    V(r) = A(r^-m - B)exp(c/(r - a)) for r < a
    V(r) = B exp(d/(r - b)) for r < b
    """
    A = 5.82
    B = 1.28
    a = 1.87
    b = 1.94
    c = 1.10
    d = 0.27
    m = 16
    
    n = positions.shape[0]
    # Calculate all pair distances
    # diff[i, j] = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Mask for i < j to avoid double counting and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_pairs = r[mask]
    diff_pairs = diff[mask]
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    if len(r_pairs) == 0:
        return energy, grad

    # Initialize force array for pairs
    forces = np.zeros_like(diff_pairs)
    
    # Region 1: r < a
    mask_a = r_pairs < a
    if np.any(mask_a):
        r_a = r_pairs[mask_a]
        # Avoid division by zero
        r_a_safe = np.maximum(r_a, 1e-6)
        
        # V(r) = A * (r^-m - B) * exp(c / (r - a))
        # dV/dr = A * [ -m*r^(-m-1)*exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2) ]
        
        exp_term = np.exp(c / (r_a_safe - a))
        r_inv_m = r_a_safe ** (-m)
        r_inv_m1 = r_a_safe ** (-m - 1)
        
        v = A * (r_inv_m - B) * exp_term
        dv_dr = A * (-m * r_inv_m1 * exp_term - (r_inv_m - B) * exp_term * c / (r_a_safe - a)**2)
        
        # Force on i due to j is -dV/dr * r_hat. 
        # r_hat = (r_i - r_j) / r. 
        # diff_pairs is r_i - r_j.
        # So force_vec = -dv_dr * diff_pairs / r
        # Note: dv_dr is dV/dr. The force vector F_ij acting on i is -dV/dr * (r_i - r_j)/r.
        # Wait, V depends on r = |r_i - r_j|.
        # dV/d(r_i) = dV/dr * dr/d(r_i) = dV/dr * (r_i - r_j)/r.
        # Force = -dV/d(r_i) = -dV/dr * (r_i - r_j)/r.
        
        force_mag = -dv_dr / r_a_safe
        forces[mask_a] = force_mag[:, np.newaxis] * diff_pairs[mask_a]
        energy += np.sum(v)

    # Region 2: a <= r < b
    mask_b = (r_pairs >= a) & (r_pairs < b)
    if np.any(mask_b):
        r_b = r_pairs[mask_b]
        r_b_safe = np.maximum(r_b, 1e-6)
        
        # V(r) = B * exp(d / (r - b))
        # dV/dr = B * exp(d / (r - b)) * (-d / (r - b)^2)
        
        exp_term = np.exp(d / (r_b_safe - b))
        v = B * exp_term
        dv_dr = -B * d * exp_term / (r_b_safe - b)**2
        
        force_mag = -dv_dr / r_b_safe
        forces[mask_b] = force_mag[:, np.newaxis] * diff_pairs[mask_b]
        energy += np.sum(v)
        
    # Region 3: r >= b
    # V(r) = 0, F(r) = 0. No contribution.

    # Sum forces into gradient
    # grad[i] = sum_{j != i} F_ij
    # Using the upper triangle mask, we need to add force to i and subtract from j.
    # It's easier to use a full matrix approach or scatter add.
    # Let's create a full force matrix
    F_mat = np.zeros((n, n, 3))
    # Scatter pair forces into full matrix
    # We need indices i, j for mask
    i_idx, j_idx = np.where(mask)
    F_mat[i_idx, j_idx] = forces
    # Force on i from j is F_ij. Force on j from i is -F_ij.
    # F_mat[i, j] is force on i due to j.
    # Total force on i is sum_j F_mat[i, j].
    grad = np.sum(F_mat, axis=1)
    
    return energy, grad

def _langevin_anneal(n, rng, t_start, t_end, steps, dt):
    """
    Perform Langevin Dynamics with annealing.
    """
    # Initial configuration: Random sphere
    R = 2.5 * (n / 4.0) ** (1.0/3.0)
    positions = np.zeros((n, 3))
    # Simple rejection sampling for initial pos
    count = 0
    while count < n:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            # Check min distance 0.5
            if count == 0 or np.all(np.linalg.norm(p - positions[:count], axis=1) > 0.5):
                positions[count] = p
                count += 1
    positions = positions[:n]
    
    velocities = np.zeros((n, 3))
    
    mass = 1.0
    k_b = 1.0
    
    best_energy = float('inf')
    best_positions = positions.copy()
    
    current_energy, current_grad = _compute_energy_and_grad(positions)
    if current_energy < best_energy:
        best_energy = current_energy
        best_positions = positions.copy()
        
    for step in range(steps):
        # Check time
        if _get_time_remaining() < 5.0:
            break
            
        # Temperature
        t = t_start * (t_end / t_start) ** (step / steps)
        
        # Langevin step (Bussi-Parrinello style or simple Euler-Maruyama for simplicity/speed)
        # Simple Langevin:
        # v_{n+1} = v_n * exp(-gamma*dt) + sqrt(2*k_b*T*gamma*dt/mass) * N(0,1) + (-grad/mass) * dt * (1 - exp(-gamma*dt))
        # For simplicity and stability with stiff potentials, we use a smaller dt and explicit Euler with friction.
        # v_{n+1} = v_n + (-grad/mass - gamma*v_n) * dt + sqrt(2*k_b*T*gamma*dt/mass) * N(0,1)
        # r_{n+1} = r_n + v_{n+1} * dt
        
        gamma = 1.0 # Friction coefficient
        noise_scale = np.sqrt(2 * k_b * t * gamma * dt / mass)
        
        # Update velocities
        velocities = velocities * (1 - gamma * dt) - current_grad * dt / mass + noise_scale * rng.standard_normal(n, 3)
        
        # Update positions
        positions = positions + velocities * dt
        
        # Recompute energy and gradient
        new_energy, new_grad = _compute_energy_and_grad(positions)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_positions = positions.copy()
            
        # Check for NaNs or explosions
        if not np.isfinite(new_energy):
            break
            
        current_energy = new_energy
        current_grad = new_grad
        
    # Final relaxation
    try:
        final_energy, final_pos = relax(best_positions, 'dzugutov', None)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    except Exception:
        pass
        
    return best_positions, best_energy

def _worker(args):
    n, seed_idx, t_start, t_end, steps, dt = args
    rng = np.random.default_rng(seed_idx)
    try:
        pos, energy = _langevin_anneal(n, rng, t_start, t_end, steps, dt)
        return pos, energy
    except Exception as e:
        # Fallback to random
        R = 2.5 * (n / 4.0) ** (1.0/3.0)
        positions = np.zeros((n, 3))
        count = 0
        while count < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                if count == 0 or np.all(np.linalg.norm(p - positions[:count], axis=1) > 0.5):
                    positions[count] = p
                    count += 1
        try:
            e, p = relax(positions, 'dzugutov', None)
            return p, e
        except:
            return positions, float('inf')

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    
    # Strategy: Run multiple Langevin annealing runs in parallel with different parameters
    # We have 300s total. Let's allocate ~60s per run, running 4-5 runs in parallel?
    # No, we can run them sequentially or in parallel. Parallel is better for wall-clock.
    # Max 4 threads.
    
    # Parameters for annealing
    # T_start high enough to explore, T_end low to settle.
    # Steps: enough to cover the time budget.
    # dt: small enough for stability. Dzugutov is stiff.
    
    runs = []
    # Run 1: Standard
    runs.append((n, 1, 5.0, 0.1, 5000, 0.01))
    # Run 2: Higher start temp
    runs.append((n, 2, 10.0, 0.1, 5000, 0.01))
    # Run 3: Slower cooling
    runs.append((n, 3, 5.0, 0.01, 8000, 0.01))
    # Run 4: Different seed, standard
    runs.append((n, 4, 5.0, 0.1, 5000, 0.01))
    
    best_pos = None
    best_energy = float('inf')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, r) for r in runs]
        for f in concurrent.futures.as_completed(futures):
            pos, energy = f.result()
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
                
    # If we have time left, do a quick basin hopping on the best result?
    # Or just return the best.
    # Let's add a short refinement loop if time permits.
    if best_pos is not None and _get_time_remaining() > 10.0:
        current_pos = best_pos
        current_energy = best_energy
        rng = np.random.default_rng(n + 100)
        
        # Simple basin hopping on top of LD result
        step_size = 0.5
        for i in range(20):
            if _get_time_remaining() < 5.0:
                break
            new_pos = current_pos + rng.normal(0, step_size, current_pos.shape)
            try:
                e, p = relax(new_pos, 'dzugutov', None)
                if e < current_energy:
                    current_energy = e
                    current_pos = p
                    step_size *= 0.9
                else:
                    step_size *= 1.1
                    if step_size > 2.0: step_size = 2.0
            except:
                pass
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos

    return best_pos
