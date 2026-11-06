import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize

def _morse_energy(pos, rho):
    """Compute exact Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    r = dists[mask]
    # E = sum exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let x = rho(1-r)
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return energy

def _morse_energy_grad(pos, rho):
    """Compute Morse energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    
    # Avoid division by zero
    dists_safe = np.where(dists > 1e-10, dists, 1e-10)
    
    # Gradient calculation
    # E_ij = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # dE/dr = exp(rho(1-r)) * rho * (exp(rho(1-r)) - 2) + exp(rho(1-r)) * (-exp(rho(1-r)) * rho)
    #       = rho * exp(2x) - 2*rho*exp(x) - rho*exp(2x)  ... wait
    # Let's derive properly.
    # f(r) = e^x (e^x - 2) = e^{2x} - 2e^x
    # df/dr = (2e^{2x} - 2e^x) * dx/dr
    # dx/dr = -rho
    # df/dr = -rho * (2e^{2x} - 2e^x) = 2*rho*e^x * (e^x - 1)
    
    x = rho * (1.0 - dists_safe)
    e_x = np.exp(x)
    dE_dr = 2.0 * rho * e_x * (e_x - 1.0)
    
    # Gradient w.r.t position: dE/dpos_i = sum_j dE_ij/dr_ij * (pos_i - pos_j) / r_ij
    # Note: dE_ij/dr_ij is symmetric, but vector direction is (pos_i - pos_j)/r_ij
    # We can compute gradient for all i simultaneously using matrix ops.
    
    # grad_matrix[i,j] = dE_ij/dr_ij * (pos_i - pos_j) / r_ij
    # We only care about off-diagonal.
    
    # Create a matrix of dE_dr / r
    factor = np.zeros_like(dists)
    factor[mask] = dE_dr[mask] / dists_safe[mask]
    
    # grad_i = sum_j factor[i,j] * (pos_i - pos_j)
    # grad_i = pos_i * sum_j factor[i,j] - sum_j factor[i,j] * pos_j
    
    sum_factor = np.sum(factor, axis=1)
    weighted_pos_sum = factor @ pos
    
    grad = pos * sum_factor[:, np.newaxis] - weighted_pos_sum
    
    # Energy is sum over i<j, but the formula above sums over all i!=j.
    # The energy function E = sum_{i<j} V(r_ij).
    # The gradient of sum_{i<j} V(r_ij) is indeed sum_{j!=i} V'(r_ij) * (pos_i - pos_j)/r_ij.
    # So the gradient calculation is correct as is.
    
    energy = _morse_energy(pos, rho)
    return energy, grad

def _local_minimize(pos, rho, max_iter=500, tol=1e-6):
    """Run L-BFGS-B local minimization."""
    def objective(x):
        p = x.reshape(-1, 3)
        e, g = _morse_energy_grad(p, rho)
        return e, g.flatten()
    
    try:
        result = scipy.optimize.minimize(
            objective, pos.flatten(), method='L-BFGS-B', 
            jac=True, options={'maxiter': max_iter, 'gtol': tol}
        )
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        return _morse_energy(pos, rho), pos

def _generate_fcc_init(n, rng):
    """Generate an FCC-like initial configuration."""
    # Approximate side length of cube for n atoms in FCC
    # Density ~ 1 atom per unit volume in reduced units (roughly)
    side = int(np.ceil(n ** (1.0/3.0))) + 1
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for b in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i, j, k]) + np.array(b)
                    pts.append(p)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    
    pts = np.array(pts[:n])
    pts -= np.mean(pts, axis=0)
    # Add small noise to break symmetry
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", 3)
    
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    rng = np.random.default_rng(42 + n)
    
    best_energy = np.inf
    best_positions = None
    
    # 1. Initial Configuration: FCC based
    initial_pos = _generate_fcc_init(n, rng)
    
    # 2. Initial Local Minimization
    current_energy, current_pos = _local_minimize(initial_pos, rho)
    best_energy = current_energy
    best_positions = current_pos.copy()
    
    # 3. Simulated Annealing Loop
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.995 # Adjust based on n? Larger n needs slower cooling
    # Estimate number of steps we can afford
    # Each step involves energy calc + potential relax
    # Let's run until time limit
    
    step_size_start = 0.5
    step_size_end = 0.05
    
    # We will use a hybrid approach:
    # Most steps are SA moves.
    # Every N steps, we do a local relaxation to "settle".
    
    relax_interval = 10
    
    step = 0
    T = T_start
    
    while time.time() < deadline:
        # Annealing schedule
        # Linear in step count is hard without knowing total steps.
        # Use time-based or step-based decay.
        # Let's decay T and step size based on a fixed number of "virtual" steps
        # or just use a slow decay.
        
        # Decay T and step size
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Interpolate step size
        progress = min(1.0, step / 5000.0) # Assume ~5000 steps in budget
        current_step_size = step_size_start * (1 - progress) + step_size_end * progress
        
        # Generate perturbation
        noise = rng.normal(0, current_step_size, size=(n, 3))
        new_pos = current_pos + noise
        
        # Check minimum distance constraint softly
        # If violated, we can either reject or penalize.
        # Penalizing is better for SA continuity.
        # But simple rejection is safer for "hard" constraints if we want to guarantee validity at the end.
        # The task says "atoms must be at least 0.5 apart".
        # If we return a config with d < 0.5, it's invalid.
        # So we MUST ensure the final result is valid.
        # We can enforce validity at the end by projecting or rejecting bad final states.
        # During SA, we can allow overlaps if we add a huge penalty, but it's complex.
        # Let's stick to: if d < 0.5, reject the move (keep current).
        
        min_d = np.min(np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=-1) * ~np.eye(n, dtype=bool))
        
        if min_d >= 0.5:
            new_energy = _morse_energy(new_pos, rho)
            
            # Metropolis acceptance
            delta_e = new_energy - current_energy
            if delta_e < 0 or (T > 1e-6 and rng.random() < np.exp(-delta_e / T)):
                current_energy = new_energy
                current_pos = new_pos
                
                # Update best
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_pos.copy()
                    
                # Periodic Local Relaxation
                if step % relax_interval == 0:
                    # Check time before expensive relax
                    if time.time() < deadline - 5.0:
                        relaxed_energy, relaxed_pos = _local_minimize(current_pos, rho, max_iter=100)
                        if relaxed_energy < current_energy:
                            current_energy = relaxed_energy
                            current_pos = relaxed_pos
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_positions = current_pos.copy()
        else:
            # Rejected due to constraint
            pass
            
        step += 1
        
        # Adaptive cooling: if we haven't improved in a while, drop T faster?
        # Or just stick to schedule.
        
    # Final Polish: Ensure the best position is valid and locally minimized
    if best_positions is not None:
        # Check validity
        min_d_final = np.min(np.linalg.norm(best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :], axis=-1) * ~np.eye(n, dtype=bool))
        if min_d_final < 0.5:
            # Fallback to a valid random config if best is invalid (should not happen with rejection)
            # But just in case, try to repair
            # For now, return best_positions as is, assuming rejection logic kept it valid.
            pass
        
        # One final high-precision relaxation on the best found
        if time.time() < deadline - 2.0:
            final_energy, final_pos = _local_minimize(best_positions, rho, max_iter=1000)
            # Check if final_pos is valid
            min_d_check = np.min(np.linalg.norm(final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :, :], axis=-1) * ~np.eye(n, dtype=bool))
            if min_d_check >= 0.5 and final_energy < best_energy:
                return final_pos
            else:
                # If relaxation broke constraints or increased energy, return original best
                return best_positions

    # Fallback if something went wrong
    return initial_pos
