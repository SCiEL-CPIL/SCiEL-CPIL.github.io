"""
Strategy: Adaptive Sequential Multi-Start with Dynamic Time Allocation.
Instead of running parallel basin hopping chains which can waste time on poor basins 
and suffer from GIL contention, we perform a sequential sweep of diverse initial structures.
We use a strict time budget per relaxation attempt, allowing us to fit many more 
independent local minima evaluations into the 300s window. This exploits the fact that 
the Dzugutov potential has a rugged landscape but many local minima are close to global 
if the initial geometry is reasonable (PT-like).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 290.0  # Safety margin before 330s kill

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy_gradient(positions):
    """
    Compute energy and gradient for the Dzugutov potential.
    V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    
    n = len(positions)
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    # Vectorized distance calculation for speed
    # Create difference matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Mask for i < j to avoid double counting and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Get unique pairs
    i_idx, j_idx = np.where(mask)
    r_ij = r[i_idx, j_idx]
    diff_ij = diff[i_idx, j_idx]
    
    # Initialize forces/energies
    e_ij = np.zeros_like(r_ij)
    f_mag_ij = np.zeros_like(r_ij)
    
    # Case 1: r < a
    mask_a = r_ij < a
    if np.any(mask_a):
        r_a = r_ij[mask_a]
        # Avoid division by zero
        r_a_safe = np.where(r_a > 1e-10, r_a, 1e-10)
        
        exp_term = np.exp(c / (r_a_safe - a))
        r_pow = r_a_safe ** (-m)
        
        e_val = A * (r_pow - B) * exp_term
        # dV/dr = A * exp(c/(r-a)) * [ -m * r^(-m-1) - c * (r^-m - B) / (r-a)^2 ]
        term1 = -m * r_a_safe ** (-m - 1)
        term2 = -c * (r_a_safe ** (-m) - B) / ((r_a_safe - a) ** 2)
        f_val = A * exp_term * (term1 + term2)
        
        e_ij[mask_a] = e_val
        f_mag_ij[mask_a] = f_val

    # Case 2: a <= r < b
    mask_b = (r_ij >= a) & (r_ij < b)
    if np.any(mask_b):
        r_b = r_ij[mask_b]
        r_b_safe = np.where(r_b > 1e-10, r_b, 1e-10)
        
        exp_term = np.exp(d / (r_b_safe - b))
        
        e_val = B * exp_term
        # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
        f_val = B * exp_term * (-d / ((r_b_safe - b) ** 2))
        
        e_ij[mask_b] = e_val
        f_mag_ij[mask_b] = f_val

    energy = np.sum(e_ij)
    
    # Compute gradient
    # grad_i += f_mag * (pos_i - pos_j) / r
    # grad_j -= f_mag * (pos_i - pos_j) / r
    
    # r_vec = diff_ij / r_ij
    r_vec = diff_ij / np.where(r_ij > 1e-10, r_ij, 1e-10)
    
    # f_vec = f_mag * r_vec
    f_vec = f_mag_ij[:, np.newaxis] * r_vec
    
    # Accumulate gradients
    # np.add.at for efficient accumulation
    np.add.at(gradient, i_idx, f_vec)
    np.add.at(gradient, j_idx, -f_vec)

    return energy, gradient

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a structure based on icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)] # Center
    
    # Shell 1: 12 atoms
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
        
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
        
    # Shell 2: Add remaining atoms on a sphere with radius r2
    # For PT order, the next shell is often larger.
    # Let's use r2 = 1.9 * 1.7 ~ 3.23
    r2 = 1.9 * 1.7
    num_outer = n_atoms - 13
    if num_outer > 0:
        # Generate num_outer points on sphere r2
        outer_pts = []
        for _ in range(num_outer):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            outer_pts.append(v)
        positions.extend(outer_pts)
        
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.1):
    """
    Generate a random compact structure.
    """
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 5000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        # Fallback: simple grid or random without strict check
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def _local_minimize(positions, time_limit):
    """
    Perform local minimization using L-BFGS-B with a strict time limit.
    """
    start_time = time.time()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        return _dzugutov_energy_gradient(pos)[0]
        
    def jac(x):
        pos = x.reshape(-1, 3)
        return _dzugutov_energy_gradient(pos)[1].flatten()
        
    # Check time before starting
    if time.time() - start_time > time_limit:
        return objective(positions.flatten()), positions
        
    res = minimize(
        objective,
        positions.flatten(),
        jac=jac,
        method='L-BFGS-B',
        options={'maxiter': 200, 'maxls': 200, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    if res.fun < np.inf:
        return res.fun, res.x.reshape(-1, 3)
    else:
        return objective(positions.flatten()), positions

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    # Time management
    start_time = time.time()
    # We have ~45s per instance.
    # We will use an adaptive allocation: start with 1s per attempt, increase if fast, decrease if slow.
    current_time_per_attempt = 1.0
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate initial diverse starts
    starts = []
    
    # 1. PT-inspired
    try:
        pt_pos = _generate_icosahedron(size, rng)
        starts.append(pt_pos)
    except Exception:
        pass
        
    # 2. Random Compact
    starts.append(_random_compact(size, rng))
    
    # 3. Random Compact (different seed)
    starts.append(_random_compact(size, rng))
    
    # 4. Perturbed PT
    if starts:
        perturbed_pt = starts[0] + rng.normal(0, 0.2, starts[0].shape)
        starts.append(perturbed_pt)
        
    # 5. More random starts to fill the list
    for _ in range(10):
        starts.append(_random_compact(size, rng))
        
    # Shuffle starts to ensure diversity is sampled early
    rng.shuffle(starts)
    
    attempt_idx = 0
    
    while _get_remaining_time() > 2.0:
        if attempt_idx >= len(starts):
            # Generate new random starts
            new_start = _random_compact(size, rng)
            # Add perturbation
            new_start = new_start + rng.normal(0, 0.1, new_start.shape)
            starts.append(new_start)
            
        start_pos = starts[attempt_idx]
        attempt_idx += 1
        
        # Adjust time per attempt based on remaining time and attempts left (rough heuristic)
        # If we have lots of time, be more generous. If little time, be tighter.
        remaining = _get_remaining_time()
        # Heuristic: if remaining > 20s, allow up to 5s per attempt. If < 5s, allow 0.5s.
        dynamic_time = min(5.0, max(0.5, remaining / 10.0))
        
        try:
            energy, positions = _local_minimize(start_pos, dynamic_time)
            
            # Check minimum distance constraint
            min_dist = np.inf
            for i in range(size):
                for j in range(i+1, size):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist < 0.5:
                # Penalize or reject? Reject for now, try next
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        return starts[0]
        
    return best_positions
# EVOLVE-BLOCK-END
