"""Combined Morse cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy_grad(pos, rho):
    """
    Compute Morse potential energy and gradient vectorized.
    pos: (n, 3) array
    rho: range parameter
    Returns: (energy, gradient_flat)
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros(pos.shape)
    
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # (n, n, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=-1)) # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # x_ij = rho * (1 - r_ij)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    
    # Energy term: e^x * (e^x - 2)
    # We only sum i < j, so divide by 2
    e_mat = ex * (ex - 2.0)
    np.fill_diagonal(e_mat, 0.0)
    energy = 0.5 * np.sum(e_mat)
    
    # Gradient term
    # dE/dx_ij = 2 * e^x * (e^x - 1)
    de_dx = 2.0 * ex * (ex - 1.0)
    np.fill_diagonal(de_dx, 0.0)
    
    # d x_ij / d r_ij = -rho
    # d r_ij / d r_i = (r_i - r_j) / r_ij
    # So dE/d r_i = sum_j (dE/dx_ij * -rho * (r_i - r_j) / r_ij)
    
    # Factor = -rho * de_dx / dists
    factor = -rho * de_dx / dists
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * diff_ij
    # diff is (n, n, 3), factor is (n, n)
    # We need to sum over j (axis 1)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad.flatten()

def _lattice_init(n, rng):
    """
    Generate an initial configuration based on a cubic lattice.
    """
    # Approximate side length
    side = int(np.ceil(n ** (1.0 / 3.0)))
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(pts) < n:
                    pts.append([i, j, k])
    
    if len(pts) < n:
        # Fill remaining with random points near the lattice
        for _ in range(n - len(pts)):
            p = rng.uniform(-0.5, side + 0.5, 3)
            pts.append(p)
    
    pts = np.array(pts[:n], dtype=float)
    # Center
    pts -= pts.mean(axis=0)
    # Small perturbation to break symmetry and help optimization
    pts += rng.normal(0, 0.01, pts.shape)
    return pts

def _random_packed_init(n, rng):
    """
    Generate a random packed configuration with minimum distance 0.6.
    """
    radius = n ** (1.0 / 3.0) * 0.8
    pts = []
    min_dist = 0.6
    max_attempts = 20000
    attempts = 0
    
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        
        valid = True
        for q in pts:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        
        if valid:
            pts.append(p)
        attempts += 1
    
    # If we couldn't place all, just add random points (will be high energy but valid start for optimizer)
    while len(pts) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        pts.append(p)
        
    pts = np.array(pts[:n], dtype=float)
    pts -= pts.mean(axis=0)
    return pts

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    """
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance.get("potential", "morse")
    
    # Fixed seed for reproducibility per instance
    seed = hash((n, rho)) % (2**32)
    rng = np.random.default_rng(seed)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin for safety
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Lattice-based
    inits.append(_lattice_init(n, rng))
    
    # 2. Random packed
    inits.append(_random_packed_init(n, rng))
    
    # 3. Another lattice variant (different perturbation)
    inits.append(_lattice_init(n, rng))
    
    # 4. Another random variant
    inits.append(_random_packed_init(n, rng))
    
    # 5. Maybe a slightly larger radius random
    radius = n ** (1.0 / 3.0) * 1.2
    pts = []
    for _ in range(n):
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            pts.append(p)
    if len(pts) == n:
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        inits.append(pts)
    
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        # Check time budget for this attempt
        # Each L-BFGS-B run on N=80 might take a few seconds.
        
        try:
            # Use the provided relax function first as a quick local minimizer?
            # The prompt says relax and morse_energy_gradient may be called.
            # relax is likely a robust local minimizer. Let's use it to polish the start.
            # However, L-BFGS-B with our vectorized gradient is likely faster and more controllable.
            
            def fun_and_grad(x):
                pos = x.reshape(-1, 3)
                e, g = _morse_energy_grad(pos, rho)
                return e, g
            
            x0 = init_pos.flatten()
            
            # L-BFGS-B optimization
            result = minimize(
                fun_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 5000,
                    'disp': False
                }
            )
            
            if result.fun < best_energy:
                pos_opt = result.x.reshape(-1, 3)
                
                # Verify minimum distance constraint (>= 0.5)
                diff = pos_opt[:, np.newaxis, :] - pos_opt[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff * diff, axis=-1))
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                
                if min_d >= 0.5:
                    best_energy = result.fun
                    best_positions = pos_opt
                else:
                    # If constraint violated, we might still keep it if it's the best, 
                    # but the evaluator will reject it. 
                    # Better to try to fix or discard.
                    # For now, discard if invalid.
                    pass
                    
        except Exception as e:
            # If L-BFGS fails, fall back to relax
            try:
                energy_relaxed, pos_relaxed = relax(init_pos.copy(), potential, rho)
                # Check constraint
                diff = pos_relaxed[:, np.newaxis, :] - pos_relaxed[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff * diff, axis=-1))
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                
                if min_d >= 0.5 and energy_relaxed < best_energy:
                    best_energy = energy_relaxed
                    best_positions = pos_relaxed
            except Exception:
                pass
    
    # If we still don't have a valid position, use the last relaxed one or generate a valid one
    if best_positions is None:
        # Fallback: use the first init relaxed
        try:
            energy, positions = relax(inits[0], potential, rho)
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                best_positions = positions
            else:
                # Last resort: return a simple lattice that is definitely valid
                # (Lattice points are 1.0 apart, so min_dist=1.0 >= 0.5)
                side = int(np.ceil(n ** (1.0 / 3.0)))
                pts = []
                for i in range(side):
                    for j in range(side):
                        for k in range(side):
                            if len(pts) < n:
                                pts.append([i, j, k])
                pts = np.array(pts[:n], dtype=float)
                pts -= pts.mean(axis=0)
                best_positions = pts
        except Exception:
            # Absolute fallback
            side = int(np.ceil(n ** (1.0 / 3.0)))
            pts = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        if len(pts) < n:
                            pts.append([i, j, k])
            pts = np.array(pts[:n], dtype=float)
            pts -= pts.mean(axis=0)
            best_positions = pts
            
    return best_positions
# EVOLVE-BLOCK-END
