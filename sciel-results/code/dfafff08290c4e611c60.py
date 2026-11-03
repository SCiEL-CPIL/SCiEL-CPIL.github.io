"""Rewritten program for minimal logarithmic energy of 100 points on the sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere, handling zero vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-14, None)
    return pts / norms

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and tangent-space gradient.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient of E w.r.t. x_k (ambient):
    # dE/dx_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    #         = sum_{j != k} (x_k - x_j) / dist_sq[k,j]
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_ambient[k] = sum_j W[k,j] * (x_k - x_j) = x_k * sum_j W[k,j] - sum_j W[k,j]*x_j
    grad_ambient = W @ pts * (-1.0) + W.sum(axis=1, keepdims=True) * pts
    # Project to tangent space
    dot = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_ambient - dot * pts
    
    return energy, grad_tangent

def _fast_energy(pts):
    """Quick energy computation for sorting/comparison."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.1)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _precise_energy(pts):
    """High-precision energy using math.fsum."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _generate_fibonacci(n, offset=0.0):
    """Fibonacci sphere point distribution."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_sobol(n, seed=42):
    """Sobol sequence mapped to sphere."""
    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    samples = sampler.random(n)
    samples = samples - 0.5
    return _normalize(samples)

def _generate_random(n, rng):
    """Random points on sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _generate_lattice(n):
    """Generate points using a more structured lattice approach."""
    # Use a combination of rings with golden angle spacing
    pts = []
    # Place points in rings with varying z
    n_rings = 10
    pts_per_ring = n // n_rings
    remaining = n % n_rings
    
    ring_indices = 0
    for ring in range(n_rings):
        count = pts_per_ring + (1 if ring < remaining else 0)
        z = 1.0 - 2.0 * (ring + 0.5) / n_rings
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        for k in range(count):
            theta = 2.0 * np.pi * k / count + ring * np.pi / n_rings
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
        ring_indices += count
    
    pts = np.array(pts[:n])
    return _normalize(pts)

def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    """
    Gradient flow with annealing and noise for exploration.
    """
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _energy_and_grad(current)
        
        # Add noise that decreases over time
        noise_scale = 0.01 * (1.0 - progress) ** 2
        if noise_scale > 1e-6:
            noise = rng.standard_normal((n, 3)) * noise_scale
        else:
            noise = np.zeros((n, 3))
        
        current = current - lr * g + noise
        current = _normalize(current)
        
        e_new = _fast_energy(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()
    
    return best, best_energy

def _optimize_lbfgs(pts, deadline, max_iter=3000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 5,
                'ftol': ftol,
                'gtol': gtol,
            }
        )
        new_pts = _normalize(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts

def _local_rearrange(pts, deadline, n_trials=20):
    """
    Try local rearrangements: swap pairs of points with nearby neighbors
    to escape local minima.
    """
    n = pts.shape[0]
    best = pts.copy()
    best_energy = _precise_energy(pts)
    rng = np.random.default_rng(999)
    
    # Compute distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Find nearest neighbors
    nn_idx = np.argmin(dist_sq, axis=1)
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        candidate = best.copy()
        
        # Pick a random pair of nearby points and try swapping their positions
        # with their nearest neighbors
        i = rng.integers(n)
        j = nn_idx[i]
        
        if i == j:
            continue
        
        # Try moving point i slightly toward point j's antipode
        direction = -candidate[i]  # antipodal direction
        candidate[i] = candidate[i] + 0.05 * direction
        candidate[i] = _normalize(candidate[i].reshape(1, 3))[0]
        
        e_new = _precise_energy(candidate)
        if e_new < best_energy:
            best = candidate
            best_energy = e_new
    
    return best, best_energy

def _spectral_init(L=20, rng=None):
    """
    Generate initial points using a spectral/spherical harmonic approach.
    Create a potential from random spherical harmonics and find its critical points.
    This is a heuristic initialization, not a full spectral optimization.
    """
    if rng is None:
        rng = np.random.default_rng(777)
    
    # Generate random coefficients for spherical harmonics up to degree L
    # Use a simple approach: random directions with varying weights
    n_coeff = L * (L + 1)
    coeffs = rng.standard_normal((n_coeff, 3))
    
    # Create a grid of points and evaluate a "potential"
    # Use Fibonacci grid as evaluation points
    n_grid = 500
    grid = _generate_fibonacci(n_grid)
    
    # Compute "potential" at each grid point
    # V(x) = sum_k |x - c_k|^2 * w_k  (simplified)
    weights = np.abs(rng.standard_normal(n_coeff)) + 0.1
    
    # For each grid point, compute distance to each coefficient direction
    # and find local minima
    best_pts = []
    for _ in range(N):
        # Find grid point with lowest "potential"
        dists = np.linalg.norm(grid[:, np.newaxis, :] - coeffs[np.newaxis, :, :], axis=2)
        potential = np.sum(dists * weights[np.newaxis, :], axis=1)
        
        idx = np.argmin(potential)
        pt = grid[idx].copy()
        
        # Add small random perturbation to avoid duplicates
        pt = pt + rng.standard_normal(3) * 0.01
        pt = _normalize(pt.reshape(1, 3))[0]
        
        # Check if this point is too close to existing ones
        if len(best_pts) > 0:
            existing = np.array(best_pts)
            min_dist = np.min(np.linalg.norm(existing - pt, axis=1))
            if min_dist < 0.05:
                continue
        
        best_pts.append(pt)
    
    # If we didn't get enough points, fill with Fibonacci
    while len(best_pts) < N:
        best_pts.append(_generate_fibonacci(1)[0])
    
    return np.array(best_pts[:N])

def _differential_evolution_step(pts, rng, n_trials=50, deadline=None):
    """
    Simple differential-evolution-like local search.
    """
    n = pts.shape[0]
    best = pts.copy()
    best_energy = _precise_energy(pts)
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        candidate = best.copy()
        
        # Pick three random points
        i, j, k = rng.choice(n, size=3, replace=False)
        
        # Create perturbation from difference
        perturbation = (candidate[j] - candidate[k]) * 0.1
        candidate[i] = candidate[i] + perturbation
        candidate[i] = _normalize(candidate[i].reshape(1, 3))[0]
        
        e_new = _precise_energy(candidate)
        if e_new < best_energy:
            best = candidate
            best_energy = e_new
    
    return best, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral (multiple offsets)
    for offset in [0.0, 0.1, 0.3, 0.5]:
        initializations.append(_generate_fibonacci(N, offset))
    
    # 2. Sobol Sequences
    for seed in [1, 2, 3, 7, 13]:
        initializations.append(_generate_sobol(N, seed=seed))
    
    # 3. Random Points
    for _ in range(10):
        initializations.append(_generate_random(N, rng))
    
    # 4. Lattice
    initializations.append(_generate_lattice(N))
    
    # 5. Spectral initialization
    initializations.append(_spectral_init(L=20, rng=rng))
    
    # 6. Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for scale in [0.02, 0.05, 0.1, 0.2]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))
    
    # Sort initializations by quick energy
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    # Main optimization loop with time budget
    main_deadline = start_time + 90.0
    
    for e_quick, init_pts in scored_inits:
        if time.time() > main_deadline:
            break
        
        # Phase 1: Gradient flow to escape local minima
        gf_pts, gf_energy = _gradient_flow(
            init_pts, rng, n_steps=300, 
            lr_start=0.02, lr_end=0.0005, 
            deadline=main_deadline
        )
        
        # Phase 2: L-BFGS-B optimization
        opt_pts = _optimize_lbfgs(gf_pts, main_deadline, max_iter=2000, ftol=1e-14, gtol=1e-10)
        e_precise = _precise_energy(opt_pts)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = opt_pts
        
        # Phase 3: Local rearrangement
        if time.time() < main_deadline:
            lr_pts, lr_energy = _local_rearrange(opt_pts, main_deadline, n_trials=10)
            if lr_energy < best_energy:
                best_energy = lr_energy
                best_pts = lr_pts
    
    if best_pts is None:
        best_pts = _generate_fibonacci(N)
        best_energy = _precise_energy(best_pts)
    
    # Final high-precision polish
    if time.time() < deadline:
        # Multiple restarts with tiny perturbations
        polish_rng = np.random.default_rng(54321)
        for scale in [0.0, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            if time.time() > deadline:
                break
            
            if scale == 0.0:
                start = best_pts
            else:
                start = _normalize(best_pts + polish_rng.standard_normal((N, 3)) * scale)
            
            polished = _optimize_lbfgs(start, deadline, max_iter=5000, ftol=1e-16, gtol=1e-12)
            e_polished = _precise_energy(polished)
            
            if e_polished < best_energy:
                best_energy = e_polished
                best_pts = polished
        
        # Differential evolution step
        if time.time() < deadline:
            de_pts, de_energy = _differential_evolution_step(
                best_pts, polish_rng, n_trials=30, deadline=deadline
            )
            if de_energy < best_energy:
                best_energy = de_energy
                best_pts = de_pts
        
        # One more final L-BFGS pass
        if time.time() < deadline:
            final = _optimize_lbfgs(best_pts, deadline, max_iter=10000, ftol=1e-18, gtol=1e-14)
            e_final = _precise_energy(final)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final
    
    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    
    if not np.isfinite(final_energy) or final_energy > 10000:
        # Fallback to Fibonacci
        best_pts = _generate_fibonacci(N)
        best_pts = _normalize(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
