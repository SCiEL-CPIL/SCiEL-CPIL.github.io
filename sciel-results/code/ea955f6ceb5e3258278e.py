"""Combined program: soft-min surrogate optimization with L-BFGS-B + Halton initialization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
CUBE_HALF = 0.5
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _score(centers):
    """Compute the common radius (score) for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > CUBE_HALF + 1e-10):
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    mpd = np.min(dists)
    mbd = np.min(CUBE_HALF - np.abs(centers))
    return min(mpd / 2.0, mbd)


def _soft_min_objective(x, beta, w_pair=1.0, w_bound=1.0):
    """
    Soft-min surrogate objective using log-sum-exp.
    We want to maximize the minimum of:
      - half pairwise distances
      - boundary distances
    Using a soft-min approximation: -logsumexp(-values / beta) / beta
    """
    c = x.reshape(N, 3)
    
    # Pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dists, 0.0)
    # Take upper triangle only
    iu = np.triu_indices(N, 1)
    pair_dists = np.sqrt(sq_dists[iu])
    
    # Boundary clearances
    bnd = CUBE_HALF - np.abs(c)
    bnd_flat = bnd.flatten()
    
    # Combine all "distances" (we want to maximize the min)
    # Values to take soft-min of: pair_dists/2 and bnd_flat
    all_vals = np.concatenate([pair_dists / 2.0, bnd_flat])
    
    # Soft-min: -1/beta * log(sum(exp(-val/beta)))
    # For numerical stability, subtract max
    max_val = np.max(all_vals)
    exp_terms = np.exp(-(all_vals - max_val) / beta)
    soft_min_val = max_val - beta * np.log(np.sum(exp_terms))
    
    # We want to maximize soft_min_val, so minimize its negative
    return -soft_min_val


def _soft_min_gradient(x, beta, w_pair=1.0, w_bound=1.0):
    """
    Gradient of the soft-min objective.
    d/d(c_ik) of -soft_min = -sum_j (p_j * d(val_j)/d(c_ik))
    where p_j = exp(-val_j/beta) / sum(exp(-val_k/beta))
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dists, 0.0)
    iu = np.triu_indices(N, 1)
    pair_dists = np.sqrt(sq_dists[iu])
    
    # Boundary clearances
    bnd = CUBE_HALF - np.abs(c)
    bnd_flat = bnd.flatten()
    
    all_vals = np.concatenate([pair_dists / 2.0, bnd_flat])
    
    # Compute softmax weights p_j = exp(-val_j/beta) / sum
    max_val = np.max(all_vals)
    exp_terms = np.exp(-(all_vals - max_val) / beta)
    p = exp_terms / np.sum(exp_terms)
    
    # Gradient with respect to each coordinate
    # For pairwise distances: d/d(c_i) |c_i - c_j| = (c_i - c_j) / |c_i - c_j|
    # For boundary: d/d(c_ik) (0.5 - |c_ik|) = -sign(c_ik)
    
    grad = np.zeros_like(c)
    
    # Pairwise contributions
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = p[idx] / beta
        d = c[i] - c[j]
        dist = pair_dists[idx]
        if dist > 1e-12:
            grad[i] += w * d / dist
            grad[j] -= w * d / dist
    
    # Boundary contributions
    bnd_idx_start = len(pair_dists)
    for idx in range(N * 3):
        i = idx // 3
        k = idx % 3
        w = p[bnd_idx_start + idx] / beta
        grad[i, k] += w * (-np.sign(c[i, k]))
    
    # The objective is -soft_min, so gradient is negated
    return (-grad).flatten()


def _halton_initial(n, dim=3, seed=0):
    """
    Generate Halton low-discrepancy sequence in [0,1)^dim, then map to cube.
    This breaks lattice symmetries that trap gradient-based optimizers.
    """
    def halton_sequence(n, base):
        seq = np.zeros(n)
        f = 1.0 / base
        for i in range(1, n + 1):
            j = i
            h = 0.0
            while j > 0:
                j -= 1
                f = f / base
                h += f * (j % base)
            seq[i - 1] = h
        return seq
    
    primes = [2, 3, 5]  # First 3 primes for 3D
    points = np.column_stack([halton_sequence(n, p) for p in primes[:dim]])
    # Map to [-0.5, 0.5]
    return points - 0.5


def _construct_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_random():
    """Random initialization inside the cube."""
    return (np.random.rand(N, 3) - 0.5) * 0.99


def _local_optimize_soft_min(centers, beta, w_pair=1.0, w_bound=1.0, maxiter=3000):
    """Run L-BFGS-B on the soft-min objective."""
    x0 = centers.flatten()
    result = minimize(
        _soft_min_objective,
        x0,
        jac=_soft_min_gradient,
        method='L-BFGS-B',
        bounds=[(-CUBE_HALF, CUBE_HALF)] * (N * 3),
        args=(beta, w_pair, w_bound),
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x.reshape(N, 3)


def _refine_soft_min(centers, time_limit):
    """
    Multi-stage soft-min refinement with annealed beta.
    Start with large beta (smooth), then decrease for sharper min approximation.
    """
    start_time = time.time()
    current = centers.copy()
    
    # Beta schedule: start smooth, get sharper
    beta_schedules = [
        (500.0, 500),
        (200.0, 500),
        (100.0, 800),
        (50.0, 800),
        (20.0, 1000),
        (10.0, 1500),
        (5.0, 2000),
    ]
    
    for beta, maxiter in beta_schedules:
        if time.time() - start_time > time_limit - 5:
            break
        try:
            current = _local_optimize_soft_min(current, beta, w_pair=1.0, w_bound=1.0, maxiter=maxiter)
            current = np.clip(current, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            if not np.all(np.isfinite(current)):
                break
        except Exception:
            break
    
    return current


def construct():
    """
    Construct centers of N equal spheres in the unit cube to maximize common radius.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    best_centers = _construct_grid()
    best_score = _score(best_centers)
    
    # Multiple initializations
    initializations = []
    
    # 1. Grid
    initializations.append(_construct_grid())
    
    # 2. Halton low-discrepancy (breaks lattice symmetries)
    initializations.append(_halton_initial(N))
    
    # 3. Random starts
    for _ in range(5):
        initializations.append(_construct_random())
    
    # 4. Grid with perturbations
    grid = _construct_grid()
    for scale in [0.01, 0.02, 0.03]:
        perturbed = grid + np.random.randn(N, 3) * scale
        perturbed = np.clip(perturbed, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        initializations.append(perturbed)
    
    # 5. Layered structure (5x5x2)
    layered = []
    for x in np.linspace(-0.4, 0.4, 5):
        for y in np.linspace(-0.4, 0.4, 5):
            for z in [-0.25, 0.25]:
                layered.append([x, y, z])
    layered = np.array(layered[:N])
    initializations.append(layered)
    
    # 6. Face-centered arrangement
    fcc_like = []
    for x in np.linspace(-0.375, 0.375, 4):
        for y in np.linspace(-0.375, 0.375, 4):
            for z in np.linspace(-0.333, 0.333, 3):
                fcc_like.append([x, y, z])
    fcc_like = np.array(fcc_like[:N])
    initializations.append(fcc_like)
    
    # Run soft-min optimization on each initialization
    for i, init in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        init = np.clip(init, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        try:
            # Quick optimization with moderate beta
            refined = _local_optimize_soft_min(init, beta=100.0, w_pair=1.0, w_bound=1.0, maxiter=500)
            refined = np.clip(refined, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_centers = refined.copy()
        except Exception:
            continue
    
    # Final refinement of the best solution with more time
    if time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        try:
            final = _refine_soft_min(best_centers, time_limit=remaining)
            final = np.clip(final, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            s = _score(final)
            if s > best_score:
                best_score = s
                best_centers = final.copy()
        except Exception:
            pass
    
    # Final safety
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    if not np.all(np.isfinite(best_centers)):
        best_centers = _construct_grid()
    
    return best_centers
