"""Rewritten program: two-phase momentum soft-min gradient ascent for 100-sphere packing."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _compute_radius(centres):
    """Compute the maximum radius for given centres."""
    n = len(centres)
    if n < 2:
        return (0.5 - np.abs(centres)).min()
    dist = cdist(centres, centres)
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    r_boundary = (0.5 - np.abs(centres)).min()
    return min(r_pair, r_boundary)


def _objective_and_grad(x_flat, temp=1.0):
    """
    Compute negative soft-min objective and its gradient.
    Objective: -softmin(dists) where dists include pairwise and boundary distances.
    """
    c = x_flat.reshape(-1, 3)
    n = c.shape[0]
    
    # Pairwise distances
    dist = cdist(c, c)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances: for each point, distance to nearest wall
    # d_i = min over k of (0.5 - |c_ik|)
    boundary_dists = np.minimum(0.5 - np.abs(c), axis=0)  # shape (n, 3)
    # For each point, the min over 3 walls
    # Actually we need the min over all 6 walls per point
    # 0.5 - c_ik and 0.5 + c_ik for each k
    left_right = np.stack([0.5 - c[:, 0], 0.5 + c[:, 0],
                            0.5 - c[:, 1], 0.5 + c[:, 1],
                            0.5 - c[:, 2], 0.5 + c[:, 2]], axis=1)  # (n, 6)
    boundary_per_point = left_right.min(axis=1)  # (n,)
    
    # Combine all distances: pairwise (n*(n-1)/2) + boundary (n)
    # For soft-min, we use all pairwise distances (both i<j and j<i are same, but we use upper triangle)
    # Actually let's just use all pairwise distances from the matrix (excluding diagonal)
    # and boundary distances
    
    # Flatten pairwise distances (upper triangle only to avoid duplicates)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]  # shape (n*(n-1)/2,)
    
    # All distances for soft-min
    all_dists = np.concatenate([pair_dists, boundary_per_point])
    m = len(all_dists)
    
    # Soft-min: -log(sum(exp(-d/temp))) * temp
    # For numerical stability
    min_d = all_dists.min()
    shifted = all_dists - min_d
    weights = np.exp(-shifted / temp)
    weights /= weights.sum()
    soft_min = min_d + temp * np.log(weights.sum())
    
    objective = -soft_min
    
    # Gradient
    # d(soft_min)/d(d_k) = weights[k]
    # For pairwise dist d_ij = |c_i - c_j|
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
    
    grad = np.zeros_like(c)
    
    # Pairwise contributions
    # For each pair (i,j) with i<j:
    # contribution to grad[i] += weights[pair_idx] * (c_i - c_j) / d_ij
    # contribution to grad[j] += weights[pair_idx] * (c_j - c_i) / d_ij
    
    # We need to map pair_dists back to (i,j) pairs
    # iu gives us (i_indices, j_indices)
    i_idx, j_idx = iu
    pair_weights = weights[:len(pair_dists)]
    
    for idx in range(len(pair_dists)):
        i, j = i_idx[idx], j_idx[idx]
        d = pair_dists[idx]
        if d > 1e-12:
            dir_vec = c[i] - c[j]
            w = pair_weights[idx]
            grad[i] += w * dir_vec / d
            grad[j] -= w * dir_vec / d
    
    # Boundary contributions
    # For each point, the boundary distance is min over 6 walls
    # The gradient is the unit vector pointing away from the nearest wall
    bound_weights = weights[len(pair_dists):]
    for i in range(n):
        w = bound_weights[i]
        # Find which wall is closest
        for k in range(3):
            # Left wall: 0.5 - c[i,k], gradient direction is +x_k (push right)
            # Right wall: 0.5 + c[i,k], gradient direction is -x_k (push left)
            d_left = 0.5 - c[i, k]
            d_right = 0.5 + c[i, k]
            if d_left < d_right:
                # Nearest is left wall, push in +k direction
                grad[i, k] += w
            else:
                # Nearest is right wall, push in -k direction
                grad[i, k] -= w
    
    # objective = -soft_min, so grad_objective = -grad_soft_min
    # grad_soft_min = sum_k weights[k] * d(d_k)/d(x)
    # We computed grad as sum of weights * d(d_k)/d(c), which is grad(soft_min)
    # So grad(objective) = -grad(soft_min)
    grad_flat = -grad.flatten()
    
    return objective, grad_flat


def _momentum_ascent(x0, time_budget=60.0, initial_temp=0.05, final_temp=0.001,
                     lr_start=0.001, lr_end=0.0001, momentum=0.9):
    """
    Two-phase momentum-based gradient ascent with annealing.
    Phase 1: coarse with higher temperature
    Phase 2: fine with lower temperature
    """
    start_time = time.time()
    n = x0.shape[0] // 3
    x = x0.copy()
    v = np.zeros_like(x)
    best_x = x.copy()
    best_obj = _compute_radius(x.reshape(n, 3))
    
    total_time = time_budget
    phase1_time = total_time * 0.6
    phase2_time = total_time * 0.4
    
    # Phase 1
    temp = initial_temp
    lr = lr_start
    iters = 0
    while time.time() - start_time < phase1_time:
        obj, grad = _objective_and_grad(x, temp)
        v = momentum * v + lr * grad
        x_new = x + v
        # Clip to cube
        x_new = np.clip(x_new, -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Check improvement
        r = _compute_radius(x_new.reshape(n, 3))
        if r > best_obj:
            best_obj = r
            best_x = x_new.copy()
        
        x = x_new
        iters += 1
        
        # Anneal temperature and learning rate
        progress = (time.time() - start_time) / phase1_time
        temp = initial_temp * (final_temp / initial_temp) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
    
    # Phase 2: fine-tuning
    x = best_x.copy()
    v = np.zeros_like(x)
    temp = final_temp
    lr = lr_end * 0.5
    iters2 = 0
    while time.time() - start_time < total_time:
        obj, grad = _objective_and_grad(x, temp)
        v = momentum * v + lr * grad
        x_new = x + v
        x_new = np.clip(x_new, -0.5 + 1e-10, 0.5 - 1e-10)
        
        r = _compute_radius(x_new.reshape(n, 3))
        if r > best_obj:
            best_obj = r
            best_x = x_new.copy()
        
        x = x_new
        iters2 += 1
        
        # Slowly decay learning rate
        lr *= 0.999
    
    return best_x, best_obj


def _initial_grid():
    """Create initial cubic grid packing 5x5x4 = 100."""
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid.copy()


def _initial_fcc():
    """Create FCC-like initial configuration."""
    # FCC lattice in a cube
    # Place points at FCC positions
    points = []
    # Simple cubic with some FCC-like perturbations
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    # Add small FCC-like offsets to break symmetry
    rng = np.random.RandomState(123)
    points = np.array(points)
    points += rng.normal(0, 0.01, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points


def _random_initialization(rng, n=100):
    """Random initialization inside the cube."""
    points = rng.uniform(-0.45, 0.45, size=(n, 3))
    return points


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_initial_grid())
    inits.append(_initial_fcc())
    
    # Add a few random initializations
    for _ in range(3):
        inits.append(_random_initialization(rng))
    
    best_centres = inits[0]
    best_radius = _compute_radius(best_centres)
    
    # Allocate time budget across initializations
    elapsed = time.time() - start_time
    time_per_init = min(25.0, (TIME_LIMIT - elapsed - 10) / len(inits))
    
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed - 5
        if remaining < 3:
            break
        
        # Time budget for this initialization
        t_budget = min(time_per_init, remaining - 2)
        if t_budget < 2:
            break
        
        # Run momentum ascent
        try:
            result, r = _momentum_ascent(
                init.flatten(), 
                time_budget=t_budget,
                initial_temp=0.05 + 0.01 * i,
                final_temp=0.001,
                lr_start=0.001,
                lr_end=0.0001,
                momentum=0.9
            )
            result = result.reshape(-1, 3)
            if r > best_radius:
                best_radius = r
                best_centres = result
        except Exception:
            pass
    
    # Final polish with very fine optimization on the best result
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed - 2
    if remaining > 5:
        try:
            result, r = _momentum_ascent(
                best_centres.flatten(),
                time_budget=remaining - 2,
                initial_temp=0.005,
                final_temp=0.0005,
                lr_start=0.0002,
                lr_end=0.00005,
                momentum=0.95
            )
            result = result.reshape(-1, 3)
            if r > best_radius:
                best_radius = r
                best_centres = result
        except Exception:
            pass
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centres
