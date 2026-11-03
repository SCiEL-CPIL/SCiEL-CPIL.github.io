import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """
    Compute the maximum radius for given centers.
    r = min(min_pair_dist / 2, min_wall_dist)
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _softmin(v, beta):
    """
    Soft-min of vector v using log-sum-exp.
    """
    min_v = np.min(v)
    # Subtract min for numerical stability
    shifted = v - min_v
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, 0, 50.0)
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return min_v - (1.0 / beta) * log_sum

def _objective_and_grad(x, beta):
    """
    Objective: -softmin(gaps)
    Returns (value, gradient)
    """
    centers = x.reshape(N, 3)
    
    # Pairwise gaps: dists[i,j] / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_gaps = dists / 2.0
    
    # Wall gaps
    wall_gaps = CUBE_HALF - np.abs(centers)
    
    # Combine
    # We need to compute softmin of all gaps
    # For efficiency, we can compute the softmin of pair_gaps and wall_gaps separately
    # but the true softmin is over the union.
    # Let's compute the union softmin.
    
    # Flatten pair_gaps (excluding diagonal which is inf)
    # It's easier to just concatenate
    # But gradient computation is easier if we handle them separately?
    # Actually, softmin is not decomposable like that easily.
    # Let's compute the full vector.
    
    # To save memory/time, we can approximate the gradient.
    # grad of softmin(v) wrt v_i = -exp(-beta*(v_i - min_v)) / sum(exp(-beta*(v_j - min_v)))
    
    # Let's compute weights
    min_pair = np.min(pair_gaps)
    min_wall = np.min(wall_gaps)
    
    # This is getting complex. Let's use a simpler gradient approximation or just use finite differences?
    # No, L-BFGS-B can do finite differences but it's slow.
    # Let's implement the gradient properly.
    
    # We need the softmin over the combined set.
    # Let V be the combined vector.
    # Let m = min(V).
    # w_i = exp(-beta * (V_i - m))
    # softmin = m - (1/beta) * log(sum(w_i))
    # d(softmin)/d(V_i) = -w_i / sum(w_j)
    
    # We can compute this for pair and wall separately if we normalize weights?
    # No, the min m is global.
    
    # Let's just compute the full vector for gradient.
    # N=50, pairs=1225, walls=150. Total ~1400 elements.
    
    pair_flat = pair_gaps[np.triu_indices(N, k=1)]
    wall_flat = wall_gaps.flatten()
    
    v = np.concatenate([pair_flat, wall_flat])
    
    min_v = np.min(v)
    shifted = v - min_v
    shifted = np.clip(shifted, 0, 50.0)
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    val = min_v - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    weights = exp_terms / sum_exp
    
    grad_v = -weights
    
    # Map back to x
    grad = np.zeros_like(x)
    centers = x.reshape(N, 3)
    
    # Wall gradients
    # wall_gaps[i, k] = 0.5 - |c_ik|
    # d(wall_gap)/d(c_ik) = -sign(c_ik)
    # d(softmin)/d(c_ik) = sum over j where wall_gap is involved? No, each wall gap is unique to c_ik.
    # So for each i, k:
    # d(softmin)/d(c_ik) += grad_v[wall_index] * (-sign(c_ik))
    
    # The wall_flat is ordered by i then k (row-major)
    # So wall_flat[i*3 + k] corresponds to centers[i, k]
    
    wall_grads = grad_v[N*(N-1):].reshape(N, 3)
    # d(softmin)/d(c_ik) = wall_grads[i,k] * d(wall_gap)/d(c_ik)
    # d(0.5 - |c|)/d(c) = -sign(c)
    signs = np.sign(centers)
    # If c=0, sign is 0, which is fine (subgradient 0 is a valid choice)
    grad += wall_grads * (-signs)
    
    # Pair gradients
    # pair_flat is upper triangle
    # For each pair (i, j) with i < j:
    # gap = dist(i,j) / 2
    # d(gap)/d(c_i) = (c_i - c_j) / (2 * dist(i,j)) * 2 = (c_i - c_j) / dist(i,j)  ... wait
    # dist = ||c_i - c_j||
    # gap = dist / 2
    # d(gap)/d(c_i) = 0.5 * (c_i - c_j) / dist
    # d(gap)/d(c_j) = 0.5 * (c_j - c_i) / dist
    
    # We need to map grad_v[pair_index] to these.
    # pair_flat order: (0,1), (0,2), ..., (0,N-1), (1,2), ...
    
    # We can loop or vectorize.
    # Since N=50, a loop is fast enough.
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            g = grad_v[idx]
            idx += 1
            d = dists[i, j]
            if d > 1e-12:
                vec = (centers[i] - centers[j]) / d
                # d(gap)/d(c_i) = 0.5 * vec
                # d(gap)/d(c_j) = -0.5 * vec
                grad[i * 3:(i + 1) * 3] += 0.5 * g * vec
                grad[j * 3:(j + 1) * 3] += -0.5 * g * vec
    
    return -val, -grad

def _generate_fcc_lattice():
    """
    Generate FCC lattice points and select 50.
    """
    # FCC unit cell has points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in relative coords
    # Side length 1.
    # We want to fit as many as possible.
    # Let's generate a grid of FCC points in a larger box and then scale/clip.
    
    # Alternatively, generate points in [-0.5, 0.5]
    # FCC density is pi/sqrt(18) ~ 0.74
    # Volume 1. 50 spheres of radius r. Volume = 50 * (4/3) pi r^3.
    # If r ~ 0.135, vol ~ 50 * 1.06 ~ 53. This is > 1, so spheres overlap?
    # Wait, the task says non-overlapping.
    # r = 0.1359.
    # Vol of one sphere = 4/3 * pi * 0.1359^3 ~ 0.0104.
    # 50 spheres ~ 0.52. This is < 1. So it fits.
    
    # Let's just generate a grid of points and rotate/scale.
    # A simple cubic grid of 4x4x4 = 64 points.
    # FCC is denser.
    
    # Let's try to construct a specific FCC-like arrangement.
    # Or just use a random perturbation of a grid.
    
    # For now, let's use a 4x4x4 grid centered.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select 50 random points
    idx = np.random.permutation(len(grid))
    grid = grid[idx[:N]]
    
    # Add noise
    grid += np.random.normal(0, 0.02, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _generate_bcc_lattice():
    """
    Generate BCC lattice points.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add body centers
    body_centers = []
    for x in ticks[:-1]:
        for y in ticks[:-1]:
            for z in ticks[:-1]:
                body_centers.append((x + 0.125, y + 0.125, z + 0.125)) # Approx
    # This is getting complicated. Let's just use a rotated grid.
    
    idx = np.random.permutation(len(grid))
    grid = grid[idx[:N]]
    
    # Random rotation
    A = np.random.normal(0, 1, (3, 3))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    grid = grid @ Q.T
    
    max_coord = np.max(np.abs(grid))
    if max_coord > 0.5:
        grid = grid * (0.5 / max_coord)
        
    return grid

def _optimize_from_initial(x0, time_limit):
    """
    Run multi-stage optimization from initial guess.
    """
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Stage 1: Low Beta
    betas = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
    
    x = x0
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(_objective_and_grad, x, args=(beta,), method='L-BFGS-B',
                           bounds=bounds, jac=True,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            x = res.x
        except Exception:
            pass
            
    return x

def construct():
    start_time = time.time()
    
    best_centers = None
    best_score = -1.0
    
    # Initializations
    inits = []
    
    # 1. Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.permutation(len(grid))
    grid_init = grid[idx[:N]] + np.random.normal(0, 0.02, (N, 3))
    grid_init = np.clip(grid_init, -0.49, 0.49)
    inits.append(grid_init.flatten())
    
    # 2. Rotated Grid
    for _ in range(5):
        g = grid[idx[:N]].copy()
        A = np.random.normal(0, 1, (3, 3))
        Q, R = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        g = g @ Q.T
        max_coord = np.max(np.abs(g))
        if max_coord > 0.5:
            g = g * (0.5 / max_coord)
        inits.append(g.flatten())
        
    # 3. Random
    for _ in range(5):
        r = np.random.uniform(-0.5, 0.5, (N, 3))
        inits.append(r.flatten())
        
    # 4. FCC-like (just a perturbed grid for now, as constructing true FCC is complex)
    # Let's add a specific structured init: layered
    layer_y = [-0.4, -0.2, 0.0, 0.2, 0.4]
    layer_points = []
    for y in layer_y:
        # 10 points per layer
        x = np.linspace(-0.4, 0.4, 10)
        z = np.linspace(-0.4, 0.4, 10)
        # Take 10 points
        pts = []
        for i in range(10):
            px = x[i % 10]
            pz = z[(i // 10) % 10]
            pts.append((px, y, pz))
        layer_points.extend(pts)
        
    # We have 50 points
    layered = np.array(layer_points[:N])
    layered += np.random.normal(0, 0.01, layered.shape)
    inits.append(layered.flatten())
    
    # Run optimization for each init
    for i, x0 in enumerate(inits):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        # Reserve time for remaining inits
        remaining = DEADLINE_SECONDS - (time.time() - start_time)
        time_per_init = remaining / (len(inits) - i)
        
        x_opt = _optimize_from_initial(x0, time_per_init)
        
        centers = x_opt.reshape(N, 3)
        
        if not np.all(np.isfinite(centers)):
            continue
            
        score = _compute_radius(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    # Final polish on best_centers if time permits
    if best_centers is not None and time.time() - start_time < DEADLINE_SECONDS - 10:
        x0 = best_centers.flatten()
        x_opt = _optimize_from_initial(x0, DEADLINE_SECONDS - (time.time() - start_time) - 2)
        centers = x_opt.reshape(N, 3)
        if np.all(np.isfinite(centers)):
            score = _compute_radius(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                
    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
