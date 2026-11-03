"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _soft_min_logsumexp(d, beta):
    """
    Smooth surrogate for min(d) using log-sum-exp.
    Returns -log(sum(exp(-beta*d)))/beta which approximates min(d) as beta->inf.
    We want to maximize min distance, so we minimize this surrogate.
    """
    d_min = np.min(d)
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    return d_min - np.log(s) / beta


def _score(c):
    """
    Compute the actual score (largest common radius) for a configuration.
    c: (N, 3) array of centers.
    Returns the radius r.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    face_dists = 0.5 - np.abs(c)
    min_face = np.min(face_dists)
    if min_face < -1e-12:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_face)
    return r


def _objective(c, beta):
    """
    Objective function to minimize: -soft_min of constraints.
    """
    c = c.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    face_dists = 0.5 - np.abs(c)
    face_flat = face_dists.flatten()
    
    all_dists = np.concatenate([pair_dists, face_flat])
    
    r_smooth = _soft_min_logsumexp(all_dists, beta)
    
    return -r_smooth


def _generate_bcc_init():
    """
    Generate an initial configuration based on a BCC lattice.
    """
    # We want roughly 50 points.
    # A 3x4x4 BCC block has 3*4*4*2 = 96 points? No.
    # Let's use a grid of unit cells.
    # If we use a 4x4x4 grid of cells, size 4x4x4.
    # BCC points: corners (5x5x5=125) + centers (4x4x4=64) = 189 points.
    # This is too many.
    # Let's try a 3x3x3 grid of cells? Size 3x3x3.
    # BCC points: corners (4x4x4=64) + centers (3x3x3=27) = 91 points.
    # Still too many.
    # Let's just generate a random set of points and then "relax" them?
    # Or, generate a BCC lattice with a specific spacing and crop.
    
    # Let's try spacing s such that we get ~50 points in [-0.5, 0.5].
    # Volume of cube is 1.
    # BCC density is 2 per volume s^3.
    # N ~ 2 / s^3.
    # 50 ~ 2 / s^3 => s^3 ~ 0.04 => s ~ 0.34.
    # Let's use s = 0.35.
    s = 0.35
    # Generate points in a larger range and crop
    x = np.arange(-1.0, 1.0 + s, s)
    y = np.arange(-1.0, 1.0 + s, s)
    z = np.arange(-1.0, 1.0 + s, s)
    
    pts = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                # Corner
                pts.append([x[i], y[j], z[k]])
                # Center
                pts.append([x[i] + s/2, y[j] + s/2, z[k] + s/2])
    
    pts = np.array(pts)
    # Crop to [-0.5, 0.5]
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    
    # If we have more than N, pick random N.
    # If less, pad with random points.
    if len(pts) > N:
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        # Pad with random points
        n_pad = N - len(pts)
        pad = np.random.uniform(-0.4, 0.4, size=(n_pad, 3))
        pts = np.vstack([pts, pad])
        
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Initializations
    # 1. BCC
    c_bcc = _generate_bcc_init()
    # 2. Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c_grid = grid[:N].copy()
    
    # 3. Random
    rng = np.random.default_rng(42)
    c_rand = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    initializations = [c_bcc, c_grid, c_rand]
    
    best_c = c_bcc.copy()
    best_score = _score(best_c)
    
    # Multi-start optimization with beta annealing
    # Beta schedule: start low, increase
    beta_starts = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    # Number of restarts per beta stage
    n_restarts = 3
    
    for beta_idx, beta in enumerate(beta_starts):
        if time.time() > deadline:
            break
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Generate initial point
            if restart == 0:
                # Use current best
                c_init = best_c.copy()
            elif restart == 1:
                # Perturbed best
                c_init = best_c + rng.normal(0, 0.05, size=(N, 3))
            else:
                # Use one of the other initializations
                c_init = initializations[(beta_idx + restart) % len(initializations)].copy()
            
            # Clip to valid range
            c_init = np.clip(c_init, -0.49, 0.49)
            
            # Optimize
            try:
                res = minimize(
                    _objective,
                    c_init.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                c_opt = res.x.reshape(N, 3)
                
                # Clip to ensure validity
                c_opt = np.clip(c_opt, -0.4999, 0.4999)
                
                score = _score(c_opt)
                if score > best_score:
                    best_score = score
                    best_c = c_opt.copy()
            except Exception:
                pass
        
        # If we've reached a good score, try a final polish with higher beta
        if best_score > 0.145:
            break
    
    # Final polish pass with high beta on the best configuration
    if time.time() < deadline - 5.0:
        try:
            # Use a very high beta for final polish
            res = minimize(
                _objective,
                best_c.flatten(),
                args=(5000.0,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_polished = res.x.reshape(N, 3)
            c_polished = np.clip(c_polished, -0.49999, 0.49999)
            
            score = _score(c_polished)
            if score > best_score:
                best_score = score
                best_c = c_polished.copy()
        except Exception:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
