"""Improved packer: multi-restart repulsion-based optimization with polish."""
import time
import numpy as np

N = 100


def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)


def _energy_and_grad(p, target_r):
    """Energy: sum of violations + gradient. Minimizes overlap and wall proximity."""
    d = p[:, None, :] - p[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    # Pairwise: penalize when dist < 2*target_r
    diff = dists - 2.0 * target_r
    # Smooth penalty: softplus-like
    beta = 50.0
    penalty = np.log1p(np.exp(-beta * diff))  # ~1 when diff<<0, ~0 when diff>>0
    # Only off-diagonal
    penalty[np.diag_indices_from(penalty)] = 0.0
    E_pair = penalty.sum()
    
    # Wall: penalize when clearance < target_r
    clearance = 0.5 - np.abs(p)
    wdiff = clearance - target_r
    penalty_w = np.log1p(np.exp(-beta * wdiff))
    E_wall = penalty_w.sum()
    
    E = E_pair + E_wall
    
    # Gradient
    grad = np.zeros_like(p)
    # Pairwise gradient
    # d/dp_i of log1p(exp(-beta*(d_ij - 2r))) = -beta * exp(-beta*diff) / (1+exp(-beta*diff)) * d/dp_i d_ij
    # d/dp_i d_ij = (p_i - p_j) / d_ij
    exp_neg = np.exp(-beta * diff)
    sig = exp_neg / (1.0 + exp_neg)  # sigmoid-like
    sig[np.diag_indices_from(sig)] = 0.0
    # grad_i = -beta * sum_j sig_ij * (p_i - p_j) / d_ij
    factor = -beta * sig / dists  # shape (N,N)
    factor[np.diag_indices_from(factor)] = 0.0
    grad += np.sum(factor[:, :, None] * d, axis=1)
    
    # Wall gradient: d/dp_ik of log1p(exp(-beta*(0.5-|p_ik|-r)))
    # = -beta * exp(-beta*wdiff) / (1+exp(-beta*wdiff)) * d/dp_ik(0.5-|p_ik|-r)
    # = -beta * sig_w * (-sign(p_ik))
    exp_neg_w = np.exp(-beta * wdiff)
    sig_w = exp_neg_w / (1.0 + exp_neg_w)
    grad -= beta * sig_w * np.sign(p)
    
    return E, grad


def _optimize_from_init(pts, t0, deadline, target_r):
    """Local optimization from initial points, returning best found."""
    best_pts = pts.copy()
    best_r = _score(best_pts)
    step = 0.003
    no_improve = 0
    max_no_improve = 2000
    
    for it in range(100000):
        if time.time() - t0 > deadline:
            break
        if no_improve > max_no_improve:
            break
        
        E, grad = _energy_and_grad(pts, target_r)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break
        
        # Normalize gradient
        g = grad / gnorm
        new_pts = pts + step * g
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        r = _score(new_pts)
        if r > best_r:
            best_r = r
            best_pts = new_pts.copy()
            no_improve = 0
            step = min(step * 1.02, 0.01)
        else:
            no_improve += 1
            step = max(step * 0.95, 1e-6)
        
        pts = new_pts
    
    return best_pts, best_r


def _polish(pts, target_r, t0, deadline):
    """High-precision polish with smaller steps and tighter convergence."""
    best_pts = pts.copy()
    best_r = _score(best_pts)
    step = 0.001
    no_improve = 0
    
    for it in range(50000):
        if time.time() - t0 > deadline:
            break
        if no_improve > 1000:
            break
        
        E, grad = _energy_and_grad(pts, target_r)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        g = grad / gnorm
        new_pts = pts + step * g
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        r = _score(new_pts)
        if r > best_r:
            best_r = r
            best_pts = new_pts.copy()
            no_improve = 0
            step = min(step * 1.01, 0.005)
        else:
            no_improve += 1
            step = max(step * 0.9, 1e-8)
        
        pts = new_pts
    
    return best_pts, best_r


def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin
    
    # Multiple initializations: grid + random perturbations
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    rng = np.random.default_rng(42)
    
    inits = []
    # Base grid
    inits.append(grid[:N].copy())
    # Random perturbations of grid
    for trial in range(5):
        pert = grid[:N].copy()
        pert += rng.normal(0, 0.02, size=pert.shape)
        pert = np.clip(pert, -0.45, 0.45)
        inits.append(pert)
    # Random positions
    for trial in range(3):
        rand_pts = rng.uniform(-0.45, 0.45, size=(N, 3))
        inits.append(rand_pts)
    
    best_pts = inits[0].copy()
    best_r = _score(best_pts)
    
    # Phase 1: Optimize from each init with increasing target radius
    # Start with a conservative target and increase
    target_r = best_r * 0.95
    
    for phase in range(8):
        if time.time() - t0 > DEADLINE - 30.0:
            break
        
        # Try each init
        for init in inits:
            if time.time() - t0 > DEADLINE - 30.0:
                break
            
            pts, r = _optimize_from_init(init.copy(), t0, DEADLINE - 30.0, target_r)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        
        # Increase target radius
        target_r = min(best_r * 1.02, 0.112)
        # Update inits with best found
        inits[0] = best_pts.copy()
        # Add perturbed versions of best
        pert = best_pts.copy()
        pert += rng.normal(0, 0.01, size=pert.shape)
        pert = np.clip(pert, -0.45, 0.45)
        inits[1] = pert
    
    # Phase 2: Polish
    if time.time() - t0 < DEADLINE - 10.0:
        best_pts, best_r = _polish(best_pts.copy(), best_r, t0, DEADLINE)
    
    # Final safety
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = grid[:N].copy()
    
    return best_pts
