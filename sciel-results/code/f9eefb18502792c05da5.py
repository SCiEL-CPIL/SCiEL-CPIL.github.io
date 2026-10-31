"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _pairwise_dists(pts):
    """Compute all pairwise distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    iu = np.triu_indices(n, 1)
    return dists[iu]

def _score(pts):
    """Compute (max/min)^2."""
    d = _pairwise_dists(pts)
    return (d.max() / d.min())**2

def _soft_objective(pts, beta):
    """Smooth surrogate: use log-sum-exp for max, and -log-sum-exp for min (inverted)."""
    d = _pairwise_dists(pts)
    d = np.maximum(d, 1e-10)
    # Soft max via log-sum-exp
    log_max = np.log(np.sum(np.exp(beta * d))) / beta
    # Soft min: use negative log-sum-exp of 1/d
    inv_d = 1.0 / d
    log_min_inv = np.log(np.sum(np.exp(beta * inv_d))) / beta
    soft_min = 1.0 / log_min_inv
    return (log_max / soft_min)**2

def _soft_objective_sq(pts, beta):
    """Smooth surrogate using squared distances for numerical stability."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq = dists_sq[iu]
    d_sq = np.maximum(d_sq, 1e-12)
    
    max_d_sq = np.max(d_sq)
    # Soft max of d_sq
    soft_max_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    min_d_sq = np.min(d_sq)
    # Soft min of d_sq
    soft_min_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min_sq < 1e-10:
        return 1e10
    return soft_max_sq / soft_min_sq

def _make_ring(n, radius, angle_offset=0.0):
    """Create n points on a ring of given radius."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _init_single_ring(n=16):
    """Regular 16-gon."""
    return _make_ring(n, 1.0)

def _init_two_rings():
    """8 inner + 8 outer, with rotation offset for symmetry."""
    inner = _make_ring(8, 0.6, angle_offset=0.0)
    outer = _make_ring(8, 1.3, angle_offset=np.pi/8)
    return np.vstack([inner, outer])

def _init_three_rings():
    """1 center + 5 inner + 10 outer."""
    center = np.array([[0.0, 0.0]])
    inner = _make_ring(5, 0.5, angle_offset=0.0)
    outer = _make_ring(10, 1.2, angle_offset=0.0)
    return np.vstack([center, inner, outer])

def _init_hexagonal():
    """Hexagonal lattice pattern, take 16 points."""
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts[:16])
    pts -= pts.mean(axis=0)
    return pts

def _init_random(seed=42):
    """Random points in unit disk."""
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2*np.pi, N)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _generate_diverse_seeds():
    """Generate a diverse set of initial configurations."""
    seeds = []
    
    # Basic symmetric configs
    seeds.append(_init_single_ring())
    seeds.append(_init_two_rings())
    seeds.append(_init_three_rings())
    seeds.append(_init_hexagonal())
    
    # More two-ring variations
    for r_in in [0.5, 0.6, 0.7]:
        for r_out in [1.2, 1.4, 1.6]:
            for offset in [0, np.pi/8, np.pi/4]:
                inner = _make_ring(8, r_in, angle_offset=0.0)
                outer = _make_ring(8, r_out, angle_offset=offset)
                seeds.append(np.vstack([inner, outer]))
    
    # Three-ring variations
    configs = [(6, 5, 5), (4, 6, 6), (5, 6, 5)]
    radii_sets = [
        (0.4, 1.0, 1.6),
        (0.5, 1.1, 1.8),
        (0.3, 0.9, 1.5)
    ]
    for n1, n2, n3 in configs:
        for r1, r2, r3 in radii_sets:
            a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
            a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
            a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + np.pi/n3
            p1 = np.column_stack([r1*np.cos(a1), r1*np.sin(a1)])
            p2 = np.column_stack([r2*np.cos(a2), r2*np.sin(a2)])
            p3 = np.column_stack([r3*np.cos(a3), r3*np.sin(a3)])
            seeds.append(np.vstack([p1, p2, p3]))
    
    # Random seeds
    for seed_val in [42, 123, 456, 789, 101112]:
        seeds.append(_init_random(seed_val))
        
    # Perturbed versions of best seeds
    rng = np.random.default_rng(7)
    scored = [(_score(s), s) for s in seeds]
    scored.sort(key=lambda x: x[0])
    for _, s in scored[:5]:
        for _ in range(2):
            perturbed = s + rng.normal(0, 0.1, size=(16, 2))
            seeds.append(perturbed)
            
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_pts = None
    best_score = np.inf
    
    seeds = _generate_diverse_seeds()
    
    # Score all seeds and sort
    scored_seeds = [(_score(s), s) for s in seeds]
    scored_seeds.sort(key=lambda x: x[0])
    
    # Process top seeds with multi-stage optimization
    num_to_optimize = 15
    
    for i in range(min(num_to_optimize, len(scored_seeds))):
        if time.time() > deadline - 15:
            break
            
        curr_score, pts = scored_seeds[i]
        if curr_score < best_score:
            best_score = curr_score
            best_pts = pts.copy()
            
        current_pts = pts.copy()
        
        # Stage 1: Powell on exact objective
        if time.time() < deadline - 12:
            try:
                res1 = minimize(lambda p: _score(p.reshape(N, D)), current_pts.flatten(), 
                                method='Powell', options={'maxiter': 100, 'maxfev': 1000})
                sc1 = _score(res1.x.reshape(N, D))
                if sc1 < best_score:
                    best_score = sc1
                    best_pts = res1.x.reshape(N, D).copy()
                current_pts = res1.x.reshape(N, D)
            except Exception:
                pass
        
        # Stage 2: L-BFGS-B on soft objective (beta=50)
        if time.time() < deadline - 10:
            try:
                res2 = minimize(lambda p: _soft_objective_sq(p.reshape(N, D), 50.0), current_pts.flatten(),
                                method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                sc2 = _score(res2.x.reshape(N, D))
                if sc2 < best_score:
                    best_score = sc2
                    best_pts = res2.x.reshape(N, D).copy()
                current_pts = res2.x.reshape(N, D)
            except Exception:
                pass
                
        # Stage 3: Powell again on exact objective
        if time.time() < deadline - 8:
            try:
                res3 = minimize(lambda p: _score(p.reshape(N, D)), current_pts.flatten(), 
                                method='Powell', options={'maxiter': 200, 'maxfev': 2000})
                sc3 = _score(res3.x.reshape(N, D))
                if sc3 < best_score:
                    best_score = sc3
                    best_pts = res3.x.reshape(N, D).copy()
                current_pts = res3.x.reshape(N, D)
            except Exception:
                pass
                
        # Stage 4: L-BFGS-B on higher beta soft objective (beta=200)
        if time.time() < deadline - 5:
            try:
                res4 = minimize(lambda p: _soft_objective_sq(p.reshape(N, D), 200.0), current_pts.flatten(),
                                method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
                sc4 = _score(res4.x.reshape(N, D))
                if sc4 < best_score:
                    best_score = sc4
                    best_pts = res4.x.reshape(N, D).copy()
            except Exception:
                pass

    # Final polish if time permits
    if best_pts is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(lambda p: _soft_objective_sq(p.reshape(N, D), 500.0), best_pts.flatten(),
                                 method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15})
            sc_final = _score(res_final.x.reshape(N, D))
            if sc_final < best_score:
                best_score = sc_final
                best_pts = res_final.x.reshape(N, D).copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _init_single_ring()
    
    # Verify distinctness
    d_check = pdist(best_pts, 'euclidean')
    if np.min(d_check) < 1e-10:
        best_pts = _init_single_ring()
        
    return best_pts
# EVOLVE-BLOCK-END
