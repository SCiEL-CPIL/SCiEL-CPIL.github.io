"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay
from itertools import combinations

H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all C(11,3)=165 triangle index triplets
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def calc_areas_bary(bary):
    """Compute all 165 triangle areas from barycentric coordinates."""
    pi = bary[_TRI_IDX[:, 0]]
    pj = bary[_TRI_IDX[:, 1]]
    pk = bary[_TRI_IDX[:, 2]]
    
    det = (pi[:,0]*(pj[:,1]*pk[:,2] - pj[:,2]*pk[:,1])
          - pi[:,1]*(pj[:,0]*pk[:,2] - pj[:,2]*pk[:,0])
          + pi[:,2]*(pj[:,0]*pk[:,1] - pj[:,1]*pk[:,0]))
    
    return (np.sqrt(3)/4) * np.abs(det)

def clamp_bary(bary):
    """Clamp barycentric coordinates to valid simplex."""
    bary = np.maximum(bary, 0.0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary

def bary_to_cart(bary):
    """Convert barycentric to Cartesian coordinates."""
    cart = bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2
    return cart

def min_area_from_bary(bary):
    """Minimum area from barycentric coordinates."""
    areas = calc_areas_bary(bary)
    return float(np.min(areas))

def soft_min_obj(flat_bary, beta=100.0):
    """Soft-min objective in barycentric space. Returns negative soft-min area."""
    bary = flat_bary.reshape(-1, 3)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -400, 400)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(m - log_sum / beta)

def hard_min_obj(flat_bary):
    """Hard min objective in barycentric space. Returns negative min area."""
    bary = flat_bary.reshape(-1, 3)
    bary = clamp_bary(bary)
    return -min_area_from_bary(bary)

def _delaunay_smooth_obj(flat_bary, beta=50.0):
    """
    Delaunay-based smooth objective. 
    Uses Delaunay triangulation to identify critical triangles and applies
    a smooth penalty based on their areas.
    """
    bary = flat_bary.reshape(-1, 3)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    try:
        # Add triangle vertices to ensure proper triangulation
        all_pts = np.vstack([cart, V0.reshape(1,2), V1.reshape(1,2), V2.reshape(1,2)])
        tri = Delaunay(all_pts)
        
        # Get triangles that involve at least 2 of our points (indices 0..N-1)
        # tri.simplices has shape (M, 3)
        areas = np.zeros(len(tri.simplices))
        for idx, simplex in enumerate(tri.simplices):
            # Check if this triangle involves at least 2 of our points
            our_pts = [s for s in simplex if s < N]
            if len(our_pts) >= 2:
                p1 = all_pts[simplex[0]]
                p2 = all_pts[simplex[1]]
                p3 = all_pts[simplex[2]]
                area = 0.5 * np.abs(np.cross(p2-p1, p3-p1))
                areas[idx] = area
        
        valid_areas = areas[areas > 1e-10]
        if len(valid_areas) == 0:
            return 0.0
        
        m = np.min(valid_areas)
        # Smooth min approximation
        shifted = -beta * (valid_areas - m)
        shifted = np.clip(shifted, -400, 400)
        log_sum = np.log(np.sum(np.exp(shifted)))
        return -(m - log_sum / beta)
    except Exception:
        # Fallback to standard soft min
        return soft_min_obj(flat_bary, beta)

def generate_hexagonal_init():
    """Generate a hexagonal/symmetric initialization in barycentric coordinates."""
    cart = np.zeros((N, 2))
    cart[0] = [0.5, H/3]  # center
    
    cx, cy = 0.5, H/3
    r = 0.18
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        cart[i+1] = [cx + r * np.cos(angle), cy + r * np.sin(angle)]
    
    cart[7] = [0.25, H/6]
    cart[8] = [0.75, H/6]
    cart[9] = [0.5, 2*H/3]
    cart[10] = [0.5, H/6]
    
    bary = np.zeros((N, 3))
    for i in range(N):
        x, y = cart[i]
        c = 2.0 * y / np.sqrt(3)
        b = x - 0.5 * c
        a = 1.0 - b - c
        bary[i] = [a, b, c]
    
    return clamp_bary(bary)

def generate_grid_init():
    """Generate a grid-like initialization in barycentric coordinates."""
    cart = np.zeros((N, 2))
    
    # 3 rows, 4 columns in a staggered pattern
    row_y = [H/5, 2*H/5, 3*H/5, 4*H/5]
    col_x = [0.2, 0.4, 0.6, 0.8]
    
    idx = 0
    for r in range(4):
        for c in range(3):
            if idx >= N:
                break
            x = col_x[c] + (0.05 if r % 2 == 0 else -0.05)
            y = row_y[r]
            # Clamp to triangle
            if y > H * (1 - x) and x > 0.5:
                x = 0.5
            if y > H * x and x <= 0.5:
                x = 0.5
            cart[idx] = [x, y]
            idx += 1
    
    # Fill remaining
    if idx < N:
        cart[idx] = [0.5, H/2]
        idx += 1
    
    bary = np.zeros((N, 3))
    for i in range(N):
        x, y = cart[i]
        c = 2.0 * y / np.sqrt(3)
        b = x - 0.5 * c
        a = 1.0 - b - c
        bary[i] = [a, b, c]
    
    return clamp_bary(bary)

def generate_random_init(seed):
    """Generate random initialization in barycentric coordinates."""
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 3))
    for i in range(N):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        bary[i] = [1 - r1 - r2, r1, r2]
    return bary

def generate_perturbed_init(base_bary, scale, seed):
    """Generate perturbed initialization."""
    rng = np.random.default_rng(seed)
    bary = base_bary + rng.normal(0, scale, size=base_bary.shape)
    return clamp_bary(bary)

def _lbfgs_polish_bary(bary, beta, maxiter, deadline):
    """L-BFGS-B polish in barycentric space."""
    try:
        res = minimize(
            soft_min_obj, bary.flatten(), args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        opt_bary = clamp_bary(res.x.reshape(-1, 3))
        return opt_bary, min_area_from_bary(opt_bary)
    except Exception:
        return bary, min_area_from_bary(bary)

def _delaunay_polish_bary(bary, beta, maxiter, deadline):
    """L-BFGS-B polish using Delaunay-based objective."""
    try:
        res = minimize(
            _delaunay_smooth_obj, bary.flatten(), args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        opt_bary = clamp_bary(res.x.reshape(-1, 3))
        return opt_bary, min_area_from_bary(opt_bary)
    except Exception:
        return bary, min_area_from_bary(bary)

def _nm_polish_bary(bary, deadline):
    """Nelder-Mead polish in barycentric space."""
    try:
        res = minimize(
            hard_min_obj, bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        opt_bary = clamp_bary(res.x.reshape(-1, 3))
        return opt_bary, min_area_from_bary(opt_bary)
    except Exception:
        return bary, min_area_from_bary(bary)

def _local_polish_bary(bary, deadline, max_rounds=100):
    """
    Local polish using targeted moves on the minimum-area triangles.
    Identifies the critical triangles and performs local search to improve them.
    """
    current = bary.copy()
    score = min_area_from_bary(current)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        areas = calc_areas_bary(current)
        min_idx = np.argmin(areas)
        i, j, k = _TRI_IDX[min_idx]
        
        # Try small perturbations of the three vertices of the min triangle
        best_trial = current.copy()
        best_score = score
        
        for vertex_idx in [i, j, k]:
            for dx in [-0.01, -0.005, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0.005, 0.01]:
                    trial = current.copy()
                    trial[vertex_idx, 0] += dx
                    trial[vertex_idx, 1] += dy
                    trial[vertex_idx, 2] -= (dx + dy)
                    trial = clamp_bary(trial)
                    s = min_area_from_bary(trial)
                    if s > best_score + 1e-15:
                        best_score = s
                        best_trial = trial.copy()
        
        if best_score > score + 1e-15:
            current = best_trial
            score = best_score
        else:
            # Try random small perturbations
            rng = np.random.default_rng(round_num)
            improved = False
            for _ in range(20):
                trial = current.copy()
                idx = rng.integers(0, N)
                trial[idx] += rng.normal(0, 0.005, size=3)
                trial = clamp_bary(trial)
                s = min_area_from_bary(trial)
                if s > score + 1e-15:
                    current = trial
                    score = s
                    improved = True
                    break
            if not improved:
                break
    
    return current, score

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    def tleft():
        return deadline - time.time()
    
    # Generate diverse initializations
    inits = [generate_hexagonal_init(), generate_grid_init()]
    for s in [1, 7, 13, 29, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000]:
        inits.append(generate_random_init(s))
    
    best_bary = None
    best_score = -1.0
    
    # Phase 1: Multi-start with both standard and Delaunay-based objectives
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    for init in inits:
        if tleft() < 15:
            break
        active_betas = betas
        if tleft() < 60:
            active_betas = betas[1:]
        if tleft() < 30:
            active_betas = betas[2:]
            
        for beta in active_betas:
            if tleft() < 5:
                break
            # Use standard soft min
            bary, sc = _lbfgs_polish_bary(init, beta, 600, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
            
            # Also try Delaunay-based objective
            if tleft() > 10:
                bary, sc = _delaunay_polish_bary(init, beta, 400, deadline)
                if sc > best_score:
                    best_score = sc
                    best_bary = bary.copy()
    
    if best_bary is None:
        best_bary = generate_hexagonal_init()
        best_score = min_area_from_bary(best_bary)
    
    # Phase 2: Nelder-Mead polish on best
    if tleft() > 15:
        bary, sc = _nm_polish_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    # Phase 3: Local polish targeting critical triangles
    if tleft() > 15:
        bary, sc = _local_polish_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    # Phase 4: Perturbation multi-start around best
    rng = np.random.default_rng(99999)
    for trial in range(30):
        if tleft() < 10:
            break
        base_scale = 0.005
        scale = base_scale * (1 + 0.5 * np.sin(trial * 0.5))
        if trial % 5 == 0:
            scale *= 2.0
            
        pert = generate_perturbed_init(best_bary, scale, 99999 + trial)
        
        for beta in [100.0, 200.0]:
            if tleft() < 5:
                break
            bary, sc = _lbfgs_polish_bary(pert, beta, 500, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
        
        if tleft() > 8:
            bary, sc = _nm_polish_bary(pert, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
        
        if tleft() > 5:
            bary, sc = _local_polish_bary(best_bary, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
    
    # Phase 5: Final refinement
    if tleft() > 10:
        bary, sc = _nm_polish_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    if tleft() > 5:
        bary, sc = _local_polish_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    # Convert best barycentric to Cartesian
    best_pts = bary_to_cart(best_bary)
    return best_pts
# EVOLVE-BLOCK-END
