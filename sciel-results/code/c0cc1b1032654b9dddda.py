import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, cKDTree
import sys

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s kill

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def compute_radius(pts):
    """
    Compute the maximum radius for the given centers.
    Returns -1e6 if invalid (outside cube).
    """
    if pts is None or pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    
    # Pairwise distances
    diff = pts[_iu] - pts[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def neg_radius(pts):
    r = compute_radius(pts)
    return -r

def soft_min_radius_and_grad(pts, beta=50.0):
    """
    Smooth surrogate for the negative radius using log-sum-exp soft-min.
    Returns (value, gradient).
    """
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6, np.zeros_like(pts)
    
    diff = pts[_iu] - pts[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-9, 1e-9, dists)
    
    pair_dists = dists_safe / 2.0
    wall_dists = 0.5 - np.abs(pts)
    wall_flat = wall_dists.flatten()
    
    pair_flat = pair_dists
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    
    # Clip to prevent overflow in exp
    shifted = np.clip(shifted, -100, 100)
    
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    soft_min_val = d_min - log_sum / beta
    
    # Gradient calculation
    grad = np.zeros_like(pts)
    
    # Weights for pairs
    pair_weights = exp_shifted[:_NUM_PAIRS]
    # Weights for walls
    wall_weights = exp_shifted[_NUM_PAIRS:]
    
    total_sum = np.sum(exp_shifted)
    
    if total_sum > 0:
        # Pair contributions
        unit_vecs = diff / dists_safe[:, np.newaxis]
        
        # Gradient of softmin w.r.t pts[i] and pts[j]
        # d(softmin)/d(pts[i]) = - (1/beta) * sum_j w_j * d(pair_dist_ij)/d(pts[i])
        # d(pair_dist_ij)/d(pts[i]) = 0.5 * (pts[i]-pts[j])/d_ij
        # So grad[i] += - (1/beta) * w_j * 0.5 * unit_vec_ij
        # grad[j] += - (1/beta) * w_j * (-0.5 * unit_vec_ij) = + (1/beta) * w_j * 0.5 * unit_vec_ij
        
        # Wait, the objective is to MAXIMIZE soft_min_radius.
        # We are minimizing -soft_min_radius.
        # Let f = soft_min_radius. grad_f[i] = - (1/beta) * sum_j w_j * 0.5 * u_ij
        # We want grad of -f. So grad_neg_f[i] = (1/beta) * sum_j w_j * 0.5 * u_ij
        
        pair_grad_mag = 0.5 * pair_weights / (beta * total_sum)
        
        for idx in range(_NUM_PAIRS):
            i, j = _PAIR_IDX[idx]
            grad[i] += pair_grad_mag[idx] * unit_vecs[idx]
            grad[j] -= pair_grad_mag[idx] * unit_vecs[idx]
            
        # Wall contributions
        # f_wall = min_k (0.5 - |pts[k]|)
        # grad_f_wall[i] = - (1/beta) * sum_k w_k * d(0.5-|pts[i,k]|)/d(pts[i,k])
        # d(0.5-|x|)/dx = -sign(x)
        # grad_f_wall[i] = - (1/beta) * sum_k w_k * (-sign(x)) = (1/beta) * sum_k w_k * sign(x)
        # grad_neg_f_wall[i] = - (1/beta) * sum_k w_k * sign(x)
        
        wall_weights_norm = wall_weights / (beta * total_sum)
        signs = np.sign(pts)
        # Reshape wall_weights to (N, 3)
        wall_weights_reshaped = wall_weights_norm.reshape(N, 3)
        grad -= wall_weights_reshaped * signs
        
    return -soft_min_val, grad

def vectorized_repulsion_anneal(pts, deadline, steps=200, t_start=0.05, t_end=0.001):
    """
    Fast global search using vectorized repulsion forces.
    """
    c = pts.copy()
    best_c = c.copy()
    best_r = compute_radius(c)
    
    # Annealing schedule
    temps = np.linspace(t_start, t_end, steps)
    
    for t in temps:
        if time.time() > deadline:
            break
            
        # Compute forces
        diff = c[_iu] - c[_ju]
        dists_sq = np.sum(diff * diff, axis=-1)
        dists = np.sqrt(dists_sq)
        
        # Avoid division by zero
        dists_sq_safe = np.where(dists_sq < 1e-6, 1e-6, dists_sq)
        
        # Repulsion force magnitude ~ 1/d^2
        force_mag = 1.0 / dists_sq_safe
        # Cap force to prevent explosions
        force_mag = np.clip(force_mag, 0, 100.0)
        
        unit_vecs = diff / dists_safe[:, np.newaxis]
        
        pair_forces = force_mag[:, np.newaxis] * unit_vecs
        
        forces = np.zeros_like(c)
        for idx in range(_NUM_PAIRS):
            i, j = _PAIR_IDX[idx]
            forces[i] += pair_forces[idx]
            forces[j] -= pair_forces[idx]
            
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_dists_sq = np.where(wall_dists**2 < 1e-6, 1e-6, wall_dists**2)
        wall_force_mag = 1.0 / wall_dists_sq
        wall_force_mag = np.clip(wall_force_mag, 0, 100.0)
        
        signs = np.sign(c)
        forces += wall_force_mag * signs
        
        # Update
        # Step size proportional to temperature
        step_size = 0.01 * t * 10 # Scale factor
        noise = np.random.normal(0, t * 0.5, size=c.shape)
        
        c_new = c + forces * step_size + noise
        c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = compute_radius(c_new)
        
        # Metropolis acceptance
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            
        delta_r = r_new - compute_radius(c)
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / (t * 0.1)):
            c = c_new
            
    return best_c, best_r

def voronoi_lloyd_iteration(pts, max_iter=10):
    """
    Apply Lloyd's algorithm using Voronoi cells to balance points.
    """
    c = pts.copy()
    
    for _ in range(max_iter):
        if time.time() > START_TIME + TIME_LIMIT - 10:
            break
            
        try:
            vor = Voronoi(c)
        except Exception:
            break
            
        new_c = c.copy()
        
        for i in range(N):
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            
            if -1 not in region and len(region) > 0:
                # Get vertices of the Voronoi cell
                verts = vor.vertices[region]
                
                # Clip vertices to the cube to calculate centroid inside the valid region
                # Note: This is an approximation. The true centroid of the intersection
                # of the Voronoi cell and the cube is harder to compute exactly.
                # We use the centroid of the clipped vertices as a proxy.
                verts_clipped = np.clip(verts, -0.5, 0.5)
                
                # If the cell is unbounded, Voronoi vertices might be far away.
                # We check if the centroid is within a reasonable distance
                centroid = np.mean(verts_clipped, axis=0)
                
                # Simple heuristic: if the centroid is too far, keep the point
                dist = np.linalg.norm(centroid - c[i])
                if dist < 0.2:
                    new_c[i] = centroid
                else:
                    # Move slightly towards centroid
                    new_c[i] = c[i] + 0.5 * (centroid - c[i])
            else:
                # Point is on the boundary or cell is degenerate
                # Keep point or nudge it slightly inward
                new_c[i] = c[i]
        
        # Clip to cube
        new_c = np.clip(new_c, -0.499999, 0.499999)
        
        # Check for improvement
        r_old = compute_radius(c)
        r_new = compute_radius(new_c)
        
        if r_new >= r_old:
            c = new_c
        else:
            # If it got worse, stop or revert
            break
            
    return c

def local_refine_smooth(pts, deadline):
    """
    Refine using smooth surrogate (L-BFGS-B).
    """
    x = pts.flatten().copy()
    bounds = [(-0.499999, 0.499999)] * (N * 3)
    
    # Use a few different betas to sharpen the objective
    betas = [50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        try:
            def obj(x):
                p = x.reshape(N, 3)
                val, _ = soft_min_radius_and_grad(p, beta)
                return val
            
            def grad(x):
                p = x.reshape(N, 3)
                _, g = soft_min_radius_and_grad(p, beta)
                return g.flatten()
            
            res = minimize(
                obj, x,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
    
    x_2d = x.reshape(N, 3)
    x_2d = np.clip(x_2d, -0.499999, 0.499999)
    
    # Final exact polish with Nelder-Mead
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                neg_radius, x_2d,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            x_2d = res.x
        except Exception:
            pass
            
    x_2d = np.clip(x_2d, -0.499999, 0.499999)
    return x_2d

def get_grid_init(nx=5, ny=5, nz=2):
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    pts += np.random.normal(0, 0.001, size=(N, 3))
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def get_fcc_init():
    fcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                fcc_pts.append([x0, y0, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0, z0 + 0.5 * cell_size])
                fcc_pts.append([x0, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    
    fcc_pts = np.array(fcc_pts)
    fcc_pts -= 0.5
    
    mask = np.all(np.abs(fcc_pts) <= 0.5, axis=1)
    fcc_pts = fcc_pts[mask]
    
    if len(fcc_pts) > N:
        # Select N points with max min-distance (greedy)
        selected = [fcc_pts[0]]
        remaining = fcc_pts[1:]
        while len(selected) < N and len(remaining) > 0:
            min_dists = np.min([np.linalg.norm(remaining - s, axis=1) for s in selected], axis=0)
            idx = np.argmax(min_dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
        fcc_pts = np.array(selected)
        
    if len(fcc_pts) < N:
        n_pad = N - len(fcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        fcc_pts = np.vstack([fcc_pts, pad])
        
    return fcc_pts[:N]

def construct():
    best_pts = None
    best_r = -np.inf
    
    # Generate diverse initializations
    inits = [
        get_grid_init(5, 5, 2),
        get_fcc_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    # Add more random restarts if time permits
    while time.time() - START_TIME < TIME_LIMIT - 40.0:
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Phase 1: Fast Global Search (Repulsion Annealing + Lloyd)
    # This is much faster than scipy.optimize global methods
    candidates = []
    
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
            
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # 1. Annealing
        ann_c, ann_r = vectorized_repulsion_anneal(init_pts, deadline=START_TIME + TIME_LIMIT - 15.0)
        
        # 2. Lloyd's Iteration
        lloyd_c = voronoi_lloyd_iteration(ann_c, max_iter=5)
        
        # 3. Quick local polish to get a better candidate for Phase 2
        # Use a very short L-BFGS run
        if time.time() < START_TIME + TIME_LIMIT - 10.0:
            try:
                x = lloyd_c.flatten()
                bounds = [(-0.499999, 0.499999)] * (N * 3)
                def obj_quick(x):
                    p = x.reshape(N, 3)
                    val, _ = soft_min_radius_and_grad(p, 50.0)
                    return val
                def grad_quick(x):
                    p = x.reshape(N, 3)
                    _, g = soft_min_radius_and_grad(p, 50.0)
                    return g.flatten()
                    
                res = minimize(obj_quick, x, jac=grad_quick, method='L-BFGS-B', bounds=bounds, options={'maxiter': 20})
                lloyd_c = res.x.reshape(N, 3)
            except:
                pass
                
        r = compute_radius(lloyd_c)
        if r > best_r:
            best_r = r
            best_pts = lloyd_c.copy()
            
        candidates.append(lloyd_c)

    # Phase 2: High-Precision Refinement of Top Candidates
    # Sort candidates by radius and refine the top few
    candidates_with_r = [(compute_radius(c), c) for c in candidates]
    candidates_with_r.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top 3 candidates
    for r, c in candidates_with_r[:3]:
        if time.time() - START_TIME > TIME_LIMIT - 5.0:
            break
            
        refined = local_refine_smooth(c, deadline=START_TIME + TIME_LIMIT - 2.0)
        r_ref = compute_radius(refined)
        
        if r_ref > best_r:
            best_r = r_ref
            best_pts = refined.copy()

    # Final safety check
    if best_pts is None:
        best_pts = get_grid_init(5, 5, 2)
        
    return best_pts

if __name__ == "__main__":
    # Self-test
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Time: {time.time() - START_TIME}s")
