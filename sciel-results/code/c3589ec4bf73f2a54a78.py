"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def score(pts):
        """Calculate the squared ratio of max to min pairwise distance."""
        if len(pts) < 2:
            return 1e10
        dists = pdist(pts)
        if len(dists) == 0:
            return 1e10
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        """Center and scale to unit RMS radius."""
        pts = pts - np.mean(pts, axis=0)
        rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if rms > 1e-9:
            pts = pts / rms
        return pts

    def refine_with_forces(pts, n_steps=1000, learning_rate=0.01):
        """
        Refine points using a force-directed approach.
        We want to maximize the minimum distance.
        Force on i from j:
        If d_ij < d_target, repel.
        To maximize min distance, we can use a potential that grows rapidly when distances are small.
        Let's use a simple repulsive force: F = k * (1/d^2 - 1/d0^2) * r_vec
        But we don't know d0.
        
        Alternative: Gradient ascent on min distance?
        The minimum distance is determined by the closest pair.
        Let (i, j) be the closest pair.
        We want to increase d_ij.
        Move i away from j, and j away from i.
        However, this might decrease other distances.
        
        A robust method is to minimize the sum of exp(-alpha * d_ij^2) for large alpha.
        This effectively pushes apart close pairs.
        """
        pts = pts.copy()
        alpha = 50.0
        lr = 0.01
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
                
            # Compute distances
            dists = squareform(pdist(pts))
            np.fill_diagonal(dists, 1.0) # Avoid div by zero in gradient calc if needed, though we use exp
            
            # Objective: minimize sum(exp(-alpha * d_ij^2))
            # This is equivalent to maximizing the "soft" minimum distance.
            # Gradient wrt pts[i]:
            # d/dx_i sum_j exp(-alpha * ||x_i - x_j||^2)
            # = sum_j exp(-alpha * d_ij^2) * (-2 * alpha) * (x_i - x_j)
            
            # Vectorized gradient calculation
            # dists_sq is N x N
            dists_sq = dists**2
            weights = np.exp(-alpha * dists_sq)
            np.fill_diagonal(weights, 0.0)
            
            # grad[i] = -2 * alpha * sum_j weights[i,j] * (x_i - x_j)
            # = -2 * alpha * (x_i * sum_j weights[i,j] - sum_j weights[i,j] * x_j)
            
            sum_w = np.sum(weights, axis=1)
            sum_wx = weights @ pts
            
            grad = -2.0 * alpha * (pts * sum_w[:, np.newaxis] - sum_wx)
            
            # Normalize gradient to keep steps stable
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-9:
                grad = grad / grad_norm
            
            pts = pts + lr * grad
            pts = normalize(pts)
            
        return pts

    # Strategy:
    # 1. Generate diverse initial configurations based on symmetry.
    # 2. Refine each using the force simulation.
    # 3. Keep the best one.
    
    candidates = []
    
    # Candidate 1: 4x4 Grid (perturbed)
    side = 4
    x = np.arange(side) - (side - 1) / 2.0
    y = np.arange(side) - (side - 1) / 2.0
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    candidates.append(grid_pts)
    
    # Candidate 2: Two concentric squares (4 points inner, 12 points outer? No, 4+12=16)
    # Let's try 4 inner, 12 outer.
    # Or 8 inner, 8 outer.
    # Let's try 4 inner (square), 12 outer (3x4 grid on a larger square? No, distinct points).
    # Let's try 4 inner (square), 12 outer (regular 12-gon? No, 12 points on a circle).
    # 4 points on circle of radius r1, 12 points on circle of radius r2.
    # Total 16.
    # Let's try 8 and 8.
    
    def make_concentric(n_inner, n_outer, r_inner, r_outer, angle_offset=0):
        pts = []
        if n_inner > 0:
            angles_in = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + angle_offset
            pts.extend([(r_inner * np.cos(a), r_inner * np.sin(a)) for a in angles_in])
        if n_outer > 0:
            angles_out = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + angle_offset + np.pi/n_outer
            pts.extend([(r_outer * np.cos(a), r_outer * np.sin(a)) for a in angles_out])
        return np.array(pts)
    
    # Try various ratios of inner/outer radii
    for r_ratio in [0.3, 0.5, 0.7, 0.8, 0.9]:
        # 8 inner, 8 outer
        pts = make_concentric(8, 8, r_ratio, 1.0)
        candidates.append(pts)
        
        # 4 inner, 12 outer
        pts = make_concentric(4, 12, r_ratio, 1.0)
        candidates.append(pts)
        
        # 2 inner, 14 outer
        pts = make_concentric(2, 14, r_ratio, 1.0)
        candidates.append(pts)

    # Candidate 3: Hexagonal lattice subset
    # Generate a hex grid and pick 16 closest to center
    # Hex grid spacing 1
    hex_pts = []
    R = 3
    for q in range(-R, R+1):
        for r in range(-R, R+1):
            if abs(q+r) <= R:
                x = 1.5 * q
                y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                hex_pts.append((x,y))
    hex_pts = np.array(hex_pts)
    # Center it
    hex_pts = hex_pts - np.mean(hex_pts, axis=0)
    # Sort by distance to center
    dists_to_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists_to_center)
    # Take first 16
    hex_subset = hex_pts[idx[:16]]
    candidates.append(hex_subset)
    
    # Candidate 4: Random points on a disk (a few seeds)
    rng = np.random.default_rng(42)
    for _ in range(5):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        # Ensure distinctness by adding small noise if collision (prob low)
        rand_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(rand_pts)
        
    # Candidate 5: 4x4 grid with corners cut? Or rotated grid?
    # Let's add a rotated 4x4 grid
    angle = np.pi/8
    c, s = np.cos(angle), np.sin(angle)
    rot_mat = np.array([[c, -s], [s, c]])
    rot_grid = grid_pts @ rot_mat.T
    candidates.append(rot_grid)
    
    best_score = 1e10
    best_pts = candidates[0]
    
    # Refine each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        cand = normalize(cand)
        
        # Run force refinement
        # We might need multiple passes with different alpha/lr
        refined = refine_with_forces(cand, n_steps=2000, learning_rate=0.05)
        
        # Evaluate
        sc = score(refined)
        
        if sc < best_score:
            best_score = sc
            best_pts = refined.copy()
            
        # Also try a second pass with finer steps on the best result so far?
        # Or just keep it simple.
        
    # Final polish: Use scipy minimize on the smooth objective for the best candidate found
    # This combines the global search (symmetry) with local precision (gradient).
    
    def smooth_objective(x, alpha=50.0):
        pts = x.reshape(N, 2)
        d_sq = squareform(pdist(pts))**2
        np.fill_diagonal(d_sq, 1.0)
        # Minimize sum(exp(-alpha * d_sq))
        # This is equivalent to maximizing min distance roughly
        return np.sum(np.exp(-alpha * d_sq))
        
    def smooth_grad(x, alpha=50.0):
        pts = x.reshape(N, 2)
        dists = squareform(pdist(pts))
        dists_sq = dists**2
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.exp(-alpha * dists_sq)
        np.fill_diagonal(weights, 0.0)
        
        sum_w = np.sum(weights, axis=1)
        sum_wx = weights @ pts
        
        grad = -2.0 * alpha * (pts * sum_w[:, np.newaxis] - sum_wx)
        return grad.flatten()
        
    if time.time() < deadline - 5:
        from scipy.optimize import minimize
        res = minimize(
            smooth_objective,
            best_pts.flatten(),
            jac=smooth_grad,
            args=(50.0,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success:
            final_pts = res.x.reshape(N, 2)
            final_sc = score(final_pts)
            if final_sc < best_score:
                best_pts = final_pts
                best_score = final_sc

    return best_pts

# EVOLVE-BLOCK-END
