"""
Kinetic Annealing + Active Set Polish packer for 100 equal spheres in a unit cube.

Strategy:
1. Initialization: A 5x5x4 grid, slightly jittered to break symmetry.
2. Kinetic Annealing (Global/Intermediate Search):
   - Treat points as particles.
   - Forces: 
     a) Pairwise repulsion: F_rep = k * (1/d^2) * direction (for d < 2r_target)
     b) Wall repulsion: F_wall = k_w * (1/margin^2) * direction (for margin < r_target)
     c) Thermal noise: Gaussian noise scaled by temperature T.
   - Dynamics: Velocity Verlet or simple Euler with damping.
   - Annealing: T starts high, decays exponentially.
   - This allows the system to escape local minima by "jumping" over energy barriers,
     which gradient descent (L-BFGS) often fails to do efficiently in high dimensions.
3. Active Set Polish (Local High-Precision Optimization):
   - Identify the "tight" constraints: pairs with distance close to 2r and points close to walls.
   - Formulate a small QP (or solve via Newton iterations on the active constraints) to
     maximize r subject to these active constraints being exactly satisfied.
   - This step corrects the "sloppy" positions from the simulation to achieve double-precision
     accuracy on the radius limit.

Why this is different:
- Current program uses L-BFGS on a smooth surrogate (log-sum-exp or sum of inverse powers).
  This is deterministic and follows the gradient.
- This program uses stochastic dynamics (MD simulation) to explore the landscape, 
  which is often more robust for packing problems with many local minima.
- The polish step is geometric (active set) rather than gradient-based on a surrogate.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _grid_init(jitter=0.0):
    """Generate a 5x5x4 grid with optional jitter."""
    nx, ny, nz = 5, 5, 4
    # Optimal 1D spacing for n points in [-0.5, 0.5] to maximize min dist to wall and neighbor
    # is not simply linear if we want to maximize radius.
    # However, for initialization, a standard grid is fine.
    # Let's use a grid that respects the aspect ratio roughly.
    # x: 5 pts, y: 5 pts, z: 4 pts.
    # To maximize initial min dist, we can solve for optimal 1D positions.
    # For n=5 in [-0.5, 0.5], optimal positions are roughly equidistant including walls?
    # Actually, just use linspace and let the optimizer fix it.
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    if jitter > 0:
        points += np.random.normal(0, jitter, points.shape)
        points = np.clip(points, -0.49, 0.49)
        
    return points

def _kinetic_anneal(points, time_limit=10.0, target_r=None):
    """
    Perform kinetic annealing to maximize the minimum distance.
    Uses a repulsive force model.
    """
    pts = points.copy().astype(np.float64)
    vels = np.zeros_like(pts)
    
    # Parameters
    dt = 0.005
    damping = 0.95
    k_rep = 1.0
    k_wall = 1.0
    
    # Initial temperature
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.99
    
    start_time = time.time()
    last_improve_time = start_time
    best_r = _score(pts)
    best_pts = pts.copy()
    
    # Determine target radius for force calculation
    # We want to push points apart until they hit the "wall" of the current best r
    # Or just use a fixed "soft" target that increases over time?
    # Better: Use the current min distance as the "equilibrium" distance.
    
    for step in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Calculate forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # Only calculate for pairs closer than 2 * (best_r + margin) to save time?
        # For N=100, O(N^2) is 10k ops, which is fast in NumPy.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # We want to maximize min_dist. 
        # Force should be strong when dist < 2 * r_target.
        # Let r_target be the current best_r * 1.01 to encourage improvement.
        r_target = best_r * 1.02
        cutoff = 2 * r_target + 0.05
        
        # Mask for pairs within cutoff
        mask = dists < cutoff
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-6, dists, 1e-6)
        
        # Force magnitude: proportional to 1/d^2
        # Direction: diff / d
        # F_ij = k * (1/d^2) * (diff_ij / d) = k * diff_ij / d^3
        # Sum over j for each i.
        
        # To be efficient, we can compute the sum for all pairs at once.
        # force_sum = sum_j (k * diff_ij / d_ij^3)
        
        # Create a tensor of forces? No, just loop or use broadcasting carefully.
        # With N=100, a loop over 100 points is fine.
        for i in range(N):
            f_i = np.zeros(3)
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < cutoff:
                    # Repulsive force pushing i away from j
                    # Vector from j to i is diff[i, j]
                    # Force on i is in direction of diff[i, j]
                    f_i += k_rep * diff[i, j] / (d ** 3)
            forces[i] = f_i
            
        # Wall repulsion
        # For each point, if close to wall, push inward.
        # Walls at +/- 0.5.
        # Margin = 0.5 - |coord|
        # If margin < r_target + 0.05, push inward.
        for i in range(N):
            for k in range(3):
                coord = pts[i, k]
                margin = 0.5 - abs(coord)
                if margin < r_target + 0.05:
                    # Push away from wall
                    # If coord > 0, push negative. If coord < 0, push positive.
                    # Force magnitude ~ 1/margin^2
                    f_wall = k_wall / (max(margin, 1e-6) ** 2)
                    forces[i, k] -= np.sign(coord) * f_wall
        
        # Update velocities
        vels = vels * damping + forces * dt
        
        # Add thermal noise
        noise = np.random.normal(0, T, pts.shape)
        vels += noise
        
        # Update positions
        pts = pts + vels * dt
        
        # Clip to box to prevent flying out
        pts = np.clip(pts, -0.5, 0.5)
        
        # Check score
        current_r = _score(pts)
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
            last_improve_time = time.time()
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # If no improvement for a long time, reheat slightly?
        # Or just let it settle.
        
    return best_pts, best_r

def _polish_active_set(points, max_iter=50):
    """
    Polish the solution by identifying active constraints and solving a QP-like step.
    We approximate the QP by moving points in the direction that increases the
    minimum slack of the active constraints.
    """
    pts = points.copy()
    r = _score(pts)
    
    for _ in range(max_iter):
        # Calculate distances and margins
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        margin = 0.5 - np.abs(pts).max(axis=1)
        
        # Identify active constraints
        # Pair constraints: dist < 2r + eps
        eps = 1e-4
        active_pairs = np.argwhere(dists < 2 * r + eps)
        # Wall constraints: margin < r + eps
        active_walls = np.argwhere(margin < r + eps)
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
            
        # Construct a system to find a step direction d such that:
        # For active pairs (i,j): grad_dist . d >= 0 (increase dist)
        # For active walls (i,k): grad_margin . d >= 0 (increase margin)
        
        # This is a linear programming problem: maximize min(slack + grad.slack)
        # We can use a simple heuristic: average gradient of active constraints.
        
        grad_sum = np.zeros_like(pts)
        
        # Pair gradients
        for i, j in active_pairs:
            if i == j: continue
            d = dists[i, j]
            if d < 1e-9: continue
            # grad of dist w.r.t p_i is (p_i - p_j) / d
            # grad of dist w.r.t p_j is (p_j - p_i) / d
            grad_vec = (pts[i] - pts[j]) / d
            grad_sum[i] += grad_vec
            grad_sum[j] -= grad_vec
            
        # Wall gradients
        for i, k in active_walls:
            # margin = 0.5 - |p_ik|
            # grad w.r.t p_ik is -sign(p_ik)
            # We want to increase margin, so move in direction -sign(p_ik)
            grad_sum[i, k] -= np.sign(pts[i, k])
            
        # Normalize step
        step_norm = np.linalg.norm(grad_sum)
        if step_norm > 1e-9:
            step = grad_sum / step_norm
            # Try a small step
            alpha = 0.01
            new_pts = pts + alpha * step
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
            new_r = _score(new_pts)
            if new_r > r:
                pts = new_pts
                r = new_r
            else:
                # If step didn't work, try smaller or different direction
                # For now, break to avoid wasting time
                break
                
    return pts, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial solution
    points = _grid_init(jitter=0.01)
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Kinetic Annealing from grid
    if time.time() < deadline - 60:
        p1, s1 = _kinetic_anneal(points, time_limit=30.0)
        if s1 > best_score:
            best_points = p1
            best_score = s1
            
    # Phase 2: Polish the best found so far
    if time.time() < deadline - 20:
        p_polish, s_polish = _polish_active_set(best_points, max_iter=100)
        if s_polish > best_score:
            best_points = p_polish
            best_score = s_polish
            
    # Phase 3: Multi-start with Kinetic Annealing
    # We have ~80s left. Each anneal takes ~30s. So we can do ~2 more starts.
    # Let's do 2 diverse starts.
    
    starts = []
    # Start 2: Jittered grid
    starts.append(_grid_init(jitter=0.05))
    # Start 3: Random uniform
    starts.append(np.random.uniform(-0.45, 0.45, (N, 3)))
    
    for i, start_pts in enumerate(starts):
        if time.time() < deadline - 40:
            p_new, s_new = _kinetic_anneal(start_pts, time_limit=20.0)
            if s_new > best_score:
                best_points = p_new
                best_score = s_new
                
            # Polish this new candidate
            if time.time() < deadline - 20:
                p_polish, s_polish = _polish_active_set(p_new, max_iter=50)
                if s_polish > best_score:
                    best_points = p_polish
                    best_score = s_polish
            
    # Final Polish on the absolute best
    if time.time() < deadline - 5:
        p_final, s_final = _polish_active_set(best_points, max_iter=200)
        if s_final > best_score:
            best_points = p_final
            best_score = s_final

    return best_points

# EVOLVE-BLOCK-END
