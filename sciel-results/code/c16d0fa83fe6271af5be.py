"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    """
    Place 100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. Multi-start initialization with different grid topologies (5x5x4, 10x10x1, 5x10x2).
    2. Two-phase soft-min gradient ascent with momentum (coarse->fine alpha).
    3. Vectorized gradient calculation using log-sum-exp.
    4. Simulated annealing acceptance for escaping local minima.
    5. High-precision polish on critical pairs/boundaries with anti-stall nudges.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # --- Helper Functions ---
    
    def compute_radius(pts):
        """Compute the maximum valid radius for given points (vectorized)."""
        if not np.all(np.abs(pts) <= 0.5 + 1e-12):
            return 0.0
        
        n = len(pts)
        # Use cdist for efficient pairwise distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    def compute_radius_and_crit(pts):
        """Compute radius and return the critical pair/boundary indices."""
        if not np.all(np.abs(pts) <= 0.5 + 1e-12):
            return 0.0, None, None
        
        n = len(pts)
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum pair
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_dist = dists[min_pair_idx]
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
        min_boundary = boundary_dists[min_boundary_idx]
        
        if min_pair_dist / 2.0 <= min_boundary:
            r = min_pair_dist / 2.0
            i_idx, j_idx = min_pair_idx
            if i_idx > j_idx:
                i_idx, j_idx = j_idx, i_idx
            crit_pair = (i_idx, j_idx)
            crit_boundary = None
        else:
            r = min_boundary
            crit_pair = None
            crit_boundary = min_boundary_idx
        
        return r, crit_pair, crit_boundary
    
    def soft_min_grad(pts, alpha):
        """
        Compute gradient of soft-min radius w.r.t. point positions.
        Uses log-sum-exp for numerical stability.
        """
        n = len(pts)
        # Pairwise distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        # Get upper triangle indices
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu] / 2.0
        pair_diffs = pts[iu[0]] - pts[iu[1]]
        
        # Boundary terms
        boundary_terms = 0.5 - np.abs(pts)
        boundary_flat = boundary_terms.flatten()
        
        all_terms = np.concatenate([pair_dists, boundary_flat])
        
        # Log-sum-exp for numerical stability
        neg_alpha_x = -alpha * all_terms
        max_val = np.max(neg_alpha_x)
        exp_vals = np.exp(neg_alpha_x - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(pts)
        
        # Pairwise term gradients
        pair_weights = weights[:len(pair_dists)]
        safe_dists = np.maximum(dists[iu], 1e-10)
        unit_vecs = pair_diffs / safe_dists[:, np.newaxis]
        
        i_idx, j_idx = iu
        np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * unit_vecs)
        
        # Boundary term gradients
        boundary_weights = weights[len(pair_dists):]
        bw = boundary_weights.reshape(n, 3)
        signs = np.sign(pts)
        grad -= bw * signs
        
        return grad

    def run_gradient_optimization(init_pts, time_budget):
        """
        Runs the two-phase gradient ascent + annealing on a given initialization.
        Returns the best points and radius found within the time budget.
        """
        current_pts = init_pts.copy()
        current_radius = compute_radius(current_pts)
        best_radius = current_radius
        best_pts = current_pts.copy()
        
        # Phase 1: Coarse optimization
        T = 0.003
        lr = 0.02
        alpha = 25.0
        momentum = np.zeros_like(current_pts)
        beta = 0.9
        no_improve = 0
        
        phase1_end = time.time() + time_budget * 0.7
        
        while time.time() < phase1_end:
            grad = soft_min_grad(current_pts, alpha)
            
            # Momentum update
            momentum = beta * momentum + (1 - beta) * grad
            new_pts = current_pts + lr * momentum
            new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_radius = compute_radius(new_pts)
            
            if new_radius >= current_radius:
                current_radius = new_radius
                current_pts = new_pts
                no_improve = 0
                T *= 0.999
                lr *= 0.9998
                alpha *= 1.0005
                if lr < 0.003:
                    lr = 0.008
            else:
                delta = new_radius - current_radius
                if np.random.random() < np.exp(delta / max(T, 1e-8)):
                    current_radius = new_radius
                    current_pts = new_pts
                no_improve += 1
                T *= 0.98
                lr *= 0.96
                if lr < 0.003:
                    if no_improve > 100:
                        # Perturb to escape local minimum
                        idx = np.random.choice(N, 4, replace=False)
                        current_pts[idx] += np.random.uniform(-0.02, 0.02, (4, 3))
                        current_pts = np.clip(current_pts, -0.5, 0.5)
                        current_radius = compute_radius(current_pts)
                        no_improve = 0
                        lr = 0.02
                        T = 0.003
                        alpha = 25.0
                        momentum = np.zeros_like(current_pts)
                    else:
                        lr = 0.02
                        T = 0.003
                        alpha = 25.0
                        momentum = np.zeros_like(current_pts)
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
        
        # Phase 2: Fine optimization
        alpha = 80.0
        lr = 0.005
        momentum = np.zeros_like(current_pts)
        beta = 0.9
        no_improve = 0
        T = 0.001
        
        phase2_end = time.time() + time_budget * 0.3
        
        while time.time() < phase2_end:
            grad = soft_min_grad(current_pts, alpha)
            
            momentum = beta * momentum + (1 - beta) * grad
            new_pts = current_pts + lr * momentum
            new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_radius = compute_radius(new_pts)
            
            if new_radius >= current_radius:
                current_radius = new_radius
                current_pts = new_pts
                no_improve = 0
                lr *= 0.999
                alpha *= 1.0005
                if lr < 0.001:
                    lr = 0.003
            else:
                delta = new_radius - current_radius
                if np.random.random() < np.exp(delta / max(T, 1e-8)):
                    current_radius = new_radius
                    current_pts = new_pts
                no_improve += 1
                T *= 0.95
                lr *= 0.95
                if lr < 0.001:
                    if no_improve > 50:
                        idx = np.random.choice(N, 3, replace=False)
                        current_pts[idx] += np.random.uniform(-0.01, 0.01, (3, 3))
                        current_pts = np.clip(current_pts, -0.5, 0.5)
                        current_radius = compute_radius(current_pts)
                        no_improve = 0
                        lr = 0.005
                        T = 0.001
                        alpha = 80.0
                        momentum = np.zeros_like(current_pts)
                    else:
                        lr = 0.005
                        T = 0.001
                        alpha = 80.0
                        momentum = np.zeros_like(current_pts)
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
        
        return best_pts, best_radius

    def polish_critical(pts, r, time_budget):
        """
        High-precision polish: iteratively adjust the critical pair and boundary
        to equalize all constraints. Includes anti-stall random nudges.
        """
        pts = pts.copy()
        r = r
        stall_count = 0
        last_time = time.time()
        
        while time.time() < last_time + time_budget:
            if time.time() > deadline - 2.0:
                break
            
            _, crit_pair, crit_boundary = compute_radius_and_crit(pts)
            
            moved = False
            
            if crit_pair is not None:
                i, j = crit_pair
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    # Coincident points, separate them randomly
                    pts[i] += np.random.uniform(-0.01, 0.01, 3)
                    pts[j] -= np.random.uniform(-0.01, 0.01, 3)
                    moved = True
                else:
                    target_r = r
                    deficit = 2 * target_r - dist
                    if deficit > 1e-14:
                        unit = d / dist
                        step = 0.5 * deficit * 0.98
                        pts[i] += step * unit
                        pts[j] -= step * unit
                        moved = True
            elif crit_boundary is not None:
                i, k = crit_boundary
                sign = np.sign(pts[i, k])
                if sign == 0:
                    sign = 1
                deficit = 0.5 - np.abs(pts[i, k]) - r
                if deficit > 1e-14:
                    pts[i, k] += sign * 0.98 * deficit
                    moved = True
            
            # Clip to cube
            pts = np.clip(pts, -0.5 + 1e-12, 0.5 - 1e-12)
            
            new_r = compute_radius(pts)
            
            if new_r > r + 1e-15:
                r = new_r
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 20:
                    # Anti-stall: small random nudge to critical points
                    if crit_pair is not None:
                        i, j = crit_pair
                        pts[i] += np.random.uniform(-0.005, 0.005, 3)
                        pts[j] += np.random.uniform(-0.005, 0.005, 3)
                    elif crit_boundary is not None:
                        i, k = crit_boundary
                        pts[i, k] += np.random.uniform(-0.005, 0.005)
                    
                    pts = np.clip(pts, -0.5 + 1e-12, 0.5 - 1e-12)
                    stall_count = 0
                    if time.time() > deadline - 5.0:
                         break

        return pts, r

    # --- Multi-Start Initialization ---
    
    np.random.seed(42)
    
    # Define different grid topologies
    grids = []
    
    # 1. 5x5x4 Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    base1 = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    grids.append(base1 + np.random.uniform(-0.02, 0.02, (N, 3)))
    
    # 2. 10x10x1 Grid (Layered)
    # This topology often allows for better packing in flat layers
    nx, ny, nz = 10, 10, 1
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.array([0.0])
    base2 = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    # Add jitter to break symmetry and allow 3D optimization to move layers
    grids.append(base2 + np.random.uniform(-0.05, 0.05, (N, 3)))
    
    # 3. 5x10x2 Grid (Intermediate)
    nx, ny, nz = 5, 10, 2
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    base3 = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    grids.append(base3 + np.random.uniform(-0.03, 0.03, (N, 3)))
    
    # Allocate time budget for each start
    # Total search time ~ 80s, Polish time ~ 30s
    search_time_total = 80.0
    time_per_start = search_time_total / len(grids)
    
    best_pts = None
    best_radius = -1.0
    
    for i, init_grid in enumerate(grids):
        if time.time() > deadline - 35.0:
            break
            
        init_pts = np.clip(init_grid, -0.5, 0.5)
        
        # Run gradient optimization for this start
        pts, radius = run_gradient_optimization(init_pts, time_per_start)
        
        if radius > best_radius:
            best_radius = radius
            best_pts = pts.copy()
            
    # Final Polish on the best result
    if best_pts is not None and time.time() < deadline - 5.0:
        polish_time = min(deadline - time.time() - 5.0, 30.0)
        if polish_time > 1.0:
            best_pts, best_radius = polish_critical(best_pts, best_radius, polish_time)
    
    # Final verification and clipping
    if best_pts is None:
        # Fallback: return a simple grid if something went wrong
        best_pts = grids[0]
        
    best_pts = np.clip(best_pts, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_pts
# EVOLVE-BLOCK-END
