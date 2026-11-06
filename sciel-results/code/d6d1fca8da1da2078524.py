"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _smooth_min(x, k=100.0):
    """
    Compute a smooth approximation of the minimum of a set of values.
    Uses the log-sum-exp trick: -1/k * log(sum(exp(-k*x_i)))
    This is differentiable and approximates min(x) as k -> infinity.
    """
    # Shift for numerical stability
    x_min = np.min(x)
    # If x is very small, exp(-k*(x-x_min)) might overflow if x-x_min is large positive?
    # No, x >= x_min, so -k*(x-x_min) <= 0. exp is safe.
    # However, if x is very negative, exp is small.
    # Let's use: -1/k * log(sum(exp(-k * (x - x_min)))) + x_min
    # This is numerically stable.
    
    # Handle edge case where x might contain -inf or similar, but we assume valid centers
    shifted = x - x_min
    # To avoid underflow for large k and large shifted values, we can't just sum exp.
    # But shifted >= 0, so exp(-k*shifted) <= 1.
    # Sum is at least 1 (from the min element).
    # log(sum) is fine.
    
    # Clip shifted to prevent extreme values if k is huge? 
    # k=100, shifted could be up to 0.5. exp(-50) is ~1e-22, which is fine for double.
    vals = np.exp(-k * shifted)
    s = np.sum(vals)
    if s == 0:
        return x_min
    return x_min - (1.0 / k) * np.log(s)

def _radius_score(centers, k=100.0):
    """
    Compute a smooth proxy for the radius.
    Radius is min( min_pair_dist/2, min_wall_dist ).
    We want to maximize this.
    Let's define a smooth function that is close to the radius.
    
    Actually, for optimization, we want to minimize -radius.
    Or maximize radius.
    
    Let d_pair be the pairwise distances.
    Let d_wall be the wall distances.
    
    We want to maximize min( min(d_pair)/2, min(d_wall) ).
    
    Let's use the smooth min on the set of all constraints:
    Constraints:
    1. d_pair[i,j] / 2 for all i<j
    2. d_wall[i,k] for all i,k
    
    R_smooth = smooth_min( [d_pair[i,j]/2, d_wall[i,k]] )
    
    This value is close to the true radius.
    We will maximize this.
    """
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.any(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    # Distance to wall in dimension d is 0.5 - |c_d|
    # We need the minimum of these for each sphere, then the minimum over spheres?
    # No, the constraint is that ALL spheres must be inside.
    # So the radius is limited by the MINIMUM wall distance over ALL spheres and dimensions.
    wall_dists = 0.5 - np.abs(centers) # shape (N, 3)
    
    # Pairwise distances
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # We need the minimum of (dist/2) and (wall_dists).
    # Flatten them into one array for the smooth min.
    # Size: N*(N-1)/2 + N*3
    # For N=50, ~1225 + 150 = ~1375 values.
    
    # Extract upper triangle for pairs to avoid duplicates (though min is same)
    # Actually, just taking all i!=j is fine for min, but doubles computation.
    # Let's take i < j.
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dist[i_idx, j_idx] / 2.0
    
    # Combine
    all_constraints = np.concatenate([pair_dists, wall_dists.flatten()])
    
    # Compute smooth min
    r_smooth = _smooth_min(all_constraints, k=k)
    
    return r_smooth

def _neg_radius_score(centers_flat, k=100.0):
    """Negative smooth radius for minimization."""
    centers = centers_flat.reshape(N, 3)
    # Clip to valid range to avoid huge penalties or invalid states during optimization
    # The optimizer should handle bounds, but let's be safe.
    centers = np.clip(centers, -0.499, 0.499)
    return -_radius_score(centers, k=k)

def _true_radius(centers):
    """Compute the exact radius for scoring."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)

def _make_bcc_initial(seed=42):
    """
    Create an initial configuration based on a distorted BCC lattice.
    BCC has higher density than Simple Cubic.
    We generate a 4x4x4 grid of BCC unit cells (or similar) and select 50 points.
    """
    rng = np.random.RandomState(seed)
    
    # BCC unit cell has points at (0,0,0) and (0.5, 0.5, 0.5) (relative to cell size)
    # Let's create a grid of cells.
    # If we use a 4x4x4 grid of cells, we have 64 cells.
    # Each cell has 2 points. Total 128 points.
    # We need 50.
    
    m = 4
    cell_size = 1.0 / m
    ticks = (np.arange(m) + 0.5) * cell_size - 0.5 # Centers of cells?
    # Actually, let's define the lattice points explicitly.
    
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner of the cell (shifted to center of cube)
                x = ix * cell_size - 0.5 + cell_size/2
                y = iy * cell_size - 0.5 + cell_size/2
                z = iz * cell_size - 0.5 + cell_size/2
                # This puts points at cell centers.
                # BCC: Add center of cell.
                cx = x
                cy = y
                cz = z
                # Wait, standard BCC in a cube [-0.5, 0.5]:
                # Points at corners and center.
                # If we tile the cube with m^3 sub-cubes, we can place BCC points in each.
                # But boundaries might cause overlaps or gaps.
                
                # Let's use a simpler approach:
                # Generate points on a grid with spacing d.
                # BCC structure: points at (i*d, j*d, k*d) and ((i+0.5)*d, (j+0.5)*d, (k+0.5)*d).
                # To fit in [-0.5, 0.5], we need to scale.
                
                # Let's just use a dense grid and pick the best ones?
                # Or, generate a 4x4x4 grid of "corners" and 4x4x4 grid of "centers".
                # Total 2 * 4^3 = 128 points.
                
                # Corner-like points (shifted to avoid boundary)
                px = ix * cell_size - 0.5 + cell_size/2
                py = iy * cell_size - 0.5 + cell_size/2
                pz = iz * cell_size - 0.5 + cell_size/2
                points.append([px, py, pz])
                
                # Body center of this cell
                bcx = px
                bcy = py
                bcz = pz
                # Wait, if px is the center of the cell, where is the body center?
                # The cell is [px-cell_size/2, px+cell_size/2].
                # The body center is at the same location as the "corner" if we define it that way?
                # No.
                # Let's define the grid points at (i+0.5)*d - 0.5.
                # And the BCC points at (i+0.5)*d - 0.5 + d/2? No.
                
                # Let's restart the BCC construction properly.
                # Lattice vectors for BCC: a/2 (1,1,1), a/2(1,-1,1), a/2(-1,1,1) etc.
                # Simpler: Points at (i,j,k)*a and (i+0.5, j+0.5, k+0.5)*a.
                # We need to fit 50 points in size 1.
                # Volume per point in BCC is a^3/2.
                # 50 * a^3/2 <= 1 => a^3 <= 0.04 => a <= 0.34.
                # If a=0.34, 1/0.34 ~ 3. So a 3x3x3 grid of BCC cells gives 3^3 * 2 = 54 points.
                # This is very close to 50.
                
    # Let's use a 3x3x3 BCC grid (54 points) and remove 4 random ones?
    # Or just take the first 50.
    
    m = 3
    a = 1.0 / m # Side length of BCC unit cell
    # Points in one BCC cell: (0,0,0) and (0.5, 0.5, 0.5) relative to cell origin.
    # We want to center the whole structure in [-0.5, 0.5].
    # Total extent is m*a = 1.
    # So cell origins are at -0.5 + i*a.
    
    pts = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                ox = -0.5 + ix * a
                oy = -0.5 + iy * a
                oz = -0.5 + iz * a
                # Corner
                pts.append([ox, oy, oz])
                # Body center
                pts.append([ox + a/2, oy + a/2, oz + a/2])
    
    pts = np.array(pts)
    # We have 54 points. We need 50.
    # The points at the corners of the big cube (-0.5, -0.5, -0.5) etc are on the boundary.
    # They will have radius limited by wall distance 0.
    # We should shift them inwards or remove them.
    # Actually, the task says "inside the cube". Centers can be on the boundary?
    # "Place the centres ... inside the unit cube". Usually implies strictly inside or on boundary.
    # But if a center is on the boundary, radius is 0.
    # So we MUST shift them inwards.
    
    # Let's add a small jitter and shift inwards.
    # Better: Scale the whole structure slightly to fit inside?
    # If we scale by factor s < 1, the points move inwards.
    # But the distances also shrink.
    # It's better to optimize positions.
    
    # Let's just take 50 points from the 54, preferring interior ones.
    # Distance to boundary:
    d_bound = 0.5 - np.abs(pts)
    min_d = np.min(d_bound, axis=1)
    # Sort by min distance to boundary (descending)
    order = np.argsort(-min_d)
    # Take first 50
    centers = pts[order[:N]].copy()
    
    # Add small random jitter to break symmetry
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    # Clip to ensure they are strictly inside
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Initializations
    inits = [
        _make_bcc_initial(42),
        _make_bcc_initial(123),
        # Add a random init as fallback
        np.random.RandomState(1).uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    for init_centers in inits:
        if time.time() > deadline - 10.0:
            break
            
        current = init_centers.copy()
        current_radius = _true_radius(current)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current.copy()
            
        # Simulated Annealing
        # Objective: Maximize smooth radius
        # We minimize -smooth_radius
        
        x = current.flatten()
        
        # Parameters
        T_start = 0.01
        T_end = 1e-6
        alpha = 0.995 # Cooling rate
        max_iters = 5000
        step_size = 0.01
        
        f_curr = -_radius_score(current, k=50.0)
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            T = T_start * (alpha ** i)
            if T < T_end:
                break
                
            # Generate new solution
            # Perturb a few random coordinates
            n_perturb = np.random.randint(1, 4)
            idxs = np.random.choice(N * 3, n_perturb, replace=False)
            
            x_new = x.copy()
            for idx in idxs:
                x_new[idx] += np.random.normal(0, step_size)
                # Ensure bounds
                if x_new[idx] > 0.499: x_new[idx] = 0.499
                if x_new[idx] < -0.499: x_new[idx] = -0.499
                
            centers_new = x_new.reshape(N, 3)
            f_new = -_radius_score(centers_new, k=50.0)
            
            delta_f = f_new - f_curr
            
            if delta_f < 0:
                # Improvement
                x = x_new
                f_curr = f_new
                centers = centers_new
                r_check = _true_radius(centers)
                if r_check > best_radius:
                    best_radius = r_check
                    best_centers = centers.copy()
            else:
                # Accept with probability
                prob = np.exp(-delta_f / T)
                if np.random.random() < prob:
                    x = x_new
                    f_curr = f_new
                    centers = centers_new
                    r_check = _true_radius(centers)
                    if r_check > best_radius:
                        best_radius = r_check
                        best_centers = centers.copy()
            
            # Adaptive step size?
            # If we haven't improved in a while, maybe reduce step size?
            # Or if we are stuck, increase?
            # Let's keep it simple.
            
        # Local polish with L-BFGS on the smooth objective
        # This helps to fine-tune the position after SA
        if time.time() < deadline - 5.0:
            from scipy.optimize import minimize
            try:
                res = minimize(
                    _neg_radius_score,
                    x,
                    args=(50.0,),
                    method='L-BFGS-B',
                    bounds=[(-0.499, 0.499)] * (N * 3),
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.success:
                    centers_polished = res.x.reshape(N, 3)
                    r_polished = _true_radius(centers_polished)
                    if r_polished > best_radius:
                        best_radius = r_polished
                        best_centers = centers_polished.copy()
            except Exception:
                pass

    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
