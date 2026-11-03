"""Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x is a flat array of length 3N.
    target_r is the desired radius.
    Returns (energy, gradient).
    Energy is sum of squared violations of the constraints:
    1. Pairwise distance >= 2 * target_r
    2. Wall distance >= target_r
    """
    c = x.reshape(N, 3)
    
    # --- Pairwise Part ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Active pairs: those closer than 2 * target_r
    active = dists < 2.0 * target_r
    # Ensure diagonal is not active
    active[np.diag_indices_from(active)] = False
    
    # Violations: (2r - d) for active pairs, 0 otherwise
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    # Energy is sum of squared violations (0.5 factor for clean gradient)
    energy_pair = 0.5 * np.sum(violations**2)
    
    # Gradient calculation for pairs
    # d/dx_i [ 0.5 * (2r - ||x_i - x_j||)^2 ]
    # = (2r - d) * d/dx_i (2r - d)
    # = (2r - d) * (-(x_i - x_j) / d)
    # = - (2r - d) / d * (x_i - x_j)
    # Note: violations[i,j] = 2r - d. So we need - violations[i,j] / d * (x_i - x_j)
    # To avoid division by zero, we use safe_dists
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # weights[i,j] = violations[i,j]
    # grad_i = - sum_j weights[i,j] * unit_vecs[i,j]
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    # Distance to each wall
    # Walls are at -0.5 and 0.5 for each dimension
    # dist_wall_k = 0.5 - |c_k|
    wall_dists = HALF - np.abs(c)
    
    # Violations: target_r - wall_dist if positive
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    # Gradient for walls
    # d/dc [ (r - (0.5 - |c|))^2 ]
    # = 2 * (r - (0.5 - |c|)) * d/dc (0.5 - |c|)
    # = 2 * wall_violation * (-sign(c))
    signs = np.sign(c)
    # Handle c=0 case: sign(0)=0, which is fine, gradient is 0 at center
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _optimize_bfgs(init_pts, target_r, max_iter=1000, gtol=1e-10, time_limit=50.0):
    """
    Run BFGS optimization with a time limit.
    Returns (optimized_pts, success).
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def _generate_hybrid_init(rng):
    """
    Generate hybrid initialization:
    64 spheres in a 4x4x4 simple cubic lattice.
    36 spheres in interstitial voids (center of each 4x4x4 cell sub-cube).
    
    The 4x4x4 lattice has spacing s.
    The interstitial sites are at the centers of the 3x3x3 sub-cubes formed by the lattice.
    Wait, 4x4x4 lattice has 3x3x3 = 27 interstitial sites in the "center of cube" positions.
    We need 36 more.
    
    Let's rethink the geometry.
    A 4x4x4 grid of points has 64 points.
    The space is divided into 3x3x3 = 27 large cubes by the grid lines? No.
    The grid points are at indices 0,1,2,3 in each dimension.
    The "cells" are between them. There are 3 intervals in x, 3 in y, 3 in z.
    So there are 27 unit cells in the lattice structure.
    The center of each unit cell is a candidate for an interstitial sphere.
    That gives 27 spheres.
    We need 36. So we need 9 more.
    
    Perhaps we can place some on the faces or edges?
    Or maybe a different lattice?
    
    Let's try a 5x5x4 grid = 100 points.
    Spacing: 1/4 in z, 1/5 in x,y.
    This is the BCC-like or simple grid from the seed.
    
    The directive says: "64 spheres in a 4x4x4 simple cubic lattice and 36 spheres in a centered cubic arrangement within the interstitial voids".
    
    Let's stick to the directive's specific numbers.
    4x4x4 lattice: 64 points.
    Interstitial voids:
    In a simple cubic lattice, the primary interstitial sites are:
    1. Body center of each cube: 3x3x3 = 27 sites.
    2. Face centers of each cube: 3 faces * 2 orientations * 3x3 grid = 54 sites?
       Face centers:
       x-faces: 2 * 3 * 3 = 18
       y-faces: 2 * 3 * 3 = 18
       z-faces: 2 * 3 * 3 = 18
       Total face centers = 54.
    
    We need 36 spheres.
    We have 27 body centers.
    We need 9 more. We can pick 9 face centers.
    
    However, placing spheres in interstitial sites of a 4x4x4 lattice might lead to close contacts if the lattice spacing is too small.
    Let's calculate the spacing.
    If we fit 4 points in [-0.5, 0.5], spacing is 1/3? No.
    If we have 4 points, there are 3 gaps.
    Let spacing be s.
    Points at -0.5, -0.5+s, -0.5+2s, -0.5+3s.
    To be symmetric, let's center the lattice.
    Points at -0.5 + s/2, -0.5 + 3s/2, ...
    Actually, let's just define the 4x4x4 grid to fill the cube roughly.
    Ticks: (np.arange(4) + 0.5) / 4 - 0.5
    This gives spacing 0.25.
    Distance between neighbors is 0.25.
    Radius would be 0.125 if only nearest neighbors.
    But interstitial spheres will be closer.
    Distance from lattice point to body center of adjacent cell:
    Lattice point at (0,0,0) relative to cell corner.
    Body center at (0.125, 0.125, 0.125) relative to corner?
    If spacing is 0.25, body center is at (0.125, 0.125, 0.125) from corner.
    Distance = sqrt(3 * 0.125^2) = 0.125 * sqrt(3) ≈ 0.2165.
    This is LARGER than 0.25. So interstitial spheres are further from lattice points than lattice neighbors are from each other?
    Wait.
    Nearest lattice neighbor distance: 0.25.
    Distance to body center: 0.2165.
    So the body center is CLOSER to the lattice points than the lattice points are to each other?
    No. 0.2165 < 0.25.
    So if we put spheres at both lattice points and body centers, the distance between a lattice sphere and an interstitial sphere is 0.2165.
    The distance between two lattice spheres is 0.25.
    The distance between two interstitial spheres (body centers of adjacent cells):
    Body center of cell (0,0,0) is at (0.125, 0.125, 0.125).
    Body center of cell (1,0,0) is at (0.375, 0.125, 0.125).
    Distance = 0.25.
    So the minimum distance in this hybrid packing would be ~0.2165.
    Radius = 0.2165 / 2 = 0.10825.
    This is better than the 0.100 of the simple 5x5x5 grid (spacing 0.2).
    And potentially a good starting point.
    
    Let's implement this.
    1. Create 4x4x4 grid.
    2. Create 3x3x3 body centers.
    3. Select 9 face centers to reach 36 interstitials.
       Which face centers? The ones furthest from other interstitials?
       Or just pick the first 9?
       Let's pick face centers that are not too close to other face centers.
       Actually, any face center is distance 0.125 from 4 lattice points and 2 other face centers?
       Face center at (0.125, 0, 0.125) (on x-face of cell 0,0,0).
       Distance to lattice point (0,0,0) is 0.125.
       Distance to lattice point (0,0,0.25) is 0.125.
       Distance to lattice point (0,0.25,0) is 0.125.
       Distance to lattice point (0,0.25,0.25) is 0.125.
       So face centers are CLOSER to lattice points (0.125) than body centers are (0.2165).
       So if we use face centers, the radius drops to 0.125/2 = 0.0625. That's bad.
       
       So we should ONLY use body centers for interstitials if we want to maintain a decent radius.
       But we only have 27 body centers.
       We need 36.
       
       Maybe the "centered cubic arrangement" implies something else?
       Or maybe we should use a different base lattice?
       
       Let's try 5x5x4 = 100 points.
       Spacing x,y: 0.2. Spacing z: 0.25.
       Min distance 0.2. Radius 0.1.
       
       Let's try to find 36 "good" interstitial sites in a 4x4x4 lattice.
       If we can't find 36 that are far enough, maybe we should just use the 27 body centers and 9 other points that are optimized?
       
       Alternative: Use the 4x4x4 lattice (64 pts) and 36 random points, then optimize.
       The directive says "centered cubic arrangement within the interstitial voids".
       Maybe it means a BCC lattice?
       A BCC lattice with 4x4x4 unit cells has 4*4*4*2 = 128 points. Too many.
       
       Let's reconsider the 4x4x4 + 27 body centers.
       Total 91 points.
       We need 9 more.
       Where can we put 9 more points?
       Maybe at the corners of the big cube?
       The 4x4x4 grid points are at:
       -0.375, -0.125, 0.125, 0.375.
       The cube corners are at -0.5, 0.5.
       Distance from corner (-0.5, -0.5, -0.5) to nearest grid point (-0.375, -0.375, -0.375) is sqrt(3 * 0.125^2) = 0.2165.
       So we can put points at the 8 corners of the cube!
       Distance to grid points is 0.2165.
       Distance to other corners is 1.0.
       Distance to body centers?
       Body center of corner cell (-0.375, -0.375, -0.375) relative to corner?
       Wait, the body center of the cell adjacent to the corner (-0.5, -0.5, -0.5) is at:
       Cell spans [-0.5, -0.25] in x? No.
       Grid points are at -0.375, -0.125, 0.125, 0.375.
       The "cells" are between these points.
       The outermost cells are not full cubes bounded by the cube walls.
       The body centers I calculated earlier were for the 3x3x3 internal cells.
       The cell between -0.375 and -0.125 has center -0.25.
       The corner of the cube is -0.5.
       Distance from corner (-0.5, -0.5, -0.5) to body center (-0.25, -0.25, -0.25) is sqrt(3 * 0.25^2) = 0.433.
       This is large.
       
       So:
       1. 4x4x4 Grid (64 pts).
       2. 3x3x3 Body Centers (27 pts).
       3. 8 Cube Corners (8 pts).
       Total = 64 + 27 + 8 = 99 pts.
       We need 1 more.
       Where?
       Maybe the center of the cube?
       Center is (0,0,0).
       Nearest grid point is (0.125, 0.125, 0.125)? No, grid is symmetric.
       Grid points: -0.375, -0.125, 0.125, 0.375.
       Center (0,0,0) is between -0.125 and 0.125.
       Distance to nearest grid point (0.125, 0.125, 0.125) is sqrt(3 * 0.125^2) = 0.2165.
       Distance to nearest body center?
       Body centers are at -0.25, 0, 0.25 in each dim?
       Let's check body center coordinates.
       Cell between -0.375 and -0.125 has center -0.25.
       Cell between -0.125 and 0.125 has center 0.
       Cell between 0.125 and 0.375 has center 0.25.
       So body centers are at combinations of {-0.25, 0, 0.25}.
       The body center at (0,0,0) is one of them!
       So the center of the cube IS a body center.
       So we already have it in the 27 body centers.
       
       So we have 99 points.
       We need 1 more.
       Maybe a face center of the cube?
       Face center (0, 0, 0.5).
       Distance to nearest grid point (0.125, 0.125, 0.375) is sqrt(0.125^2 + 0.125^2 + 0.125^2) = 0.2165.
       Distance to nearest body center (0, 0, 0.25) is 0.25.
       Distance to corner (0.5, 0.5, 0.5) is sqrt(0.5^2 + 0.5^2) = 0.707.
       So (0, 0, 0.5) is a valid spot.
       
       Total 100 points:
       - 64 Grid
       - 27 Body Centers
       - 8 Corners
       - 1 Face Center (0, 0, 0.5)
       
       Let's verify min distance.
       Grid-Grid: 0.25
       Grid-Body: 0.2165
       Grid-Corner: 0.2165
       Grid-Face: 0.2165
       Body-Body: 0.25 (e.g. -0.25 to 0)
       Body-Corner: 0.433
       Body-Face: 0.25
       Corner-Corner: 0.707
       Corner-Face: 0.707
       Face-Grid: 0.2165
       
       Min distance is 0.2165.
       Radius = 0.10825.
       This is a good starting point, better than the 0.1 grid.
       
       Let's implement this specific initialization.
    """
    # 4x4x4 Grid
    ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # 3x3x3 Body Centers
    body_ticks = (np.arange(3) + 0.5) / 4.0 - 0.5  # -0.25, 0.0, 0.25
    bodies = np.array([(x, y, z) for z in body_ticks for y in body_ticks for x in body_ticks])
    
    # 8 Corners
    corners = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    ])
    
    # 1 Face Center
    face_center = np.array([[0.0, 0.0, 0.5]])
    
    pts = np.vstack([grid, bodies, corners, face_center])
    pts = pts[:N]
    
    # Add small noise to break symmetry
    pts = pts + rng.normal(0, 0.001, size=(N, 3))
    pts = np.clip(pts, -0.499, 0.499)
    
    return pts

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin under 120s
    
    rng = np.random.default_rng(42)
    
    # Start with hybrid initialization
    best_pts = _generate_hybrid_init(rng)
    best_r = _compute_radius(best_pts)
    
    # Strategy: Iteratively push the target radius higher.
    current_pts = best_pts.copy()
    
    # Initial phase: Try a few fixed targets to get out of the hybrid local minimum
    # Hybrid radius is ~0.108.
    initial_targets = [0.110, 0.112, 0.114, 0.116, 0.118, 0.120]
    
    for target_r in initial_targets:
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.005, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining_time = DEADLINE - (time.time() - t0) - 5.0
        if remaining_time < 2.0:
            break
            
        opt_pts, success = _optimize_bfgs(
            init_pts, 
            target_r, 
            max_iter=2000, 
            gtol=1e-10,
            time_limit=remaining_time
        )
        
        if np.all(np.isfinite(opt_pts)):
            # Clip to ensure validity
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()
            else:
                # If we didn't improve radius, check if it's a better starting point
                # (e.g. lower energy for the target)
                # For now, stick to best radius config.
                pass

    # Refinement phase: "Squeeze" the radius
    num_squeezes = 20
    for i in range(num_squeezes):
        if time.time() - t0 > DEADLINE - 5.0:
            break
            
        success_improved = False
        increments = [0.002, 0.001, 0.0005, 0.0002]
        
        for inc in increments:
            if time.time() - t0 > DEADLINE - 5.0:
                break
                
            target_r = best_r * (1.0 + inc)
            
            # Add small noise to help escape local minima
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            remaining_time = DEADLINE - (time.time() - t0) - 2.0
            if remaining_time < 1.0:
                break
                
            opt_pts, success = _optimize_bfgs(
                init_pts, 
                target_r, 
                max_iter=2000, 
                gtol=1e-10,
                time_limit=remaining_time
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
                    success_improved = True
                    break
        
        if not success_improved:
            # If we can't squeeze further, try a random restart from best
            # to see if we can find a better basin
            if time.time() - t0 < DEADLINE - 10.0:
                noise = rng.normal(0, 0.01, size=(N, 3))
                init_pts = np.clip(best_pts + noise, -0.49, 0.49)
                remaining_time = DEADLINE - (time.time() - t0) - 5.0
                if remaining_time > 2.0:
                    opt_pts, success = _optimize_bfgs(
                        init_pts, 
                        best_r, 
                        max_iter=1000, 
                        gtol=1e-10,
                        time_limit=remaining_time
                    )
                    if np.all(np.isfinite(opt_pts)):
                        opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                        r_opt = _compute_radius(opt_pts)
                        if r_opt > best_r:
                            best_r = r_opt
                            best_pts = opt_pts.copy()

    return best_pts
