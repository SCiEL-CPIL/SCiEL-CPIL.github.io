"""Improved packing: 7x7 grid init + Nelder-Mead local optimization with time guard."""
import numpy as np
import time

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _radius(centers):
    """Compute the common radius for a set of centers inside [-0.5, 0.5]^2."""
    n = len(centers)
    # Boundary constraint
    dist_to_border = 0.5 - np.max(np.abs(centers), axis=1)
    r_border = np.min(dist_to_border)
    # Pairwise distances
    if n > 1:
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
    else:
        r_pair = r_border
    return min(r_border, r_pair)


def _negative_radius(centers):
    """Objective to minimize: negative radius."""
    return -_radius(centers)


def _init_7x7_plus():
    """
    Initialize with a 7x7 grid (49 points) plus one extra point.
    7x7 grid in [-0.5, 0.5]^2 gives spacing 1/7, so radius ~ 1/14 ≈ 0.07143.
    The 50th point is placed optimally among the gaps.
    """
    m = 7
    # 7x7 grid points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid.copy()  # 49 points

    # Place the 50th point. Try several candidate positions and pick the best.
    # The grid has spacing 1/7 ≈ 0.14286. Gaps between grid points are at midpoints.
    # Best place for an extra point is likely near a corner or edge gap.
    # Let's try placing it at various positions and compute the radius.
    
    best_pos = None
    best_r = -1
    
    # Candidate positions: midpoints of edges, corners of cells, etc.
    # The grid cells have corners at ticks. Midpoints between adjacent grid points:
    # For a 7x7 grid, there are 6x6 internal cells. Midpoints of these cells.
    # Also try positions near the boundary.
    
    candidates = []
    # Cell centers (midpoints of adjacent grid points)
    for i in range(m - 1):
        for j in range(m - 1):
            cx = (ticks[i] + ticks[i + 1]) / 2
            cy = (ticks[j] + ticks[j + 1]) / 2
            candidates.append([cx, cy])
    
    # Also try some positions near corners and edges
    # For example, slightly offset from grid points
    for i in range(m):
        for j in range(m):
            # Small offsets
            for dx in [-0.02, 0.02, -0.03, 0.03]:
                for dy in [-0.02, 0.02, -0.03, 0.03]:
                    x = ticks[i] + dx
                    y = ticks[j] + dy
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        candidates.append([x, y])
    
    # Try each candidate
    for pos in candidates:
        test_centers = np.vstack([centers, np.array(pos)])
        r = _radius(test_centers)
        if r > best_r:
            best_r = r
            best_pos = pos
    
    if best_pos is not None:
        centers = np.vstack([centers, np.array(best_pos)])
    
    return centers


def _nelder_mead(centers, max_iter=5000, tol=1e-10, start_time=None):
    """
    Nelder-Mead optimization to maximize the common radius.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(centers)
    dim = 2 * n
    
    # Flatten centers to 1D
    x0 = centers.flatten().copy()
    
    # Create initial simplex
    simplex = [x0.copy()]
    for i in range(dim):
        x = x0.copy()
        if x0[i] != 0:
            x[i] *= 1.05
        else:
            x[i] += 0.005
        simplex.append(x)
    
    # Evaluate initial simplex
    fvals = [_negative_radius(np.array(simplex[i].reshape(n, 2))) for i in range(dim + 1)]
    
    # Sort by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    best_x = simplex[0].copy()
    best_f = fvals[0]
    
    for iteration in range(max_iter):
        if time.time() - start_time > DEADLINE:
            break
        
        # Reflection
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = _negative_radius(np.array(xr.reshape(n, 2)))
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = _negative_radius(np.array(xe.reshape(n, 2)))
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (centroid - simplex[-1])
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
            fc = _negative_radius(np.array(xc.reshape(n, 2)))
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = _negative_radius(np.array(simplex[i].reshape(n, 2)))
        
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Update best
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0].copy()
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
    
    return np.array(best_x.reshape(n, 2)), -best_f


def construct():
    start_time = time.time()
    
    # Initialize with 7x7 grid + 1 extra point
    centers = _init_7x7_plus()
    current_r = _radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Run Nelder-Mead optimization
    if time.time() - start_time < DEADLINE:
        optimized_centers, opt_r = _nelder_mead(centers, max_iter=10000, start_time=start_time)
        if opt_r > best_r:
            best_centers = optimized_centers
            best_r = opt_r
    
    # Final polish: run a few more iterations with tighter tolerance if time permits
    if time.time() - start_time < DEADLINE - 5:
        polished_centers, polish_r = _nelder_mead(best_centers, max_iter=5000, tol=1e-12, start_time=start_time)
        if polish_r > best_r:
            best_centers = polished_centers
            best_r = polish_r
    
    # Ensure all centers are within the square
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
