"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import KDTree

N = 200
_TIME_LIMIT = 110  # seconds, leave margin before 120s kill


def _score(points):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    if len(points) < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    return min(min_pair / 2.0, min_boundary)


def _neg_radius(points):
    """Negative of the radius for minimization."""
    if not np.all(np.isfinite(points)):
        return 1.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 1.0
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    d = pdist(points)
    min_pair = np.min(d)
    return -min(min_pair / 2.0, min_boundary)


def _square_grid(n):
    """Generate a square grid packing."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:n]
    return points


def _hex_grid(n):
    """Generate a hexagonal grid packing with best row/col split."""
    best_points = None
    best_score = 0
    
    for rows in range(10, 30):
        cols = n // rows
        if cols < 1:
            continue
        
        s_x = 1.0 / cols
        s_y = 1.0 / ((rows - 1) * np.sqrt(3) / 2)
        s = min(s_x, s_y)
        
        points = []
        for j in range(rows):
            offset = (j % 2) * s / 2
            for i in range(cols):
                x = -0.5 + s/2 + i * s + offset
                y = -0.5 + s * np.sqrt(3) / 4 + j * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points[:n])
        if len(points) == n:
            sc = _score(points)
            if sc > best_score:
                best_score = sc
                best_points = points.copy()
    
    if best_points is None:
        best_points = _square_grid(n)
    return best_points


def _force_directed_refine(points, deadline, iterations=500, lr=0.005):
    """Use force-directed repulsion with KDTree for fast global spreading."""
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(iterations):
        if time.time() > deadline:
            break
        
        # Build KDTree for fast neighbor queries
        tree = KDTree(pts)
        # Find neighbors within a reasonable distance
        neighbors = tree.query_ball_point(pts, r=0.15)
        
        forces = np.zeros_like(pts)
        
        for i in range(len(pts)):
            for j in neighbors[i]:
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random perturbation for coincident points
                    forces[i] += np.random.randn(2) * 0.01
                elif dist < 0.1:
                    # Repulsive force, stronger when closer
                    f_mag = (0.1 - dist) / 0.1
                    forces[i] += f_mag * diff / dist
        
        # Boundary forces: push points away from walls
        for i in range(len(pts)):
            for k in range(2):
                if pts[i, k] < -0.4:
                    forces[i, k] += 0.5 * (-0.4 - pts[i, k])
                elif pts[i, k] > 0.4:
                    forces[i, k] += 0.5 * (0.4 - pts[i, k])
        
        # Update positions
        pts += lr * forces
        
        # Clamp to valid region
        pts = np.clip(pts, -0.5, 0.5)
        
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.995
    
    return best_pts, best_score


def _bottleneck_refine(points, deadline, max_iter=1000):
    """Targeted optimization focusing on the bottleneck constraint (closest pair or wall)."""
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find the bottleneck: closest pair or closest to wall
        d = pdist(pts)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        
        # Convert flat index to pair indices
        n = len(pts)
        # pdist uses compact indexing: index i corresponds to pair (i//2, i%2) approximately
        # More precisely: for n points, pairs are (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # Let's use scipy's approach
        ii = np.where(np.triu(np.ones((n, n), dtype=bool), k=1).ravel())[0]
        ii = np.array(ii)
        # Actually, let's just use the inverse
        # For pdist, the k-th distance corresponds to pair (i, j) where
        # k = i*n - i*(i+1)//2 + j - i - 1
        # Let's just compute all pairs and find the min
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair_dist = np.min(D)
        i_idx, j_idx = np.unravel_index(np.argmin(D), D.shape)
        
        dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary = np.min(dist_boundary)
        wall_idx = np.argmin(dist_boundary)
        
        # Determine which is the bottleneck
        if min_pair_dist / 2.0 < min_boundary:
            # Closest pair is bottleneck
            target = min_pair_dist / 2.0
            # Move the two closest points apart
            diff = pts[i_idx] - pts[j_idx]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random perturbation
                pts[i_idx] += np.random.randn(2) * 0.001
                pts[j_idx] -= np.random.randn(2) * 0.001
            else:
                # Push apart
                push = 0.001 * diff / dist
                pts[i_idx] += push
                pts[j_idx] -= push
        else:
            # Wall is bottleneck
            # Move the point closest to wall inward
            for k in range(2):
                if pts[wall_idx, k] < -0.5:
                    pts[wall_idx, k] += 0.001
                elif pts[wall_idx, k] > 0.5:
                    pts[wall_idx, k] -= 0.001
                else:
                    # Move away from the nearest wall
                    if abs(pts[wall_idx, k] + 0.5) < abs(pts[wall_idx, k] - 0.5):
                        pts[wall_idx, k] += 0.001
                    else:
                        pts[wall_idx, k] -= 0.001
        
        # Also apply small random perturbations to escape local minima
        pts += np.random.randn(N, 2) * 0.0005
        
        # Clamp to valid region
        pts = np.clip(pts, -0.5, 0.5)
        
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
    
    return best_pts, best_score


def _soft_min_optimize(points, deadline, max_iter=200, beta=50):
    """Gradient-based optimization using soft-min approximation."""
    x0 = points.flatten()
    best_x = x0.copy()
    best_f = _neg_radius(x0.reshape(-1, 2))
    
    def soft_min_obj(x):
        pts = x.reshape(-1, 2)
        if not np.all(np.isfinite(pts)):
            return 1.0
        if not np.all((pts >= -0.5) & (pts <= 0.5)):
            return 1.0
        
        # Pairwise distances
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Soft min of pairwise distances
        min_pair = -np.logaddexp.reduce(np.log(D + 1e-10) * (-beta)) / beta
        # Actually, soft min: -1/beta * log(sum(exp(-beta * d)))
        # Let's use a simpler approach: penalize small distances
        small_d = D[D < 0.1]
        if len(small_d) > 0:
            pair_penalty = np.sum(np.exp(-beta * small_d)) / len(small_d)
        else:
            pair_penalty = 0.0
        
        # Boundary distance
        dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary = np.min(dist_boundary)
        
        # Objective: minimize negative radius approximation
        # radius = min(min_pair/2, min_boundary)
        # Use soft approximation
        r_est = min(min_pair / 2.0, min_boundary)
        return -r_est + 0.1 * pair_penalty
    
    def grad(x):
        pts = x.reshape(-1, 2)
        g = np.zeros_like(x)
        eps = 1e-6
        f0 = soft_min_obj(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            g[i] = (soft_min_obj(xp) - f0) / eps
        return g
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        try:
            result = minimize(
                soft_min_obj,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6},
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            break
        if time.time() > deadline - 5:
            break
    
    return best_x.reshape(-1, 2), -best_f


def _optimize(points, deadline, max_iter=300):
    """Combined force-directed + gradient optimization."""
    # Phase 1: Force-directed for global spreading
    fd_pts, fd_score = _force_directed_refine(points, deadline, iterations=200, lr=0.003)
    
    # Phase 2: Gradient-based refinement
    opt_pts, opt_score = _soft_min_optimize(fd_pts, deadline, max_iter=max_iter)
    
    # Phase 3: Bottleneck-targeted refinement
    if time.time() < deadline - 5:
        bn_pts, bn_score = _bottleneck_refine(opt_pts, deadline, max_iter=500)
        if bn_score > opt_score:
            opt_pts, opt_score = bn_pts, bn_score
    
    if opt_score > fd_score:
        return opt_pts, opt_score
    else:
        return fd_pts, fd_score


def _rect_grid_score(params, rows, cols):
    """Score (radius) for a rectangular grid. params = [dx, dy, x0, y0]."""
    dx, dy, x0, y0 = params
    if dx <= 0 or dy <= 0:
        return -1e10
    xs = x0 + np.arange(cols) * dx
    ys = y0 + np.arange(rows) * dy
    pts = np.array([(x, y) for y in ys for x in xs])[:N]
    if len(pts) < N:
        return -1e10
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2)
    if r <= 0:
        return -1e10
    return r


def _hex_grid_score(params, rows, cols):
    """Score for a hexagonal (staggered) grid.
    params = [dx, dy, x0, y0, offset]
    """
    dx, dy, x0, y0, off = params
    if dx <= 0 or dy <= 0:
        return -1e10
    pts = []
    for i in range(rows):
        y = y0 + i * dy
        shift = off * dx if i % 2 == 1 else 0.0
        xs = x0 + shift + np.arange(cols) * dx
        for x in xs:
            pts.append((x, y))
    pts = np.array(pts[:N])
    if len(pts) < N:
        return -1e10
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2)
    if r <= 0:
        return -1e10
    return r


def _neg_rect_score(params, rows, cols):
    return -_rect_grid_score(params, rows, cols)


def _neg_hex_score(params, rows, cols):
    return -_hex_grid_score(params, rows, cols)


def _find_best_grid():
    """Find the best rectangular or hexagonal grid via Nelder-Mead."""
    t0 = time.time()
    best_r = 0.0
    best_pts = _square_grid(N)
    
    # Try rectangular grids with various row/col splits
    factor_pairs = []
    for c in range(1, int(np.sqrt(N)) + 1):
        if N % c == 0:
            r = N // c
            factor_pairs.append((r, c))
            factor_pairs.append((c, r))
    
    extra = [(14, 15), (15, 14), (20, 10), (10, 20), (25, 8), (8, 25),
             (16, 13), (13, 16), (12, 17), (17, 12)]
    for pair in extra:
        if pair not in factor_pairs:
            factor_pairs.append(pair)
    
    factor_pairs = list(set(factor_pairs))
    
    # Rectangular grids
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > 20:
            break
        dx0 = 1.0 / cols
        dy0 = 1.0 / rows
        x00 = -0.5 + dx0 / 2
        y00 = -0.5 + dy0 / 2
        x0 = np.array([dx0, dy0, x00, y00])
        try:
            res = minimize(_neg_rect_score, x0, args=(rows, cols), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            r = -res.fun
            if r > best_r:
                best_r = r
                dx, dy, x0v, y0v = res.x
                xs = x0v + np.arange(cols) * dx
                ys = y0v + np.arange(rows) * dy
                pts = np.array([(x, y) for y in ys for x in xs])[:N]
                if len(pts) == N and \
                   np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
                   np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
                    best_pts = pts
        except Exception:
            pass
    
    # Hexagonal grids
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > 40:
            break
        dx0 = 1.0 / cols
        dy0 = dx0 * np.sqrt(3) / 2
        if (rows - 1) * dy0 > 1.0:
            dy0 = 1.0 / (rows - 1) if rows > 1 else 1.0
        x00 = -0.5 + dx0 / 2
        y00 = -0.5 + dy0 / 2
        off0 = 0.5
        x0 = np.array([dx0, dy0, x00, y00, off0])
        try:
            res = minimize(_neg_hex_score, x0, args=(rows, cols), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            r = -res.fun
            if r > best_r:
                best_r = r
                dx, dy, x0v, y0v, offv = res.x
                pts = []
                for i in range(rows):
                    y = y0v + i * dy
                    shift = offv * dx if i % 2 == 1 else 0.0
                    xs = x0v + shift + np.arange(cols) * dx
                    for x in xs:
                        pts.append((x, y))
                pts = np.array(pts[:N])
                if len(pts) == N and \
                   np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
                   np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
                    best_pts = pts
        except Exception:
            pass
    
    return best_pts, best_r


def construct():
    deadline = time.time() + _TIME_LIMIT
    
    # Phase 1: Find best grid structure
    grid_pts, grid_score = _find_best_grid()
    
    # Also generate standard grids
    sq_pts = _square_grid(N)
    hex_pts = _hex_grid(N)
    
    np.random.seed(42)
    hex_pts2 = hex_pts + np.random.randn(N, 2) * 0.001
    
    candidates = [grid_pts, sq_pts, hex_pts, hex_pts2]
    best_points = None
    best_score = 0
    
    for pts in candidates:
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # Phase 2: Optimize best candidate with force-directed + gradient
    if time.time() < deadline - 30:
        opt_pts, opt_score = _optimize(best_points, deadline, max_iter=200)
        if opt_score > best_score:
            best_points = opt_pts
            best_score = opt_score
    
    # Phase 3: Try optimizing other candidates if time permits
    if time.time() < deadline - 20:
        for pts in candidates:
            if np.array_equal(pts, best_points):
                continue
            sc = _score(pts)
            if sc < best_score - 0.002:
                continue
            if time.time() > deadline - 15:
                break
            opt_pts, opt_score = _optimize(pts, deadline, max_iter=100)
            if opt_score > best_score:
                best_points = opt_pts
                best_score = opt_score
    
    # Phase 4: Final polish with bottleneck refinement
    if time.time() < deadline - 5:
        opt_pts, opt_score = _optimize(best_points, deadline, max_iter=50)
        if opt_score > best_score:
            best_points = opt_pts
            best_score = opt_score
    
    # Phase 5: Additional bottleneck refinement if time permits
    if time.time() < deadline - 2:
        bn_pts, bn_score = _bottleneck_refine(best_points, deadline, max_iter=300)
        if bn_score > best_score:
            best_points = bn_pts
            best_score = bn_score
    
    return best_points
