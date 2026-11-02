"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 140  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps


def _validate(circles, tol=1e-7):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Bounds check
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
        
    # Overlap check
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol * (min_dist * 2 + tol):
                return False
    return True


def _fix_boundaries(circles):
    """Project circles inside unit square to handle numerical drift."""
    circles = circles.copy()
    for i in range(N):
        r = max(0, circles[i, 2])
        if r < 0.5:
            circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)
        else:
            circles[i, 0] = 0.5
            circles[i, 1] = 0.5
        circles[i, 2] = r
    return circles


def _get_contact_edges(circles, threshold_factor=1.5):
    """
    Get contact edges from Delaunay triangulation of centers.
    """
    centers = circles[:, :2]
    radii = circles[:, 2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                if a == b: continue
                i, j = (a, b) if a < b else (b, a)
                dist = np.linalg.norm(centers[a] - centers[b])
                min_dist = radii[i] + radii[j]
                if dist < threshold_factor * min_dist:
                    edges.add((i, j))
        return list(edges)
    except Exception:
        return None


def _objective(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Objective function: Minimize -sum(r) + penalty_weight * penalty
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0

    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)

    # Negative radius penalty
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    # Overlap penalties
    n = len(r)
    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap, smooth_eps) ** 2
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty


def _initial_grid():
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
            if len(cells) >= N:
                break
        if len(cells) >= N:
            break
    while len(cells) < N:
        cells.append([0.5, 0.5, 0.05])
    return np.array(cells[:N])

def _initial_hex():
    """Create a hexagonal initial configuration."""
    positions = []
    r = 0.1
    dy = r * np.sqrt(3)
    dx = 2 * r
    y = r
    row = 0
    while y < 1 - r and len(positions) < N:
        x_offset = 0 if row % 2 == 0 else r
        x = r + x_offset
        while x < 1 - r and len(positions) < N:
            positions.append([x, y, r])
            x += dx
        y += dy
        row += 1
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
               [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.09])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        positions.append([x, y, rng.uniform(0.08, 0.11)])
    return np.array(positions[:N])

def _physics_relax(xyz, iterations=500, time_limit=5.0):
    """
    Force-directed relaxation to push overlapping circles apart.
    This is a structural change from pure gradient descent to a physics-based
    relaxation that can escape local minima in the penalty landscape.
    """
    start_time = time.time()
    xyz = xyz.copy()
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Adaptive step size
    step = 0.05
    best_score = -np.inf
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsion forces for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) + 1e-10
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Force proportional to overlap
                    force = (min_dist - dist) * 10.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 20.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 20.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 20.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 20.0
        
        # Update positions
        x += step * fx
        y += step * fy
        
        # Clip to boundaries
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        # Check validity and track best
        temp_xyz = np.column_stack([x, y, r])
        if _validate(temp_xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Decay step size
        step *= 0.995
        
        # If no improvement for a while, try increasing radii slightly
        if it % 50 == 49:
            # Try to expand radii if we have space
            for i in range(N):
                # Calculate max possible radius given current position and neighbors
                max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, dist - r[j])
                if max_r > r[i]:
                    r[i] = np.clip(r[i] * 1.01, r[i], max_r)
                    
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
    
    # Return best valid configuration found during relaxation
    if best_score > -np.inf:
        return np.column_stack([best_x, best_y, best_r])
    else:
        return xyz

def _optimize_single(xyz0, deadline):
    """
    Optimize a single configuration using progressive penalty annealing
    combined with physics-based relaxation steps.
    """
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    best_valid = False
    best_valid_score = -np.inf

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Define annealing schedule
    phases = [
        (1.0, 1e-2, 50),   # Phase 1: Low penalty, smooth, layout focus
        (5.0, 1e-3, 50),   # Phase 2: Medium penalty
        (20.0, 1e-4, 50),  # Phase 3: High penalty, tighter
        (100.0, 1e-5, 50), # Phase 4: Very high penalty, strict
        (500.0, 1e-6, 50), # Phase 5: Extreme penalty, final polish
    ]

    for p_weight, s_eps, max_iter in phases:
        if time.time() > deadline - 5.0:
            break
            
        # Update contact edges for this phase if penalty is high
        contact_edges = None
        if p_weight >= 20.0:
            circles = params.reshape(N, 3)
            contact_edges = _get_contact_edges(circles)

        try:
            res = minimize(
                _objective, 
                params, 
                method='L-BFGS-B', 
                bounds=bounds, 
                args=(p_weight, s_eps, contact_edges),
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
            )
            params = res.x
            current_score = np.sum(params[2::3])
            
            if current_score > best_score:
                best_score = current_score
                best_params = params.copy()
                
            # Check validity
            circles = params.reshape(N, 3)
            if _validate(circles):
                if current_score > best_valid_score:
                    best_valid_score = current_score
                    best_valid = True
        except Exception:
            pass

        # After each gradient step, apply physics relaxation to escape local minima
        if time.time() < deadline - 10.0:
            circles = params.reshape(N, 3)
            relaxed = _physics_relax(circles, iterations=100, time_limit=2.0)
            if _validate(relaxed):
                relaxed_score = np.sum(relaxed[:, 2])
                if relaxed_score > best_score:
                    best_score = relaxed_score
                    best_params = relaxed.flatten().copy()
                if relaxed_score > best_valid_score:
                    best_valid_score = relaxed_score
                    best_valid = True
                # Continue optimization from the relaxed position
                params = relaxed.flatten().copy()

    # Final check and fix
    best_circles = best_params.reshape(N, 3)
    if not _validate(best_circles):
        best_circles = _fix_boundaries(best_circles)
        if not _validate(best_circles):
            for _ in range(20):
                best_circles[:, 2] *= 0.98
                if _validate(best_circles):
                    break
    
    if _validate(best_circles):
        return best_circles
    else:
        return best_circles

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global _start_time
    _start_time = time.time()
    deadline = _TIME_LIMIT

    best_xyz = None
    best_score = -np.inf

    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(42),
        _initial_corner_weighted(123),
        _initial_corner_weighted(789)
    ]

    for i, xyz0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        time_budget = min(20.0, deadline - time.time() - 5.0)
        if time_budget < 5.0:
            break
            
        xyz_opt = _optimize_single(xyz0, deadline)
        
        if _validate(xyz_opt):
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
        else:
            xyz_fixed = _fix_boundaries(xyz_opt)
            if _validate(xyz_fixed):
                score = np.sum(xyz_fixed[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_fixed

    # If no valid solution found, fallback to grid
    if best_xyz is None:
        best_xyz = _initial_grid()

    return best_xyz

# EVOLVE-BLOCK-END
