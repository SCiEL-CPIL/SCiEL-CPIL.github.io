"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _validate(circles, tol=1e-7):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _fix_boundaries(circles):
    """Project circles inside unit square and ensure non-overlap by shrinking if necessary."""
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        r = min(r, 0.5)
        circles[i, 2] = r
        circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)
        
    for _iter in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2.0 + 1e-5
                    circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                    circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                    circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                    circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                    changed = True
        if not changed:
            break
            
    return circles


def _get_contact_edges(xyz, tol=1e-6):
    """
    Get contact edges using Delaunay triangulation.
    Returns a list of (i, j) pairs that are adjacent in the Delaunay triangulation
    AND are in contact (distance <= r_i + r_j + tol).
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    # Delaunay triangulation
    try:
        points = np.column_stack([x, y])
        tri = Delaunay(points)
    except Exception:
        return []
    
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                if a > b:
                    a, b = b, a
                dx = x[a] - x[b]
                dy = y[a] - y[b]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= r[a] + r[b] + tol:
                    edges.add((a, b))
    
    return list(edges)


def _energy_delaunay(xyz, beta, use_delaunay=True):
    """
    Compute energy using Delaunay-based contact graph if possible,
    otherwise fall back to all pairs.
    """
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    # Boundary violations
    lx = np.maximum(0, r - x)
    ux = np.maximum(0, x + r - 1.0)
    ly = np.maximum(0, r - y)
    uy = np.maximum(0, y + r - 1.0)
    boundary_penalty = np.sum(lx**2) + np.sum(ux**2) + np.sum(ly**2) + np.sum(uy**2)
    
    # Negative radius
    neg_r = np.maximum(0, -r)
    neg_penalty = np.sum(neg_r**2)
    
    # Pairwise overlaps
    if use_delaunay:
        edges = _get_contact_edges(xyz, tol=1e-4)
        if len(edges) > 0:
            idx_i = np.array([e[0] for e in edges])
            idx_j = np.array([e[1] for e in edges])
        else:
            idx_i, idx_j = np.triu_indices(N, k=1)
    else:
        idx_i, idx_j = np.triu_indices(N, k=1)
    
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
    overlap_penalty = np.sum(overlap**2)
    
    return -np.sum(r) + beta * (boundary_penalty + neg_penalty + overlap_penalty)


def _polish_analytic(xyz, time_budget=8.0):
    """
    High-precision polish using analytic gradient L-BFGS-B with softplus penalties.
    This is the refinement machinery from the inspiration.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    
    def _softplus(z, eps=1e-4):
        return np.logaddexp(0, z / eps) * eps
    
    def _sigmoid(z, eps=1e-4):
        return 1.0 / (1.0 + np.exp(-z / eps))
    
    def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]

        val = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0

        # Boundary violations
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0

        sp_lx = _softplus(lx, smooth_eps)
        sg_lx = _sigmoid(lx, smooth_eps)
        grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
        grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

        sp_ux = _softplus(ux, smooth_eps)
        sg_ux = _sigmoid(ux, smooth_eps)
        grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
        grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

        sp_ly = _softplus(ly, smooth_eps)
        sg_ly = _sigmoid(ly, smooth_eps)
        grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
        grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

        sp_uy = _softplus(uy, smooth_eps)
        sg_uy = _sigmoid(uy, smooth_eps)
        grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
        grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

        sp_nr = _softplus(-r, smooth_eps)
        sg_nr = _sigmoid(-r, smooth_eps)
        grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

        val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                                 np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                                 np.sum(sp_nr**2))

        # Pairwise overlaps (vectorized)
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        sp_ov = _softplus(overlap, smooth_eps)
        sg_ov = _sigmoid(overlap, smooth_eps)

        factor = 2.0 * penalty_weight * sp_ov * sg_ov
        inv_dist = 1.0 / dist

        np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
        np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
        np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
        np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
        np.add.at(grad[2::3], idx_i, factor)
        np.add.at(grad[2::3], idx_j, factor)

        val += penalty_weight * np.sum(sp_ov**2)

        return val, grad
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v
        
        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14, 
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)


def _physics_relax(xyz, time_limit=5.0, grow_rate=0.0008):
    """
    Force-directed physics relaxation that grows radii over time.
    Uses Delaunay contact graph for efficiency.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    dt = 0.003
    damping = 0.95
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    for iteration in range(3000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(wall_max, 1e-6))
        
        # Compute forces using Delaunay contact graph
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        edges = _get_contact_edges(np.column_stack([x, y, r]), tol=1e-4)
        
        if len(edges) == 0:
            # Fall back to all pairs if no contacts found
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(max(dist_sq, 1e-12))
                    if dist < r[i] + r[j]:
                        overlap = r[i] + r[j] - dist
                        force = overlap / dist
                        fx[i] += force * dx
                        fy[i] += force * dy
                        fx[j] -= force * dx
                        fy[j] -= force * dy
        else:
            for i, j in edges:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                if dist < r[i] + r[j]:
                    overlap = r[i] + r[j] - dist
                    force = overlap / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 10
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 10
        
        # Update velocities and positions
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        x += vx * dt
        y += vy * dt
        
        # Clip to boundaries
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Check if valid and update best
        xyz_test = np.column_stack([x, y, r])
        if _validate(xyz_test, tol=1e-5):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz_test.copy()
    
    return best_xyz


def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])


def _monte_carlo_annealing(xyz0, deadline, beta_start=10.0, beta_end=500.0, 
                           temp_start=0.1, temp_end=0.001, step_scale=0.02):
    """
    Monte Carlo simulated annealing with Delaunay-based energy computation.
    """
    circles = xyz0.copy()
    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    n = N
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    t0 = time.time()
    total_steps = 0
    accepted = 0
    
    n_phases = 5
    for phase in range(n_phases):
        phase_start = phase / n_phases
        phase_end = (phase + 1) / n_phases
        beta = beta_start * (beta_end / beta_start) ** phase_start
        temp = temp_start * (temp_end / temp_start) ** phase_start
        
        phase_deadline = time.time() + (deadline - time.time()) * 0.8 / n_phases
        while time.time() < phase_deadline:
            move_type = np.random.randint(0, 3)
            new_circles = circles.copy()
            
            if move_type == 0:
                i = np.random.randint(n)
                new_circles[i, 0] += np.random.normal(0, step_scale)
                new_circles[i, 1] += np.random.normal(0, step_scale)
            elif move_type == 1:
                i = np.random.randint(n)
                new_circles[i, 2] += np.random.normal(0, step_scale * 0.5)
            else:
                i = np.random.randint(n)
                new_circles[i, 0] += np.random.normal(0, step_scale)
                new_circles[i, 1] += np.random.normal(0, step_scale)
                new_circles[i, 2] += np.random.normal(0, step_scale * 0.5)
            
            new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
            new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
            new_circles[:, 2] = np.clip(new_circles[:, 2], 0, 0.5)
            
            e_old = _energy_delaunay(circles, beta, use_delaunay=True)
            e_new = _energy_delaunay(new_circles, beta, use_delaunay=True)
            dE = e_new - e_old
            
            if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-10)):
                circles = new_circles
                accepted += 1
                if _validate(circles, tol=1e-6):
                    score = np.sum(circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles.copy()
            
            total_steps += 1
            if time.time() > deadline - 5:
                break
        
        step_scale *= 0.8
    
    return best_circles, best_score


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses Monte Carlo simulated annealing with Delaunay contact graph,
    followed by physics relaxation and analytic gradient polishing.
    """
    deadline = _start_time + _TIME_LIMIT
    
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_corner_weighted(seed=999),
    ]
    
    best_overall = None
    best_score_overall = -np.inf
    
    for i, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        # Allocate time: give more time to earlier starts
        alloc = min(remaining - 15, max(15, 25 - i * 3))
        local_deadline = time.time() + alloc
        
        # Phase 1: Monte Carlo annealing
        circles, score = _monte_carlo_annealing(xyz0, local_deadline)
        
        # Phase 2: Physics relaxation
        if time.time() < deadline - 10:
            physics_time = min(5.0, deadline - time.time() - 8)
            circles_phys = _physics_relax(circles, time_limit=physics_time)
            score_phys = np.sum(circles_phys[:, 2])
            if _validate(circles_phys, tol=1e-5) and score_phys > score:
                circles = circles_phys
                score = score_phys
        
        # Phase 3: Analytic gradient polish
        if time.time() < deadline - 5:
            polish_time = min(8.0, deadline - time.time() - 3)
            circles_polished = _polish_analytic(circles, time_budget=polish_time)
            score_polished = np.sum(circles_polished[:, 2])
            if _validate(circles_polished, tol=1e-5) and score_polished > score:
                circles = circles_polished
                score = score_polished
        
        # Validate and fix
        if _validate(circles):
            if score > best_score_overall:
                best_score_overall = score
                best_overall = circles.copy()
        else:
            circles_fixed = _fix_boundaries(circles)
            if _validate(circles_fixed):
                score_fixed = np.sum(circles_fixed[:, 2])
                if score_fixed > best_score_overall:
                    best_score_overall = score_fixed
                    best_overall = circles_fixed.copy()
    
    # Final repair
    if best_overall is not None:
        repaired = _fix_boundaries(best_overall)
        if _validate(repaired):
            best_overall = repaired
        else:
            best_overall = None
    
    if best_overall is None:
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        return result
    
    return best_overall

# EVOLVE-BLOCK-END
