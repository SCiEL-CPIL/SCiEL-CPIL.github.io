"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _md_initialize(seed=0):
    """
    Initialize circles in a jittered grid to provide a good starting configuration
    for the molecular dynamics simulation.
    """
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    # Grid layout
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Add small jitter
                x += rng.uniform(-0.01, 0.01)
                y += rng.uniform(-0.01, 0.01)
                positions.append([x, y])
    
    for i in range(N):
        xy[i, 0] = positions[i][0]
        xy[i, 1] = positions[i][1]
        r[i] = 0.4 / side # Start with a reasonable radius

    return xy, r


def _md_simulate(xy, r, deadline, seed=0):
    """
    Physics-based molecular dynamics simulation.
    Uses velocity Verlet integration with repulsive forces.
    """
    rng = np.random.RandomState(seed)
    n = N
    xy = xy.copy()
    r = r.copy()
    v = np.zeros((n, 2))
    
    # Simulation parameters
    dt = 0.01
    repulsion_k = 100.0
    damping = 0.95
    
    steps = 5000
    for step in range(steps):
        if time.time() > deadline:
            break
        
        forces = np.zeros((n, 2))
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-6:
                    dist = 1e-6
                    # Randomize direction if too close
                    dx = rng.uniform(-0.1, 0.1)
                    dy = rng.uniform(-0.1, 0.1)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Spring force
                    overlap = min_dist - dist
                    f = repulsion_k * overlap
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            if xy[i, 0] < r[i]:
                overlap = r[i] - xy[i, 0]
                forces[i, 0] += repulsion_k * overlap
            # Right wall
            if xy[i, 0] > 1.0 - r[i]:
                overlap = xy[i, 0] - (1.0 - r[i])
                forces[i, 0] -= repulsion_k * overlap
            # Bottom wall
            if xy[i, 1] < r[i]:
                overlap = r[i] - xy[i, 1]
                forces[i, 1] += repulsion_k * overlap
            # Top wall
            if xy[i, 1] > 1.0 - r[i]:
                overlap = xy[i, 1] - (1.0 - r[i])
                forces[i, 1] -= repulsion_k * overlap
            
            # Try to grow radius slightly if space allows? 
            # For now, just keep radii fixed during MD to stabilize positions,
            # then grow them in a separate phase.
            
        # Update velocity and position (Velocity Verlet simplified)
        v += forces * dt
        v *= damping
        xy += v * dt
        
        # Hard clip to ensure they don't fly out of bounds during simulation
        for i in range(n):
            xy[i, 0] = np.clip(xy[i, 0], 0, 1)
            xy[i, 1] = np.clip(xy[i, 1], 0, 1)

    return xy, r


def _grow_and_resolve(xy, r, deadline, max_iter=100):
    """
    Iteratively grow radii and resolve collisions by pushing circles apart.
    This is a "rigid body" expansion phase.
    """
    n = N
    xy = xy.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # 1. Try to grow all radii
        # Find the maximum possible growth for each circle
        # limited by walls and current neighbors
        growths = np.zeros(n)
        
        for i in range(n):
            # Wall limits
            wall_limit = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1]) - r[i]
            growths[i] = wall_limit
            
            # Neighbor limits
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # If dist < r[i] + r[j], they are overlapping.
                # We want to increase r[i] such that new_r[i] + r[j] <= dist
                # new_r[i] <= dist - r[j]
                # growth = new_r[i] - r[i] <= dist - r[j] - r[i]
                # So growth is limited by (dist - r[i] - r[j])
                # But if they are already overlapping, this is negative, so no growth.
                # We only grow if there is space.
                limit = dist - r[j] - r[i]
                if limit < growths[i]:
                    growths[i] = limit
        
        # Find the minimum positive growth (bottleneck)
        # We want to grow all circles as much as possible, but we are limited by the tightest constraint.
        # Actually, we can grow each circle independently up to its limit.
        # Let's apply a small fraction of the available growth to all to keep them balanced?
        # Or just apply the full growth?
        # Applying full growth might cause new overlaps if we don't re-solve immediately.
        
        # Let's apply a small step of growth to all
        step = 0.005
        for i in range(n):
            if growths[i] > 0:
                r[i] += min(step, growths[i])
            else:
                # If no growth, maybe shrink slightly to resolve overlap?
                # Or just leave it. The resolution step will handle it.
                pass
        
        # 2. Resolve collisions (push apart)
        for _ in range(20): # Inner iterations for resolution
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 1e-6:
                        dx = 1e-6
                        dy = 0
                        dist = 1e-6
                    min_dist = r[i] + r[j]
                    if dist < min_dist - 1e-9:
                        overlap = min_dist - dist
                        # Push apart
                        # Weight by inverse mass? Assume equal mass.
                        push = overlap / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Check if pushing i left hits wall
                        new_xi = xy[i, 0] - push * nx
                        new_yi = xy[i, 1] - push * ny
                        new_xj = xy[j, 0] + push * nx
                        new_yj = xy[j, 1] + push * ny
                        
                        # Clamp to walls
                        # If i hits left wall, we can't push it further left
                        if new_xi < r[i]:
                            # Push j more
                            extra = r[i] - new_xi
                            new_xj += extra * nx
                            new_yj += extra * ny
                            new_xi = r[i]
                        if new_xi > 1.0 - r[i]:
                            extra = new_xi - (1.0 - r[i])
                            new_xj -= extra * nx
                            new_yj -= extra * ny
                            new_xi = 1.0 - r[i]
                        
                        if new_yi < r[i]:
                            extra = r[i] - new_yi
                            new_xj += extra * ny
                            new_yj += extra * ny
                            new_yi = r[i]
                        if new_yi > 1.0 - r[i]:
                            extra = new_yi - (1.0 - r[i])
                            new_xj -= extra * ny
                            new_yj -= extra * ny
                            new_yi = 1.0 - r[i]
                        
                        # Similarly for j
                        if new_xj < r[j]:
                            extra = r[j] - new_xj
                            new_xi -= extra * nx
                            new_yi -= extra * ny
                            new_xj = r[j]
                        if new_xj > 1.0 - r[j]:
                            extra = new_xj - (1.0 - r[j])
                            new_xi += extra * nx
                            new_yi += extra * ny
                            new_xj = 1.0 - r[j]
                        
                        if new_yj < r[j]:
                            extra = r[j] - new_yj
                            new_xi -= extra * ny
                            new_yi -= extra * ny
                            new_yj = r[j]
                        if new_yj > 1.0 - r[j]:
                            extra = new_yj - (1.0 - r[j])
                            new_xi += extra * ny
                            new_yi += extra * ny
                            new_yj = 1.0 - r[j]
                        
                        xy[i, 0] = new_xi
                        xy[i, 1] = new_yi
                        xy[j, 0] = new_xj
                        xy[j, 1] = new_yj
                        moved = True
            
            if not moved:
                break
        
        # Final check for validity
        total, valid = _eval_packing(xy, r)
        if not valid:
             # If invalid, shrink radii slightly to recover
             r *= 0.99
             
    return xy, r


def _smooth_objective(v, n):
    """
    Smooth objective for final polish.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    beta = 50.0
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.logaddexp(0.0, beta * viol) / beta

    return -np.sum(r) + 100.0 * (wall_pen + total_viol)


def _smooth_grad(v, n):
    """
    Gradient of the smooth objective.
    """
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)
    beta = 50.0

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-10:
            continue
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        if min_idx == 0:
            grad[2*n + i] += 100.0 * s
            grad[i] -= 100.0 * s
        elif min_idx == 1:
            grad[2*n + i] += 100.0 * s
            grad[i] += 100.0 * s
        elif min_idx == 2:
            grad[2*n + i] += 100.0 * s
            grad[n + i] -= 100.0 * s
        else:
            grad[2*n + i] += 100.0 * s
            grad[n + i] += 100.0 * s

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-10:
                continue
            factor = 100.0 * s
            grad[i] += factor * dx / dist
            grad[n + i] += factor * dy / dist
            grad[j] -= factor * dx / dist
            grad[n + j] -= factor * dy / dist
            grad[2*n + i] += factor
            grad[2*n + j] += factor

    grad[2*n:3*n] -= 1.0
    return grad


def _gradient_polish(xy, r, deadline):
    """
    Final polish using L-BFGS-B.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    bounds = [(0.001, 0.999)] * n + [(0.001, 0.999)] * n + [(0.001, 0.5)] * n

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 5000})
        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r
    except Exception:
        pass
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Molecular Dynamics + Grow/Resolve
    # Try a few different seeds for MD initialization
    for seed in range(5):
        if time.time() > deadline:
            break
        
        xy, r = _md_initialize(seed)
        
        # Run MD to relax positions
        xy, r = _md_simulate(xy, r, deadline, seed)
        
        # Grow and resolve
        xy, r = _grow_and_resolve(xy, r, deadline)
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

    # Phase 2: Gradient Polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 20:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            xy[i, 0] = (gx + 0.5) / side
            xy[i, 1] = (gy + 0.5) / side
            r[i] = 0.4 / side
        best_xy = xy
        best_r = r
        best_total, best_valid = _eval_packing(best_xy, best_r)

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
