"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _flower_init():
    """
    Create a flower-like pattern: one large central circle surrounded by smaller ones.
    This provides a good structured starting point.
    """
    positions = []
    # Central circle
    positions.append([0.5, 0.5, 0.15])
    
    # Ring of circles around center
    n_ring = 8
    ring_r = 0.18
    ring_radius = 0.22
    for k in range(n_ring):
        angle = 2 * np.pi * k / n_ring
        x = 0.5 + ring_radius * np.cos(angle)
        y = 0.5 + ring_radius * np.sin(angle)
        r = 0.08
        positions.append([x, y, r])
    
    # Second ring
    n_ring2 = 9
    ring2_radius = 0.35
    for k in range(n_ring2):
        angle = 2 * np.pi * k / n_ring2 + np.pi / n_ring2
        x = 0.5 + ring2_radius * np.cos(angle)
        y = 0.5 + ring2_radius * np.sin(angle)
        r = 0.05
        positions.append([x, y, r])
    
    # Corner circles
    corners = [(0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)]
    for cx, cy in corners:
        positions.append([cx, cy, 0.04])
    
    # Fill remaining with small circles
    while len(positions) < N:
        rng = np.random.RandomState(len(positions))
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = 0.02
        positions.append([x, y, r])
    
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _smooth_objective(params, beta=50.0):
    """
    Smooth objective: sum of radii - penalty for violations.
    Uses soft-min via log-sum-exp for smooth penalty.
    params: (2n + n) = 3n vector: [x0,y0,x1,y1,...,x_{n-1},y_{n-1}, r0,r1,...,r_{n-1}]
    """
    n = N
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    
    # Wall violations
    wall_viol = np.maximum(0, -xy[:, 0] + r) + np.maximum(0, xy[:, 0] - (1 - r)) + \
                np.maximum(0, -xy[:, 1] + r) + np.maximum(0, xy[:, 1] - (1 - r))
    
    # Pairwise violations (soft)
    pair_viol = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            violation = r[i] + r[j] - dist
            if violation > 0:
                pair_viol[i] += violation
                pair_viol[j] += violation
    
    total_viol = np.sum(wall_viol) + np.sum(pair_viol)
    
    # Smooth penalty using soft-max
    penalty = np.log(1 + np.exp(beta * total_viol)) / beta
    
    return -np.sum(r) + penalty


def _smooth_grad(params, beta=50.0):
    """Analytic gradient of the smooth objective."""
    n = N
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    
    grad = np.zeros_like(params)
    
    # Wall contributions
    wall_viol = np.zeros(4 * n)
    wall_viol[0:n] = np.maximum(0, -xy[:, 0] + r)
    wall_viol[n:2*n] = np.maximum(0, xy[:, 0] - (1 - r))
    wall_viol[2*n:3*n] = np.maximum(0, -xy[:, 1] + r)
    wall_viol[3*n:4*n] = np.maximum(0, xy[:, 1] - (1 - r))
    
    total_viol = np.sum(wall_viol)
    
    # Pairwise violations
    pair_viol = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-12:
                violation = r[i] + r[j] - dist
                if violation > 0:
                    pair_viol[i] += violation
                    pair_viol[j] += violation
    
    total_viol += np.sum(pair_viol)
    
    # Derivative of log(1 + exp(beta * total_viol)) / beta w.r.t. total_viol
    # = exp(beta * total_viol) / (1 + exp(beta * total_viol)) = sigmoid(beta * total_viol)
    if total_viol > 500 / beta:
        sig = 1.0
    else:
        sig = 1.0 / (1.0 + np.exp(-beta * total_viol))
    
    # Gradient w.r.t. xy
    for i in range(n):
        # Wall gradients
        if -xy[i, 0] + r[i] > 0:
            grad[2*i, 0] += sig  # d/dx of (-x + r)
        if xy[i, 0] - (1 - r[i]) > 0:
            grad[2*i, 0] -= sig  # d/dx of (x - 1 + r)
        if -xy[i, 1] + r[i] > 0:
            grad[2*i, 1] += sig
        if xy[i, 1] - (1 - r[i]) > 0:
            grad[2*i, 1] -= sig
        
        # Pairwise gradients
        for j in range(n):
            if i == j:
                continue
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-12:
                violation = r[i] + r[j] - dist
                if violation > 0:
                    # d(violation)/d(x_i) = -dx/dist
                    grad[2*i, 0] -= sig * dx / dist
                    grad[2*i, 1] -= sig * dy / dist
    
    # Gradient w.r.t. r
    for i in range(n):
        # Wall gradients
        if -xy[i, 0] + r[i] > 0:
            grad[2*n + i] += sig
        if xy[i, 0] - (1 - r[i]) > 0:
            grad[2*n + i] += sig
        if -xy[i, 1] + r[i] > 0:
            grad[2*n + i] += sig
        if xy[i, 1] - (1 - r[i]) > 0:
            grad[2*n + i] += sig
        
        # Pairwise gradients
        for j in range(n):
            if i == j:
                continue
            violation = r[i] + r[j] - np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2)
            if violation > 0:
                grad[2*n + i] += sig
        
        # Objective: -sum(r)
        grad[2*n + i] -= 1.0
    
    return grad


def _gradient_polish(xy, r, deadline, beta=50.0):
    """
    Use L-BFGS-B with analytic gradient to polish a packing.
    """
    n = len(xy)
    params = np.concatenate([xy.ravel(), r])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(n):
        bounds.append((0.001, 0.49))  # r
    
    result = minimize(
        _smooth_objective, params,
        jac=_smooth_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
    )
    
    new_params = result.x
    new_xy = new_params[:2*n].reshape(n, 2)
    new_r = new_params[2*n:]
    
    # Ensure validity
    for i in range(n):
        ri = new_r[i]
        new_xy[i, 0] = np.clip(new_xy[i, 0], ri, 1.0 - ri)
        new_xy[i, 1] = np.clip(new_xy[i, 1], ri, 1.0 - ri)
        new_r[i] = min(new_r[i], new_xy[i, 0], 1.0 - new_xy[i, 0], new_xy[i, 1], 1.0 - new_xy[i, 1])
        new_r[i] = max(new_r[i], 0.001)
    
    total, valid = _eval_packing(new_xy, new_r)
    if not valid:
        total = 0.0
    
    return new_xy, new_r, total, valid


def _high_precision_polish(xy, r, deadline):
    """
    Final high-precision polish with tighter tolerances and more iterations.
    """
    n = len(xy)
    params = np.concatenate([xy.ravel(), r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.001, 0.49))
    
    result = minimize(
        _smooth_objective, params,
        jac=_smooth_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 2000}
    )
    
    new_params = result.x
    new_xy = new_params[:2*n].reshape(n, 2)
    new_r = new_params[2*n:]
    
    for i in range(n):
        ri = new_r[i]
        new_xy[i, 0] = np.clip(new_xy[i, 0], ri, 1.0 - ri)
        new_xy[i, 1] = np.clip(new_xy[i, 1], ri, 1.0 - ri)
        new_r[i] = min(new_r[i], new_xy[i, 0], 1.0 - new_xy[i, 0], new_xy[i, 1], 1.0 - new_xy[i, 1])
        new_r[i] = max(new_r[i], 0.001)
    
    total, valid = _eval_packing(new_xy, new_r)
    if not valid:
        total = 0.0
    
    return new_xy, new_r, total, valid


def _bubble_relaxation(xy, r, deadline, n_steps=500, dt=0.001):
    """
    Physics-based bubble relaxation.
    Model circles as pressurized bubbles with internal pressure proportional to r^2.
    They expand against repulsive contact forces and boundary constraints.
    """
    n = len(xy)
    cur_xy = xy.copy()
    cur_r = r.copy()
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(cur_xy, cur_r)
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0
    
    # Precompute pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        forces_xy = np.zeros_like(cur_xy)
        forces_r = np.zeros(n)
        
        # Repulsive forces between circles
        for i, j in pairs:
            dx = cur_xy[i, 0] - cur_xy[j, 0]
            dy = cur_xy[i, 1] - cur_xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                continue
            overlap = cur_r[i] + cur_r[j] - dist
            if overlap > 0:
                # Repulsive force proportional to overlap
                fx = overlap * dx / dist
                fy = overlap * dy / dist
                forces_xy[i, 0] += fx
                forces_xy[i, 1] += fy
                forces_xy[j, 0] -= fx
                forces_xy[j, 1] -= fy
                # Pressure effect: larger circles push more
                forces_r[i] += overlap * cur_r[i]
                forces_r[j] += overlap * cur_r[j]
        
        # Boundary forces
        for i in range(n):
            # Left wall
            overlap_l = cur_r[i] - cur_xy[i, 0]
            if overlap_l > 0:
                forces_xy[i, 0] += overlap_l
                forces_r[i] += overlap_l
            # Right wall
            overlap_r = cur_r[i] - (1.0 - cur_xy[i, 0])
            if overlap_r > 0:
                forces_xy[i, 0] -= overlap_r
                forces_r[i] += overlap_r
            # Bottom wall
            overlap_b = cur_r[i] - cur_xy[i, 1]
            if overlap_b > 0:
                forces_xy[i, 1] += overlap_b
                forces_r[i] += overlap_b
            # Top wall
            overlap_t = cur_r[i] - (1.0 - cur_xy[i, 1])
            if overlap_t > 0:
                forces_xy[i, 1] -= overlap_t
                forces_r[i] += overlap_t
        
        # Internal pressure: bubbles want to expand
        # Pressure proportional to r^2, so force = pressure * area ~ r^2 * r = r^3
        # But we want to maximize sum of radii, so add a small expansion force
        for i in range(n):
            forces_r[i] += 0.01 * cur_r[i] ** 2  # Expansion force
        
        # Update positions and radii
        for i in range(n):
            cur_xy[i, 0] += dt * forces_xy[i, 0]
            cur_xy[i, 1] += dt * forces_xy[i, 1]
            cur_r[i] += dt * forces_r[i]
            cur_r[i] = max(0.001, min(0.49, cur_r[i]))
            cur_xy[i, 0] = np.clip(cur_xy[i, 0], cur_r[i], 1.0 - cur_r[i])
            cur_xy[i, 1] = np.clip(cur_xy[i, 1], cur_r[i], 1.0 - cur_r[i])
        
        # Evaluate
        total, valid = _eval_packing(cur_xy, cur_r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = cur_xy.copy()
            best_r = cur_r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _sa_packing(xy, r, deadline, seed=42):
    """
    Simulated annealing for circle packing.
    """
    n = len(xy)
    rng = np.random.RandomState(seed)
    
    current_xy = xy.copy()
    current_r = r.copy()
    current_total, current_valid = _eval_packing(current_xy, current_r)
    if not current_valid:
        current_total = 0.0
    
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_total = current_total
    best_valid = current_valid
    
    T0 = 0.05
    T_end = 0.001
    cooling = 0.995
    
    T = T0
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline and T > T_end and no_improve_count < max_no_improve:
        move_type = rng.random()
        
        if move_type < 0.4:
            i = rng.randint(n)
            dx = rng.normal(0, T * 2)
            dy = rng.normal(0, T * 2)
            new_xy = current_xy.copy()
            new_xy[i, 0] += dx
            new_xy[i, 1] += dy
            new_r = current_r.copy()
        elif move_type < 0.7:
            i = rng.randint(n)
            dr = rng.normal(0, T * 0.5)
            new_xy = current_xy.copy()
            new_r = current_r.copy()
            new_r[i] = max(0.001, min(0.49, current_r[i] + dr))
        elif move_type < 0.9:
            i, j = rng.choice(n, size=2, replace=False)
            dx_i = rng.normal(0, T * 2)
            dy_i = rng.normal(0, T * 2)
            dx_j = rng.normal(0, T * 2)
            dy_j = rng.normal(0, T * 2)
            new_xy = current_xy.copy()
            new_xy[i, 0] += dx_i
            new_xy[i, 1] += dy_i
            new_xy[j, 0] += dx_j
            new_xy[j, 1] += dy_j
            new_r = current_r.copy()
        else:
            i, j = rng.choice(n, size=2, replace=False)
            new_xy = current_xy.copy()
            new_xy[[i, j]] = current_xy[[j, i]]
            new_r = current_r.copy()
        
        for k in range(n):
            new_xy[k, 0] = np.clip(new_xy[k, 0], new_r[k], 1.0 - new_r[k])
            new_xy[k, 1] = np.clip(new_xy[k, 1], new_r[k], 1.0 - new_r[k])
        
        new_total, new_valid = _eval_packing(new_xy, new_r)
        if not new_valid:
            new_total = -10.0
        
        delta = new_total - current_total
        
        if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_xy = new_xy
            current_r = new_r
            current_total = new_total
            current_valid = new_valid
            
            if current_total > best_total:
                best_total = current_total
                best_valid = current_valid
                best_xy = current_xy.copy()
                best_r = current_r.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        T *= cooling
    
    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Multiple diverse initializations + local search
    inits = []
    
    # Grid inits
    for side in [5, 6, 7]:
        pos = _grid_init(side)
        inits.append((pos[:, :2].copy(), pos[:, 2].copy()))
    
    # Flower init
    pos = _flower_init()
    inits.append((pos[:, :2].copy(), pos[:, 2].copy()))
    
    # Random inits
    for seed in range(6):
        xy, r = _random_init(N, seed=seed)
        inits.append((xy, r))
    
    for xy, r in inits:
        if time.time() > deadline - 20:
            break
        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 20, grid_size=70)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Bubble relaxation from best
    if best_valid and best_xy is not None:
        if time.time() < deadline - 15:
            bub_xy, bub_r, bub_total, bub_valid = _bubble_relaxation(
                best_xy.copy(), best_r.copy(), deadline - 15, n_steps=300, dt=0.002)
            if bub_valid and bub_total > best_total:
                best_total = bub_total
                best_valid = True
                best_xy = bub_xy
                best_r = bub_r

    # Phase 3: Gradient polish from best
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            gp_xy, gp_r, gp_total, gp_valid = _gradient_polish(
                best_xy.copy(), best_r.copy(), deadline - 10, beta=50.0)
            if gp_valid and gp_total > best_total:
                best_total = gp_total
                best_valid = True
                best_xy = gp_xy
                best_r = gp_r

    # Phase 4: SA from multiple restarts
    if best_valid and best_xy is not None:
        if time.time() < deadline - 15:
            n_restarts = 4
            per_restart_time = (deadline - 15 - time.time()) / n_restarts
            for restart in range(n_restarts):
                if time.time() > deadline - 15:
                    break
                rng = np.random.RandomState(5000 + restart)
                perturbed_xy = best_xy.copy()
                perturbed_r = best_r.copy()
                for i in range(N):
                    perturbed_xy[i, 0] += rng.normal(0, 0.03)
                    perturbed_xy[i, 1] += rng.normal(0, 0.03)
                    perturbed_r[i] = max(0.001, perturbed_r[i] * (1 + rng.normal(0, 0.15)))
                for i in range(N):
                    perturbed_xy[i, 0] = np.clip(perturbed_xy[i, 0], perturbed_r[i], 1.0 - perturbed_r[i])
                    perturbed_xy[i, 1] = np.clip(perturbed_xy[i, 1], perturbed_r[i], 1.0 - perturbed_r[i])
                
                sub_deadline = min(deadline - 15, time.time() + per_restart_time * 0.85)
                sa_xy, sa_r, sa_total, sa_valid = _sa_packing(
                    perturbed_xy, perturbed_r, sub_deadline, seed=6000 + restart)
                
                if sa_valid and sa_total > best_total:
                    best_total = sa_total
                    best_valid = True
                    best_xy = sa_xy
                    best_r = sa_r

    # Phase 5: Final gradient polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 5:
            gp2_xy, gp2_r, gp2_total, gp2_valid = _gradient_polish(
                best_xy.copy(), best_r.copy(), deadline - 5, beta=80.0)
            if gp2_valid and gp2_total > best_total:
                best_total = gp2_total
                best_valid = True
                best_xy = gp2_xy
                best_r = gp2_r

    # Phase 6: High precision polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 2:
            hp_xy, hp_r, hp_total, hp_valid = _high_precision_polish(
                best_xy.copy(), best_r.copy(), deadline - 2)
            if hp_valid and hp_total > best_total:
                best_total = hp_total
                best_valid = True
                best_xy = hp_xy
                best_r = hp_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

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

    # Final check
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
