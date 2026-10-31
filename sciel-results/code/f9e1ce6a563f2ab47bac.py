"""Circle packing in unit square: maximize sum of radii for N=26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import linprog

N = 26
T_LIMIT = 150.0          # hard wall-clock limit (s)
SAFETY = 8.0             # return this many seconds before the limit
DEADLINE = time.time() + (T_LIMIT - SAFETY)


def _softmin(a, t):
    # a: (n,); returns (softmin, dsoftmin/da)
    a = np.asarray(a)
    m = a.min()
    z = -t * (a - m)
    ez = np.exp(z - z.max())
    logsum = z.max() + np.log(ez.sum())
    val = m - (1.0 / t) * logsum
    grad = -(ez / ez.sum())
    return val, grad


def _clearances(P, r):
    # pairwise clearances d_ij - (r_i+r_j), and wall clearances r_i - d_i
    n = len(r)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d - (r[None, :] + r[:, None])
    # wall distances
    dx = np.minimum(P[:, 0], 1.0 - P[:, 0])
    dy = np.minimum(P[:, 1], 1.0 - P[:, 1])
    wall = np.minimum(dx, dy)
    wallc = r - wall
    return pair, wallc


def _valid(P, r, tol=1e-9):
    # returns True if all circles are inside [0,1]^2 with disjoint interiors (with tol)
    n = len(r)
    if np.any(P < -tol) or np.any(P > 1.0 + tol) or np.any(r < -tol):
        return False
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if np.any(d < (r[None, :] + r[:, None]) - tol):
        return False
    return True


def _score(P, r):
    return float(np.sum(r))


def _repair(P, r, iters=100):
    """
    Robust repair using iterative projection (PBD style).
    Enforces:
    1. Circles inside [0,1]^2 (center distance to wall >= radius)
    2. Circles disjoint (distance between centers >= r_i + r_j)
    """
    P = P.copy()
    r = r.copy()
    n = len(r)
    tol = 1e-9
    
    for _ in range(iters):
        # 1. Enforce boundary constraints
        # For each circle, ensure r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        # If violated, we can either move center or shrink radius.
        # Moving center is better for maximizing sum of radii.
        # We project center to the feasible region defined by current r_i.
        for i in range(n):
            ri = r[i]
            # Box constraints: x_i in [ri, 1-ri], y_i in [ri, 1-ri]
            # If ri > 0.5, it's impossible to fit in unit square.
            # We assume ri < 0.5 from optimizer, but clamp just in case.
            if ri > 0.5 - 1e-6:
                r[i] = 0.5 - 1e-6
                ri = r[i]
            
            x_min, x_max = ri, 1.0 - ri
            y_min, y_max = ri, 1.0 - ri
            
            # Project P[i] to this box
            if P[i, 0] < x_min: P[i, 0] = x_min
            elif P[i, 0] > x_max: P[i, 0] = x_max
            
            if P[i, 1] < y_min: P[i, 1] = y_min
            elif P[i, 1] > y_max: P[i, 1] = y_max

        # 2. Enforce pairwise disjointness
        # Iterate over pairs. If overlap, push apart.
        # To converge faster, we can do multiple passes or use a queue, 
        # but simple Gauss-Seidel style is robust enough for N=26.
        for i in range(n):
            for j in range(i + 1, n):
                delta = P[i] - P[j]
                dist = np.linalg.norm(delta)
                min_dist = r[i] + r[j]
                
                if dist < min_dist - tol:
                    # Calculate push vector
                    if dist < 1e-12:
                        # Random push if coincident
                        angle = np.random.uniform(0, 2*np.pi)
                        u = np.array([np.cos(angle), np.sin(angle)])
                        dist = 1e-12 # avoid div by zero in normalization
                    else:
                        u = delta / dist
                    
                    overlap = min_dist - dist
                    # Push each circle by half the overlap
                    # Note: This might violate boundary again, which will be fixed in next iteration
                    push = 0.5 * overlap * u
                    P[i] += push
                    P[j] -= push
                    
                    # Immediate re-clamp to boundaries to prevent flying out
                    for k in (i, j):
                        rk = r[k]
                        if rk > 0.5 - 1e-6:
                            r[k] = 0.5 - 1e-6
                            rk = r[k]
                        P[k, 0] = np.clip(P[k, 0], rk, 1.0 - rk)
                        P[k, 1] = np.clip(P[k, 1], rk, 1.0 - rk)

    return P, r


def _init_grid():
    # A 5x5 grid gives 25 circles; we need 26. Use 5x5 and add one, then repair.
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    # 25 cells; add a 26th in the center-ish of the largest gap (use a small circle)
    # Place it at a corner-ish offset; repair will fix overlaps.
    cells.append([0.1, 0.1, 0.05])
    P = np.array([c[:2] for c in cells], dtype=float)
    r = np.array([c[2] for c in cells], dtype=float)
    return P, r


def _init_honeycomb():
    # Hexagonal packing: rows offset. For 26, use 5 rows with 5,6,5,5,5 pattern.
    rows = [5, 6, 5, 5, 5]
    cells = []
    row_y = [0.12, 0.31, 0.5, 0.69, 0.88]
    for ri, count in enumerate(rows):
        y = row_y[ri]
        for i in range(count):
            if count == 6:
                x = (i + 0.5) / 6
            else:
                # offset by half spacing
                x = (i + 0.5 + 0.25) / 5
                x = min(0.9, max(0.1, x))
            r = 0.08
            cells.append([x, y, r])
    P = np.array([c[:2] for c in cells], dtype=float)
    r = np.array([c[2] for c in cells], dtype=float)
    return P, r


def _compute_forces(P, r):
    """
    Compute repulsive forces for overlaps and wall collisions.
    Returns dP (forces on centers) and dr (forces on radii).
    We use a smooth potential: V = sum max(0, overlap)^2.
    Gradient of V w.r.t P and r gives the forces to apply (negative gradient).
    """
    n = len(r)
    dP = np.zeros_like(P)
    dr = np.zeros_like(r)
    
    # Pairwise
    delta = P[:, None, :] - P[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dist, 1.0) # Avoid div by 0 on diagonal
    
    # Overlap amount: r_i + r_j - dist
    overlap = (r[:, None] + r[None, :]) - dist
    # Only consider overlaps > 0
    # To be smooth, we can just use overlap where overlap > 0, but gradient is discontinuous.
    # A better smooth approx: use softplus or just squared overlap with a mask.
    # For gradient descent, a simple mask is often sufficient if step size is small.
    # Let's use the gradient of 0.5 * overlap^2 for overlapping pairs.
    # d/dP_i (0.5 * (r_i+r_j-d_ij)^2) = -(r_i+r_j-d_ij) * d(d_ij)/dP_i
    # d(d_ij)/dP_i = (P_i - P_j) / d_ij
    
    mask = overlap > 0
    # To avoid division by zero in dist, we already set diagonal to 1.
    # But for non-overlapping pairs, mask is False, so we don't care.
    # However, dist can be 0 for overlapping pairs if centers coincide.
    # Let's add a small epsilon to dist in the denominator for force calculation.
    dist_safe = np.where(dist < 1e-6, 1e-6, dist)
    
    # Force magnitude for pair i,j: overlap * (unit vector)
    # unit vector from j to i: (P_i - P_j) / dist
    unit_vec = delta / dist_safe[:, :, None]
    
    # Force on i due to j: -grad V = - (overlap) * (P_i - P_j)/dist
    # Wait, V = 0.5 * overlap^2. dV/dP_i = overlap * d(overlap)/dP_i = overlap * (- (P_i-P_j)/dist)
    # So force (negative gradient) = overlap * (P_i-P_j)/dist
    force_mag = overlap * mask
    # Sum over j for each i
    # F_i = sum_j force_mag[i,j] * unit_vec[i,j]
    dP += np.sum(force_mag[:, :, None] * unit_vec, axis=1)
    
    # Force on radii: dV/dr_i = overlap (for each j where mask is true)
    dr += np.sum(force_mag, axis=1)
    
    # Walls
    # Distance to wall: min(x, 1-x, y, 1-y)
    # Overlap with wall: r_i - dist_to_wall
    dx = np.minimum(P[:, 0], 1.0 - P[:, 0])
    dy = np.minimum(P[:, 1], 1.0 - P[:, 1])
    dist_wall = np.minimum(dx, dy)
    wall_overlap = r - dist_wall
    
    mask_wall = wall_overlap > 0
    
    # Force on center: push away from wall
    # If x < 0.5, wall is x=0, push +x. If x > 0.5, wall is x=1, push -x.
    # Gradient of dist_wall w.r.t x:
    # d(min(x, 1-x))/dx = 1 if x<0.5, -1 if x>0.5
    # d(dist_wall)/dx = d(min(dx, dy))/dx.
    # If dx < dy, dist_wall = dx. d(dist_wall)/dx = sign(x-0.5) * -1? No.
    # d(x)/dx = 1. d(1-x)/dx = -1.
    # If x < 0.5, min is x. d(min)/dx = 1.
    # If x > 0.5, min is 1-x. d(min)/dx = -1.
    # So d(dist_wall)/dx = 1 if P[i,0]<0.5 else -1 (if dx<dy)
    
    # We want to minimize V_wall = 0.5 * wall_overlap^2.
    # Force = -grad V.
    # dV/dr = wall_overlap. So dr -= wall_overlap * mask_wall? No, we want to increase r to maximize sum, but here we are minimizing energy.
    # In the main loop, we will do:
    # P += lr * dP (where dP is repulsive force)
    # r += lr * (1 - repulsive_dr) ? 
    # Actually, let's just return the "repulsive" forces.
    # The objective is to maximize sum(r).
    # We can treat this as:
    # Update P: P += lr * F_rep
    # Update r: r += lr * (1 - F_rep_r) ... this is getting complicated.
    
    # Let's stick to a simpler scheme:
    # 1. Compute repulsive forces on P.
    # 2. Compute repulsive forces on r (which effectively shrink r if overlapping).
    # 3. We also have a "growth" force on r (+1).
    # So net force on r = 1 - F_rep_r.
    
    # Wall force on P:
    # If x < 0.5 and overlapping, push +x.
    # If x > 0.5 and overlapping, push -x.
    # Magnitude is wall_overlap.
    
    fx = np.zeros(n)
    fy = np.zeros(n)
    for i in range(n):
        if not mask_wall[i]:
            continue
        mag = wall_overlap[i]
        if dx[i] < dy[i]:
            # Constrained by x
            if P[i, 0] < 0.5:
                fx[i] = mag
            else:
                fx[i] = -mag
        else:
            # Constrained by y
            if P[i, 1] < 0.5:
                fy[i] = mag
            else:
                fy[i] = -mag
                
    dP[:, 0] += fx
    dP[:, 1] += fy
    
    # Force on r from walls: shrink r if overlapping wall
    dr += wall_overlap * mask_wall
    
    return dP, dr


def _optimize(P, r, max_iters=2000, lr_start=0.01, lr_end=0.001):
    """
    Gradient descent with adaptive step size.
    Objective: Maximize sum(r) subject to no overlaps.
    We use a penalty method:
    L = sum(r) - alpha * Potential
    Gradient ascent on L:
    P += lr * grad_P (where grad_P is repulsive force)
    r += lr * (1 - grad_r) (where grad_r is repulsive force on r)
    """
    n = len(r)
    best_score = _score(P, r)
    best_P, best_r = P.copy(), r.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
            
        # Compute forces
        dP, dr_rep = _compute_forces(P, r)
        
        # Net force on r: +1 for growth, -dr_rep for repulsion
        dr_net = 1.0 - dr_rep
        
        # Adaptive learning rate
        # Decay lr linearly or exponentially
        lr = lr_start * (lr_end / lr_start) ** (it / max_iters)
        
        # Update
        P_new = P + lr * dP
        r_new = r + lr * dr_net
        
        # Clip to valid bounds
        P_new[:, 0] = np.clip(P_new[:, 0], 1e-6, 1.0 - 1e-6)
        P_new[:, 1] = np.clip(P_new[:, 1], 1e-6, 1.0 - 1e-6)
        r_new = np.clip(r_new, 1e-6, 0.5)
        
        # Check score
        s = _score(P_new, r_new)
        if s > best_score:
            best_score = s
            best_P, best_r = P_new.copy(), r_new.copy()
            
        P, r = P_new, r_new
        
        # Optional: if no overlaps, we can increase lr or just keep going
        # The repulsive forces will be zero if no overlaps, so r will grow linearly.
        # This might lead to huge r and then overlaps. The forces will kick in.
        
    return best_P, best_r, best_score


def construct():
    t0 = time.time()
    # Try multiple initializations and keep the best
    inits = [
        _init_grid(),
        _init_honeycomb(),
    ]
    # Add a random-ish perturbation init
    rng = np.random.default_rng(42)
    P_rand = rng.uniform(0.1, 0.9, size=(N, 2))
    r_rand = np.full(N, 0.05) # Start smaller to allow growth
    inits.append((P_rand, r_rand))

    best_P, best_r, best_score = None, None, -1.0
    for P0, r0 in inits:
        if time.time() > DEADLINE:
            break
        # Quick repair to start from a feasible state
        P0, r0 = _repair(P0.copy(), r0.copy(), iters=50)
        # Optimize
        P_opt, r_opt, s = _optimize(P0, r0, max_iters=3000, lr_start=0.02, lr_end=0.001)
        if s > best_score:
            best_score = s
            best_P, best_r = P_opt, r_opt
        if time.time() > DEADLINE:
            break

    if best_P is None:
        P0, r0 = _init_grid()
        best_P, best_r = _repair(P0, r0, iters=50)

    # Final validity repair
    best_P, best_r = _repair(best_P, best_r, iters=100)

    # Ensure strict validity: if any residual overlap, shrink
    if not _valid(best_P, best_r, tol=1e-9):
        # Aggressive shrink
        for _ in range(50):
            pair, wallc = _clearances(best_P, best_r)
            if np.min(pair) > -1e-10 and np.min(wallc) > -1e-10:
                break
            # Find worst violation
            all_c = np.concatenate([pair.ravel(), wallc.ravel()])
            idx = np.argmin(all_c)
            if idx < pair.size:
                i, j = np.unravel_index(idx, pair.shape)
                best_r[i] *= 0.98
                best_r[j] *= 0.98
            else:
                i = idx - pair.size
                best_r[i] *= 0.98
        best_P, best_r = _repair(best_P, best_r, iters=50)

    # Final check
    if not _valid(best_P, best_r, tol=1e-9):
        # Last resort: shrink all radii until valid
        scale = 1.0
        for _ in range(100):
            if _valid(best_P, best_r * scale, tol=1e-9):
                break
            scale *= 0.95
        best_r = best_r * scale
        best_P, best_r = _repair(best_P, best_r, iters=50)

    out = np.column_stack([best_P, best_r])
    return out
