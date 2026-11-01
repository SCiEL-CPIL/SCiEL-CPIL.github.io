"""
Combined strategy: Binary search on radius with rigid-body Newton solver on sparse contact graph.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

N = 50
HALF = 0.5

# EVOLVE-BLOCK-START

def _score(c):
    """
    Compute the largest common radius for centres c (N,2).
    Returns 0.0 if invalid.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Wall distances
    min_bound = np.min(HALF - np.abs(c))
    
    return min(min_pair, min_bound)


def _hex_seed(r):
    """
    Generate a hexagonal lattice seed with spacing approx 2r.
    """
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x_offset = 0.5 * dx if (row % 2) else 0.0
        x = -HALF + r + x_offset
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
        
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """
    Generate a simple grid seed.
    """
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _build_contact_graph(pts, r, margin=0.01):
    """
    Build sparse contact graph: pairs within (2r + margin) and wall contacts.
    Returns list of (i, j) for pair contacts and list of (i, axis, sign) for wall contacts.
    """
    pair_contacts = []
    wall_contacts = []
    
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    threshold = (2 * r + margin) ** 2
    for i in range(N):
        for j in range(i + 1, N):
            if dist_sq[i, j] < threshold:
                pair_contacts.append((i, j))
    
    for i in range(N):
        if pts[i, 0] < -HALF + r + margin:
            wall_contacts.append((i, 0, -1))
        if pts[i, 0] > HALF - r - margin:
            wall_contacts.append((i, 0, 1))
        if pts[i, 1] < -HALF + r + margin:
            wall_contacts.append((i, 1, -1))
        if pts[i, 1] > HALF - r - margin:
            wall_contacts.append((i, 1, 1))
    
    return pair_contacts, wall_contacts


def _newton_solve(pts, r, max_iter=100, tol=1e-12, deadline=None):
    """
    Solve exact distance constraints using Newton's method on sparse contact graph.
    For each contact pair (i,j), enforce |p_i - p_j| = 2r.
    For wall contacts, enforce p_i[axis] = boundary.
    """
    pts = pts.astype(np.float64).copy()
    n_vars = 2 * N
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        pair_contacts, wall_contacts = _build_contact_graph(pts, r, margin=0.02)
        
        if not pair_contacts and not wall_contacts:
            break
        
        # Build residual and Jacobian
        residual = np.zeros(n_vars)
        J = lil_matrix((n_vars, n_vars), dtype=np.float64)
        
        # Pair constraints: |p_i - p_j| = 2r
        for (i, j) in pair_contacts:
            diff = pts[i] - pts[j]
            dist = np.sqrt(np.sum(diff * diff))
            if dist < 1e-10:
                continue
            # Residual: dist - 2r (we want this to be 0)
            res = dist - 2 * r
            # Gradient w.r.t. p_i: diff/dist
            # Gradient w.r.t. p_j: -diff/dist
            grad_i = diff / dist
            grad_j = -diff / dist
            
            # We use the residual as a scalar, but we need to embed into n_vars system.
            # Use a simple approach: treat each constraint as two scalar equations isn't right.
            # Instead, use the fact that the constraint is scalar, and we have n_vars variables.
            # We'll use a least-squares-like approach with the Jacobian of the constraint.
            
            # For Newton, we need J * dx = -residual
            # But we have more variables than constraints typically.
            # Use pseudoinverse-like approach: dx = -J^T (J J^T + lambda I)^{-1} residual
            # For simplicity, build the full system and solve.
            
            # Actually, let's use a different approach: build the constraint Jacobian
            # and solve the underdetermined system using least squares.
            
            pass
        
        # Simpler approach: use projected gradient on the exact constraint
        # Let's use a different strategy: iterative relaxation
        
        # For each contact pair, project onto the constraint
        for (i, j) in pair_contacts:
            diff = pts[i] - pts[j]
            dist = np.sqrt(np.sum(diff * diff))
            if dist < 1e-10:
                continue
            # Move both points halfway toward the correct distance
            error = (dist - 2 * r) / (2 * dist)
            pts[i] += error * diff
            pts[j] -= error * diff
        
        # For wall contacts, project onto boundary
        for (i, axis, sign) in wall_contacts:
            if sign == -1:
                target = -HALF + r
                pts[i, axis] = target
            else:
                target = HALF - r
                pts[i, axis] = target
        
        # Check convergence
        pair_contacts2, _ = _build_contact_graph(pts, r, margin=0.005)
        max_viol = 0.0
        for (i, j) in pair_contacts2:
            diff = pts[i] - pts[j]
            dist = np.sqrt(np.sum(diff * diff))
            viol = abs(dist - 2 * r)
            max_viol = max(max_viol, viol)
        
        if max_viol < tol:
            break
    
    # Final feasibility check
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    
    success = (min_pair >= 2 * r - 1e-9) and (min_wall >= r - 1e-9)
    return success, pts


def _newton_rigid(pts, r, max_iter=200, tol=1e-13, deadline=None):
    """
    Rigid-body Newton solver: build contact graph, solve linear system for exact constraints.
    Uses sparse linear solver.
    """
    pts = pts.astype(np.float64).copy()
    n_vars = 2 * N
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        pair_contacts, wall_contacts = _build_contact_graph(pts, r, margin=0.03)
        
        n_constraints = len(pair_contacts) + len(wall_contacts)
        if n_constraints == 0:
            break
        
        # Build constraint Jacobian A (n_constraints x n_vars) and residual b
        A_rows = []
        b = []
        
        for (i, j) in pair_contacts:
            diff = pts[i] - pts[j]
            dist = np.sqrt(np.sum(diff * diff))
            if dist < 1e-10:
                continue
            res = dist - 2 * r
            grad = np.zeros(n_vars)
            grad[2*i] = diff[0] / dist
            grad[2*i+1] = diff[1] / dist
            grad[2*j] = -diff[0] / dist
            grad[2*j+1] = -diff[1] / dist
            A_rows.append(grad)
            b.append(res)
        
        for (i, axis, sign) in wall_contacts:
            if sign == -1:
                res = pts[i, axis] - (-HALF + r)
            else:
                res = pts[i, axis] - (HALF - r)
            grad = np.zeros(n_vars)
            grad[2*i + axis] = 1.0
            A_rows.append(grad)
            b.append(res)
        
        if not A_rows:
            break
        
        A = np.array(A_rows)
        b = np.array(b)
        
        # Solve least squares: A dx = -b
        # Use normal equations with regularization
        AA = A.T @ A
        AA += 1e-10 * np.eye(n_vars)  # Regularize for numerical stability
        Ab = A.T @ b
        dx = np.linalg.solve(AA, -Ab)
        
        # Damping
        alpha = 1.0
        for _ in range(20):
            new_pts = pts + alpha * dx.reshape(N, 2)
            # Check if this reduces total violation
            new_diff = new_pts[:, None, :] - new_pts[None, :, :]
            new_dist = np.sqrt(np.sum(new_diff * new_diff, axis=2))
            np.fill_diagonal(new_dist, np.inf)
            
            old_diff = pts[:, None, :] - pts[None, :, :]
            old_dist = np.sqrt(np.sum(old_diff * old_diff, axis=2))
            np.fill_diagonal(old_dist, np.inf)
            
            old_viol = np.sum(np.maximum(0, 2*r - old_dist)) + np.sum(np.maximum(0, -HALF + r - pts)) + np.sum(np.maximum(0, pts - (HALF - r)))
            new_viol = np.sum(np.maximum(0, 2*r - new_dist)) + np.sum(np.maximum(0, -HALF + r - new_pts)) + np.sum(np.maximum(0, new_pts - (HALF - r)))
            
            if new_viol < old_viol:
                pts = new_pts
                break
            alpha *= 0.5
        
        # Check convergence
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        min_wall = np.min(HALF - np.abs(pts))
        
        if min_pair >= 2 * r - tol and min_wall >= r - tol:
            break
    
    # Final feasibility check
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    
    success = (min_pair >= 2 * r - 1e-9) and (min_wall >= r - 1e-9)
    return success, pts


def _lbfgs_feasible(pts, r, deadline=None, max_iter=200):
    """
    Try to find a feasible configuration for radius r using L-BFGS-B.
    """
    pts = pts.astype(np.float64).copy()
    
    def _violation_and_grad(x, r):
        pts_x = x.reshape(N, 2)
        diff = pts_x[:, None, :] - pts_x[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        pair_viol = np.maximum(0, 2 * r - dist)
        weight = np.where((dist < 2 * r) & (dist > 1e-12), (2 * r - dist) / dist, 0.0)
        grad_pairs = np.sum(diff * weight[:, :, None], axis=1)
        wall_viol = np.zeros(N)
        grad_walls = np.zeros_like(pts_x)
        too_left_x = pts_x[:, 0] < -HALF + r
        if np.any(too_left_x):
            wall_viol[too_left_x] += (-HALF + r - pts_x[too_left_x, 0])
            grad_walls[too_left_x, 0] -= 1.0
        too_right_x = pts_x[:, 0] > HALF - r
        if np.any(too_right_x):
            wall_viol[too_right_x] += (pts_x[too_right_x, 0] - (HALF - r))
            grad_walls[too_right_x, 0] += 1.0
        too_low_y = pts_x[:, 1] < -HALF + r
        if np.any(too_low_y):
            wall_viol[too_low_y] += (-HALF + r - pts_x[too_low_y, 1])
            grad_walls[too_low_y, 1] -= 1.0
        too_high_y = pts_x[:, 1] > HALF - r
        if np.any(too_high_y):
            wall_viol[too_high_y] += (pts_x[too_high_y, 1] - (HALF - r))
            grad_walls[too_high_y, 1] += 1.0
        total_viol = np.sum(pair_viol) + np.sum(wall_viol)
        total_grad = grad_pairs + grad_walls
        return total_viol, total_grad.ravel()
    
    x0 = pts.ravel()
    bounds = [(-HALF + r, HALF - r)] * (2 * N)
    
    def obj(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        v, _ = _violation_and_grad(x, r)
        return v
    
    def grad(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros(2 * N)
        _, g = _violation_and_grad(x, r)
        return g
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', jac=grad,
            bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        pts = result.x.reshape(N, 2)
    except Exception:
        pass
    
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    
    success = (min_pair >= 2 * r - 1e-8) and (min_wall >= r - 1e-8)
    return success, pts


def _polish(pts, deadline):
    """
    High-precision polish: alternate between rigid-body Newton and L-BFGS.
    """
    current_r = _score(pts)
    
    for _ in range(12):
        if time.time() > deadline - 3:
            break
        target_r = current_r + 0.00003
        ok, polished = _newton_rigid(pts, target_r, max_iter=150, deadline=deadline)
        if not ok:
            ok, polished = _lbfgs_feasible(pts, target_r, deadline=deadline, max_iter=100)
        if ok:
            new_r = _score(polished)
            if new_r > current_r + 1e-10:
                pts = polished
                current_r = new_r
            else:
                break
        else:
            break
    
    return pts, current_r


def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    r_low = 0.068
    r_high = 0.073
    
    best_pts = None
    best_r = 0.0
    
    init_pts = _hex_seed(r_low)
    ok, init_pts = _newton_rigid(init_pts, r_low, max_iter=200, deadline=deadline)
    if not ok:
        ok, init_pts = _lbfgs_feasible(init_pts, r_low, deadline=deadline, max_iter=150)
    if ok:
        best_pts = init_pts
        best_r = _score(best_pts)
        r_low = best_r
    
    for _ in range(25):
        if time.time() > deadline - 15:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        found = False
        candidate_pts = None
        
        seeds = [
            _hex_seed(r_mid),
            _grid_seed(r_mid),
            _hex_seed(r_mid) + np.random.randn(N, 2) * 0.002,
            _grid_seed(r_mid) + np.random.randn(N, 2) * 0.002
        ]
        
        for seed in seeds:
            if time.time() > deadline - 15:
                break
            
            seed = np.clip(seed, -HALF + r_mid, HALF - r_mid)
            
            # Try rigid-body Newton first
            ok, pts = _newton_rigid(seed, r_mid, max_iter=150, deadline=deadline)
            if not ok:
                ok, pts = _lbfgs_feasible(seed, r_mid, deadline=deadline, max_iter=150)
            if ok:
                found = True
                candidate_pts = pts
                break
        
        if found:
            r_low = r_mid
            best_pts = candidate_pts
            best_r = _score(best_pts)
            if best_r > r_low:
                r_low = best_r
        else:
            r_high = r_mid
        
        if r_high - r_low < 1e-6:
            break
    
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_r = _polish(best_pts, deadline)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    return best_pts

# EVOLVE-BLOCK-END
