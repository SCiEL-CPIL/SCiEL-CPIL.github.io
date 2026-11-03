"""
Molecular Dynamics (soft repulsive potential) + L-BFGS-B polish
for packing 300 equal circles in a unit circle.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
TIME_LIMIT = 118.0


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk."""
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        spacing = 0.08
        k = int(1.2 / spacing) + 2
        xs, ys = [], []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                xs.append(x)
                ys.append(y)
        pts = np.column_stack([xs, ys])
        r = np.hypot(pts[:, 0], pts[:, 1])
        inside = r <= 1.0
        pts = pts[inside]
        r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius."""
    n = len(pts)
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _md_dynamics(pts, start_time, time_limit):
    """
    Molecular dynamics with soft repulsive potentials.
    - Pair potential: soft repulsion for d < 2r (we use unit "particle radius" = 0.5 in normalized coords)
    - Wall potential: soft repulsion from unit circle boundary
    - Velocity Verlet integration with cooling
    """
    n = len(pts)
    pos = pts.copy().astype(float)
    vel = np.zeros_like(pos)
    
    # Parameters
    dt = 0.005          # time step
    k_pair = 50.0       # pair spring constant
    r_pair = 0.055      # "particle radius" (half of min separation target)
    k_wall = 200.0      # wall spring constant
    damping = 0.98      # velocity damping per step
    n_steps = 20000
    temp_start = 0.001  # initial random kick magnitude
    
    rng = np.random.default_rng(123)
    
    # Initial random kick
    vel += rng.normal(0, temp_start, pos.shape)
    
    best_pos = pos.copy()
    best_score = _score(pos)
    
    for step in range(n_steps):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pair interactions: O(n^2) but vectorized
        d2 = np.sum(pos * pos, axis=1)[:, None] - 2.0 * (pos @ pos.T) + np.sum(pos * pos, axis=1)[None, :]
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(np.maximum(d2, 1e-15))
        
        # Only interact if d < 2*r_pair
        mask = d < 2.0 * r_pair
        # Force magnitude: k_pair * (2*r_pair - d) for d < 2*r_pair, else 0
        # Force direction: along (pos_j - pos_i) normalized
        # F_i = sum_j k_pair * (2*r_pair - d_ij) * (pos_j - pos_i) / d_ij
        # Vectorized: for each pair (i,j) with d_ij < 2*r_pair
        # We compute the force on i from j as: k_pair * (2*r_pair - d_ij) / d_ij * (pos_j - pos_i)
        
        # Use broadcasting for efficiency
        diff = pos[None, :, :] - pos[:, None, :]  # diff[i,j] = pos_j - pos_i
        # Force on i from j: k_pair * (2*r_pair - d_ij) / d_ij * diff[i,j] for d_ij < 2*r_pair
        factor = np.zeros_like(d)
        factor[mask] = k_pair * (2.0 * r_pair - d[mask]) / d[mask]
        # forces[i] = sum_j factor[i,j] * diff[i,j]
        forces = np.sum(factor[:, :, None] * diff, axis=1)
        
        # Wall forces: push inward if |pos| > 1 - r_wall
        r_wall = 0.005  # wall margin
        dist_to_center = np.hypot(pos[:, 0], pos[:, 1])
        outside = dist_to_center > (1.0 - r_wall)
        if np.any(outside):
            penetration = dist_to_center[outside] - (1.0 - r_wall)
            # Force direction: toward center, magnitude k_wall * penetration
            forces[outside] -= k_wall * penetration[:, None] * pos[outside] / dist_to_center[outside, None]
        
        # Velocity Verlet: half-step velocity
        vel_half = vel + 0.5 * dt * forces
        # Full-step position
        pos = pos + dt * vel_half
        # Recompute forces at new position (approximate: reuse old forces for simplicity, 
        # but let's do a proper half-step)
        # For simplicity, use Euler with damping
        vel = vel_half * damping
        pos = pos + dt * vel
        
        # Enforce hard boundary: clip to unit disk
        rad = np.hypot(pos[:, 0], pos[:, 1])
        out = rad > 1.0
        if np.any(out):
            pos[out] = pos[out] / rad[out, None]
            vel[out] = 0.0  # kill velocity on boundary hit
        
        # Track best
        if step % 100 == 0:
            s = _score(pos)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
        
        # Reduce damping over time (annealing)
        if step < n_steps // 2:
            damping = 0.98
        else:
            damping = 0.995  # less damping to allow settling
    
    # Final check
    s = _score(pos)
    if s > best_score:
        best_score = s
        best_pos = pos.copy()
    
    return best_pos


def _polish(pts, start_time, time_limit):
    """
    L-BFGS-B polish using a smooth objective that approximates the min-separation.
    Objective: -min_pair_dist (soft approximation via log-sum-exp)
    """
    n = len(pts)
    x0 = pts.reshape(-1)
    
    def objective_and_grad(x):
        pts_x = x.reshape(-1, 2)
        # Pair distances
        d2 = np.sum(pts_x * pts_x, axis=1)[:, None] - 2.0 * (pts_x @ pts_x.T) + np.sum(pts_x * pts_x, axis=1)[None, :]
        np.fill_diagonal(d2, 1e15)
        d = np.sqrt(np.maximum(d2, 1e-15))
        
        # Soft min: use log-sum-exp with small beta
        beta = 100.0
        # We want to maximize the min distance, so minimize -min_dist
        # Soft approx: -1/beta * log(sum(exp(-beta * d_ij)))
        # But this is expensive. Instead, use a simpler penalty:
        # Penalize pairs with d < target
        target = 0.105  # target separation
        violations = np.maximum(0, target - d)
        # Sum of violations
        obj = np.sum(violations ** 2)
        
        # Boundary violations
        rad = np.hypot(pts_x[:, 0], pts_x[:, 1])
        bound_viol = np.maximum(0, rad - (1.0 - 0.005))
        obj += 10.0 * np.sum(bound_viol ** 2)
        
        # Gradient
        grad = np.zeros_like(x)
        # For pair violations: d(1/2 * (target-d)^2)/dx_i = -(target-d) * d/dx_i d
        # d/dx_i d_ij = (x_i - x_j) / d_ij
        for i in range(n):
            for j in range(i + 1, n):
                dij = d[i, j]
                if dij < target and dij > 1e-10:
                    coeff = (target - dij) / dij
                    grad[2*i] += coeff * (x[2*i] - x[2*j])
                    grad[2*i+1] += coeff * (x[2*i+1] - x[2*j+1])
                    grad[2*j] -= coeff * (x[2*i] - x[2*j])
                    grad[2*j+1] -= coeff * (x[2*i+1] - x[2*j+1])
        
        # Boundary gradient
        for i in range(n):
            if rad[i] > (1.0 - 0.005) and rad[i] > 1e-10:
                coeff = 20.0 * (rad[i] - (1.0 - 0.005)) / rad[i]
                grad[2*i] += coeff * x[2*i]
                grad[2*i+1] += coeff * x[2*i+1]
        
        return obj, grad
    
    try:
        # Use a vectorized version for speed
        def obj_grad_vec(x):
            pts_x = x.reshape(-1, 2)
            d2 = np.sum(pts_x * pts_x, axis=1)[:, None] - 2.0 * (pts_x @ pts_x.T) + np.sum(pts_x * pts_x, axis=1)[None, :]
            np.fill_diagonal(d2, 1e15)
            d = np.sqrt(np.maximum(d2, 1e-15))
            
            target = 0.105
            viol = np.maximum(0, target - d)
            obj = np.sum(viol ** 2)
            
            rad = np.hypot(pts_x[:, 0], pts_x[:, 1])
            bound_viol = np.maximum(0, rad - (1.0 - 0.005))
            obj += 10.0 * np.sum(bound_viol ** 2)
            
            # Gradient for pair terms
            # grad_i = sum_j (target - d_ij) / d_ij * (x_i - x_j) for d_ij < target
            coeff_mat = np.zeros_like(d)
            mask = (d < target) & (d > 1e-10)
            coeff_mat[mask] = (target - d[mask]) / d[mask]
            # grad[i] = sum_j coeff_mat[i,j] * (x_i - x_j)
            # = sum_j coeff_mat[i,j] * x_i - sum_j coeff_mat[i,j] * x_j
            sum_coeff = np.sum(coeff_mat, axis=1)  # shape (n,)
            # First term: sum_coeff[i] * x_i
            grad_pair = sum_coeff[:, None] * pts_x
            # Second term: sum_j coeff_mat[i,j] * x_j = (coeff_mat @ pts_x)
            grad_pair -= coeff_mat @ pts_x
            # Note: coeff_mat is symmetric, so this should be antisymmetric in the pair force
            # Actually, let me re-derive: force on i from j is coeff * (x_i - x_j)
            # So grad_i = sum_j coeff_ij * (x_i - x_j)
            # This is what we computed.
            
            # Boundary gradient
            grad_bound = np.zeros_like(pts_x)
            mask_b = rad > (1.0 - 0.005)
            if np.any(mask_b):
                coeff_b = 20.0 * (rad[mask_b] - (1.0 - 0.005)) / rad[mask_b]
                grad_bound[mask_b] = coeff_b[:, None] * pts_x[mask_b]
            
            grad = (grad_pair + grad_bound).reshape(-1)
            return obj, grad
        
        res = minimize(obj_grad_vec, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6})
        
        if res.success or res.fun < obj_grad_vec(x0)[0]:
            new_pts = res.x.reshape(-1, 2)
            if np.all(np.isfinite(new_pts)):
                return new_pts
    except Exception:
        pass
    
    return pts


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    
    # Ensure valid start
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 1: Molecular Dynamics
    try:
        md_pts = _md_dynamics(pts, start_time, TIME_LIMIT)
        rad = np.hypot(md_pts[:, 0], md_pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            md_pts[out] = md_pts[out] / rad[out, None]
        md_score = _score(md_pts)
        if md_score > best_score:
            best_pts = md_pts
            best_score = md_score
    except Exception:
        pass
    
    # Phase 2: Polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10.0:
        try:
            polished_pts = _polish(best_pts, start_time, TIME_LIMIT)
            rad = np.hypot(polished_pts[:, 0], polished_pts[:, 1])
            out = rad > 1.0
            if np.any(out):
                polished_pts[out] = polished_pts[out] / rad[out, None]
            pol_score = _score(polished_pts)
            if pol_score > best_score:
                best_pts = polished_pts
                best_score = pol_score
        except Exception:
            pass
    
    # Final safety
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
    
    return best_pts
