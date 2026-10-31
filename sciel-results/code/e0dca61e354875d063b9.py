"""
Voronoi-based circle packing in the unit square with smooth surrogate objective.

Strategy:
1. Represent the packing by N seed points (x, y).
2. Use a smooth soft-min surrogate for the radius computation to avoid nonsmoothness.
3. Maximize sum of radii via L-BFGS-B with multi-start and trust-constr polish.
4. Use diverse initializations: grid, hexagonal, corner-heavy, random.
5. Deadline guard for 150s limit.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize, trust_constr, LinearConstraint, NonlinearConstraint

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin

    def compute_radii(pts):
        """Compute exact (nonsmooth) radii for validation."""
        x = pts[:, 0]
        y = pts[:, 1]
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dists, np.inf)
        min_dist_to_neighbor = np.min(dists, axis=1) / 2.0
        radii = np.minimum(r_bound, min_dist_to_neighbor)
        return radii

    def soft_min(a, beta):
        """Smooth approximation of min(a) using log-sum-exp."""
        # soft_min(a) ≈ -(1/beta) * log(sum(exp(-beta * a)))
        # This is numerically stable
        m = np.min(a)
        return m - np.log(np.sum(np.exp(-beta * (a - m)))) / beta

    def compute_radii_smooth(pts, beta):
        """Compute smooth radii using soft-min surrogate."""
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Boundary distances for each point
        r_bound = np.stack([
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        ], axis=1)  # shape (N, 2)
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dists, np.inf)
        neighbor_dists = dists / 2.0  # shape (N, N), diagonal is inf
        
        # For each point i, we want soft_min of [boundary_dists_i, neighbor_dists_i]
        # Stack all constraints for point i
        all_constraints = np.concatenate([r_bound, neighbor_dists], axis=1)  # (N, 2+N)
        
        # Soft min along axis 1 for each point
        radii = np.array([soft_min(all_constraints[i], beta) for i in range(N)])
        
        return radii

    def objective_smooth(pts_flat, beta):
        """Objective for optimization: negative sum of smooth radii."""
        pts = pts_flat.reshape(N, 2)
        radii = compute_radii_smooth(pts, beta)
        return -np.sum(radii)

    def objective_exact(pts_flat):
        """Objective using exact radii."""
        pts = pts_flat.reshape(N, 2)
        radii = compute_radii(pts)
        return -np.sum(radii)

    def gradient_objective(pts_flat, beta):
        """Numerical gradient of smooth objective."""
        eps = 1e-6
        grad = np.zeros_like(pts_flat)
        base_val = objective_smooth(pts_flat, beta)
        for i in range(len(pts_flat)):
            pts_plus = pts_flat.copy()
            pts_plus[i] += eps
            val_plus = objective_smooth(pts_plus, beta)
            grad[i] = (val_plus - base_val) / eps
            pts_plus[i] = pts_flat[i] - eps
            val_minus = objective_smooth(pts_plus, beta)
            grad[i] = (base_val - val_minus) / eps
            pts_plus[i] = pts_flat[i]
        return grad

    def pairwise_constraints(pts_flat):
        """
        Constraints for trust-constr: dist(i,j) >= r_i + r_j for all i<j,
        and r_i <= dist_to_boundary for all i.
        We use the exact radii formulation: r_i = min(dist_bound_i, min_j dist(i,j)/2)
        For trust-constr, we need explicit constraints.
        Let's define auxiliary variables t_i >= 0 as radii.
        Constraints:
          t_i <= dist_bound_i  (boundary)
          t_i + t_j <= dist(i,j)  for all i < j  (pairwise)
        Objective: maximize sum(t_i)
        """
        pts = pts_flat[:2*N].reshape(N, 2)
        t = pts_flat[2*N:]  # radii
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Boundary constraints: t_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx * dx + dy * dy)
        
        # Build constraint arrays
        # g_boundary: r_bound - t >= 0
        g_boundary = r_bound - t
        
        # g_pairwise: dist(i,j) - t_i - t_j >= 0
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        g_pairwise = np.array([dists[i, j] - t[i] - t[j] for i, j in pairs])
        
        return np.concatenate([g_boundary, g_pairwise])

    def jacobian_constraints(pts_flat):
        """Jacobian of pairwise_constraints."""
        pts = pts_flat[:2*N].reshape(N, 2)
        t = pts_flat[2*N:]
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dists, np.inf)
        
        n_cons = N + N * (N - 1) // 2
        jac = np.zeros((n_cons, 3 * N))
        
        # Boundary constraints: r_bound_i - t_i >= 0
        for i in range(N):
            # d(r_bound_i)/dx_i
            if x[i] < 0.5:
                jac[i, 2*i] = 1.0
            else:
                jac[i, 2*i] = -1.0
            if y[i] < 0.5:
                jac[i, 2*i+1] = 1.0
            else:
                jac[i, 2*i+1] = -1.0
            jac[i, 2*N+i] = -1.0
        
        # Pairwise constraints
        idx = N
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d > 1e-10:
                    dxij = x[i] - x[j]
                    dyij = y[i] - y[j]
                    # d(dist(i,j))/dx_i = dxij/d
                    jac[idx, 2*i] = dxij / d
                    jac[idx, 2*i+1] = dyij / d
                    # d(dist(i,j))/dx_j = -dxij/d
                    jac[idx, 2*j] = -dxij / d
                    jac[idx, 2*j+1] = -dyij / d
                    # d(dist(i,j) - t_i - t_j)/dt_i = -1
                    jac[idx, 2*N+i] = -1.0
                    # d(dist(i,j) - t_i - t_j)/dt_j = -1
                    jac[idx, 2*N+j] = -1.0
                idx += 1
        
        return jac

    def objective_trust(pts_flat):
        """Objective for trust-constr: minimize -sum(t)."""
        return -np.sum(pts_flat[2*N:])

    def gradient_trust(pts_flat):
        """Gradient of objective for trust-constr."""
        grad = np.zeros_like(pts_flat)
        grad[2*N:] = -1.0
        return grad

    def optimize_smooth_lbfgs(start_pts, beta, maxiter=200):
        """Optimize using L-BFGS-B with smooth objective."""
        x0 = np.concatenate([start_pts.flatten(), compute_radii(start_pts)])
        bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        
        try:
            res = minimize(
                objective_smooth,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < objective_smooth(x0, beta):
                best_pts = res.x[:2*N].reshape(N, 2)
                best_val = -np.sum(compute_radii(best_pts))
                return best_pts, best_val
        except Exception:
            pass
        
        return start_pts, -np.sum(compute_radii(start_pts))

    def optimize_trust_constr(start_pts, maxiter=100):
        """Optimize using trust-constr with explicit constraints."""
        t_init = compute_radii(start_pts)
        x0 = np.concatenate([start_pts.flatten(), t_init])
        
        # Bounds
        bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        
        # Constraints: g(x) >= 0
        n_cons = N + N * (N - 1) // 2
        cons = NonlinearConstraint(pairwise_constraints, 0.0, np.inf, jac=jacobian_constraints)
        
        try:
            res = minimize(
                objective_trust,
                x0,
                jac=gradient_trust,
                bounds=bounds,
                constraints=[cons],
                method='trust-constr',
                options={'maxiter': maxiter, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            if res.success or res.fun < objective_trust(x0):
                best_pts = res.x[:2*N].reshape(N, 2)
                # Recompute radii exactly
                best_radii = compute_radii(best_pts)
                best_val = -np.sum(best_radii)
                return best_pts, best_val
        except Exception:
            pass
        
        return start_pts, -np.sum(compute_radii(start_pts))

    def generate_grid_start():
        """Generate a 5x5 + 1 grid start."""
        x_grid, y_grid = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
        base_grid = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        extra_pt = np.array([[0.5, 0.5]])
        pts = np.vstack([base_grid, extra_pt])
        return pts

    def generate_hex_start():
        """Generate hexagonal lattice start."""
        s = 0.18
        hex_pts = []
        for i in range(6):
            for j in range(5):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * 0.866
                hex_pts.append([x, y])
        hex_pts = np.array(hex_pts[:26])
        hex_pts = (hex_pts - hex_pts.min(axis=0)) / (hex_pts.max(axis=0) - hex_pts.min(axis=0))
        hex_pts = hex_pts * 0.9 + 0.05
        return hex_pts

    def generate_corner_heavy_start():
        """Generate start with points near corners and edges."""
        pts = np.zeros((N, 2))
        # Corners
        corner_margin = 0.1
        corners = [
            [corner_margin, corner_margin],
            [1-corner_margin, corner_margin],
            [corner_margin, 1-corner_margin],
            [1-corner_margin, 1-corner_margin]
        ]
        for i, c in enumerate(corners):
            pts[i] = c
        # Edge midpoints
        edge_pts = [
            [0.5, corner_margin], [0.5, 1-corner_margin],
            [corner_margin, 0.5], [1-corner_margin, 0.5]
        ]
        for i, e in enumerate(edge_pts):
            pts[4+i] = e
        # Remaining random in interior
        remaining = np.random.rand(N-8, 2) * 0.6 + 0.2
        pts[8:] = remaining
        return pts

    def generate_random_start():
        """Generate random start."""
        return np.random.rand(N, 2)

    def generate_ring_start():
        """Generate points on a ring pattern."""
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        r1 = 0.3
        r2 = 0.6
        pts1 = np.column_stack([0.5 + r1*np.cos(angles), 0.5 + r1*np.sin(angles)])
        angles2 = np.linspace(0, 2*np.pi, 13, endpoint=False) + 0.1
        pts2 = np.column_stack([0.5 + r2*np.cos(angles2), 0.5 + r2*np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
        return pts

    # Generate initial candidates
    candidates = []
    
    # Grid starts
    for _ in range(3):
        candidates.append(generate_grid_start() + np.random.normal(0, 0.01, (N, 2)))
    
    # Hex starts
    for _ in range(3):
        candidates.append(generate_hex_start() + np.random.normal(0, 0.02, (N, 2)))
    
    # Corner heavy starts
    for _ in range(3):
        candidates.append(generate_corner_heavy_start())
    
    # Random starts
    for _ in range(10):
        candidates.append(generate_random_start())
    
    # Ring starts
    for _ in range(3):
        candidates.append(generate_ring_start() + np.random.normal(0, 0.02, (N, 2)))

    best_val = np.inf
    best_result = None
    
    # Phase 1: Smooth L-BFGS-B optimization from each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:  # Leave time for trust-constr
            break
        
        # Use decreasing beta for smoother -> sharper
        for beta in [20, 50, 100]:
            if time.time() > deadline - 30:
                break
            try:
                pts_opt, val_opt = optimize_smooth_lbfgs(cand, beta, maxiter=100)
                if val_opt < best_val:
                    best_val = val_opt
                    best_result = pts_opt
                    cand = pts_opt  # Continue from improved solution
            except Exception:
                pass

    # Phase 2: Trust-constr polish on best result
    if best_result is not None and time.time() < deadline - 10:
        try:
            pts_opt, val_opt = optimize_trust_constr(best_result, maxiter=50)
            if val_opt < best_val:
                best_val = val_opt
                best_result = pts_opt
        except Exception:
            pass

    # Phase 3: Final L-BFGS-B polish with high beta
    if best_result is not None and time.time() < deadline - 5:
        try:
            for beta in [200, 500]:
                if time.time() > deadline - 5:
                    break
                pts_opt, val_opt = optimize_smooth_lbfgs(best_result, beta, maxiter=100)
                if val_opt < best_val:
                    best_val = val_opt
                    best_result = pts_opt
        except Exception:
            pass

    # Fallback
    if best_result is None:
        best_result = generate_grid_start()

    # Compute final exact radii
    radii = compute_radii(best_result)
    
    # Construct output
    result = np.column_stack((best_result[:, 0], best_result[:, 1], radii))
    
    # Validation and correction
    result[:, 2] = np.maximum(result[:, 2], 0.0)
    result[:, 0] = np.clip(result[:, 0], 0.0, 1.0)
    result[:, 1] = np.clip(result[:, 1], 0.0, 1.0)
    
    # Fix any overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((result[i, 0] - result[j, 0])**2 + (result[i, 1] - result[j, 1])**2)
            if dist < result[i, 2] + result[j, 2] - 1e-9:
                # Reduce both radii
                new_r = max(0.0, (dist - 1e-10) / 2)
                result[i, 2] = min(result[i, 2], new_r)
                result[j, 2] = min(result[j, 2], new_r)
        # Fix boundary
        result[i, 2] = min(result[i, 2], result[i, 0], 1 - result[i, 0], result[i, 1], 1 - result[i, 1])
        result[i, 2] = max(result[i, 2], 0.0)
    
    return result

# EVOLVE-BLOCK-END
