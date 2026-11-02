"""Circle packing in perimeter-4 rectangle: Delaunay-based construction + optimization polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, ConvexHull

N = 21
_TIME_DEADLINE = 140.0  # safety margin before 150s kill


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _force_relax(x, y, r, w, h, max_iter=800, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries (vectorized)."""
    n = N
    dt = dt_init
    iu = np.triu_indices(n, k=1)
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-9:
            break

        f_pairs = overlap * 12.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        # Vectorized accumulation using np.add.at
        np.add.at(fx, iu[0], ux * f_pairs)
        np.add.at(fy, iu[0], uy * f_pairs)
        np.add.at(fx, iu[1], -ux * f_pairs)
        np.add.at(fy, iu[1], -uy * f_pairs)

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = max(0.0001, dt * 0.7)
        elif max_force < 0.05:
            dt = min(0.005, dt * 1.1)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _compute_inradii(x, y, w, h):
    """Compute maximum inscribed circle radius for each point using Delaunay triangulation.
    
    Uses the fact that the Voronoi cell of a point is bounded by the circumcircles
    of adjacent Delaunay triangles. The inradius of the Voronoi cell equals the
    minimum distance from the point to any edge of its Voronoi cell, which can be
    approximated by the minimum circumradius of adjacent triangles (clamped by walls).
    """
    n = N
    # Add boundary points to make the triangulation cover the rectangle
    margin = 0.05
    boundary_pts = np.array([
        [0, 0], [w, 0], [w, h], [0, h],
        [w/2, 0], [w, h/2], [w/2, h], [0, h/2],
        [w/4, 0], [3*w/4, 0], [w, h/4], [w, 3*h/4],
        [3*w/4, h], [w/4, h], [0, 3*h/4], [0, h/4],
    ])
    all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])
    n_total = len(all_pts)
    
    try:
        tri = Delaunay(all_pts)
    except Exception:
        # Fallback: equal radii based on grid spacing
        return np.full(n, min(w, h) / 6.0)
    
    inradii = np.zeros(n)
    for i in range(n):
        # Find triangles containing point i
        # The inradius of the Voronoi cell is bounded by the distance to the
        # closest point on the boundary of the cell. We approximate by checking
        # circumcenters of adjacent triangles.
        min_dist = min(x[i], w - x[i], y[i], h - y[i])
        
        # For each triangle, compute circumcenter and check if point i is a vertex
        for simplex in tri.simplices:
            if i in simplex:
                # Get the 3 points of this triangle
                pts = all_pts[simplex]
                ax, ay = pts[0]
                bx, by = pts[1]
                cx, cy = pts[2]
                
                # Circumcenter
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
                uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
                
                # Distance from point i to circumcenter
                dist = np.sqrt((x[i] - ux)**2 + (y[i] - uy)**2)
                min_dist = min(min_dist, dist)
        
        inradii[i] = min_dist
    
    return inradii


def _delaunay_construction(w0, h0, max_iter=200, dt=0.002):
    """Construct initial packing using Delaunay-based inradius optimization."""
    n = N
    
    # Start with a jittered grid
    rows = 4
    cols = 6
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x[i] = (col + 0.5) * (w0 / cols) + np.random.uniform(-0.1, 0.1) * w0 / cols
        y[i] = (row + 0.5) * (h0 / rows) + np.random.uniform(-0.1, 0.1) * h0 / rows
        x[i] = np.clip(x[i], 0.05, w0 - 0.05)
        y[i] = np.clip(y[i], 0.05, h0 - 0.05)
        r[i] = 0.05
    
    # Iterate: compute inradii, then move points toward their Voronoi cell centers
    for it in range(max_iter):
        # Compute inradii
        inradii = _compute_inradii(x, y, w0, h0)
        r = np.maximum(inradii, 0.01)
        
        # Compute Voronoi cell centers (approximate by averaging circumcenters of adjacent triangles)
        margin = 0.05
        boundary_pts = np.array([
            [0, 0], [w0, 0], [w0, h0], [0, h0],
            [w0/2, 0], [w0, h0/2], [w0/2, h0], [0, h0/2],
        ])
        all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])
        
        try:
            tri = Delaunay(all_pts)
        except Exception:
            break
        
        # For each point, compute weighted average of circumcenters of adjacent triangles
        new_x = x.copy()
        new_y = y.copy()
        
        for i in range(n):
            cx_sum = 0.0
            cy_sum = 0.0
            weight_sum = 0.0
            
            for simplex in tri.simplices:
                if i in simplex:
                    pts = all_pts[simplex]
                    ax, ay = pts[0]
                    bx, by = pts[1]
                    cx, cy = pts[2]
                    
                    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                    if abs(d) < 1e-12:
                        continue
                    ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
                    uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
                    
                    # Weight by inverse distance
                    dist = np.sqrt((x[i] - ux)**2 + (y[i] - uy)**2)
                    weight = 1.0 / max(dist, 0.01)
                    cx_sum += ux * weight
                    cy_sum += uy * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                new_x[i] = cx_sum / weight_sum
                new_y[i] = cy_sum / weight_sum
        
        # Move toward Voronoi centers
        x = x + dt * (new_x - x)
        y = y + dt * (new_y - y)
        x = np.clip(x, 0.05, w0 - 0.05)
        y = np.clip(y, 0.05, h0 - 0.05)
    
    # Final inradius computation
    r = _compute_inradii(x, y, w0, h0)
    
    # Force relax to resolve any overlaps
    x, y = _force_relax(x, y, r, w0, h0, max_iter=300)
    
    if _is_valid(x, y, r, w0, h0):
        return x, y, r
    else:
        # Shrink to fix
        r *= 0.9
        if _is_valid(x, y, r, w0, h0):
            return x, y, r
    return None


def _objective(z, penalty_weight=5000.0):
    """Objective for SLSQP with penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap penalty (smooth: use softplus-like for better gradient)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                             np.sum(bottom_viol**2) + np.sum(top_viol**2))

    # Gentle aspect ratio regularization
    if w < 0.35:
        obj += 500.0 * (0.35 - w) ** 2
    elif w > 1.65:
        obj += 500.0 * (w - 1.65) ** 2

    return obj


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP optimization with smooth objective."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    bounds = [(0.01, 2.0)] * (2 * n) + [(0.01, 1.0)] * n + [(0.25, 1.75)]
    remaining = time_budget - time.time()
    if remaining < 5:
        return None
    
    maxiter = int(min(600, max(150, remaining * 40)))
    
    try:
        res = minimize(_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
        else:
            # Try to repair with progressive shrinking
            for shrink in [0.995, 0.99, 0.98, 0.97, 0.95]:
                r_repair = r_opt * shrink
                if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                    return x_opt, y_opt, r_repair, w_opt, h_opt
    except Exception:
        pass
    return None


def _polish_positions(x, y, r, w, h, time_budget):
    """Polish positions via vectorized force-directed method, then grow radii."""
    n = N
    x = x.copy()
    y = y.copy()
    iu = np.triu_indices(n, k=1)

    max_iter = int(min(2000, time_budget * 100))
    dt = 0.001
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii uniformly
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _epigraph_radius_max(x, y, r, w, h, time_budget):
    """Per-circle epigraph-style radius maximization: for each circle, find the
    maximum radius it can have given all other circles and walls are fixed.
    This decouples radius from position (SOCP-like structure)."""
    n = N
    r = r.copy()
    for i in range(n):
        if time.time() > time_budget:
            break
        # Max radius from walls
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        # Max radius from each other circle: r_i <= d_ij - r_j
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_wall = min(r_wall, d - r[j])
        r_max = max(0.0, r_wall)
        if r_max > r[i] + 1e-8:
            r[i] = r_max

    if _is_valid(x, y, r, w, h):
        return r
    return None


def _expand_individual(x, y, r, w, h, time_budget):
    """Try to expand each circle individually by moving it to free space."""
    n = N
    improved = False
    for i in range(n):
        if time.time() > time_budget:
            break
        # Find max possible radius at current position
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            r_max = min(r_max, r_allowed)
        r_max = max(0.0, r_max)

        if r_max > r[i] + 1e-5:
            r[i] = r_max
            improved = True

    if _is_valid(x, y, r, w, h):
        return x, y, r, improved
    return None


def _smooth_softmin_polish(x, y, r, w, h, time_budget):
    """High-precision polish using a smooth soft-min surrogate for the minimum
    gap constraint. We maximize sum(r) subject to a smooth penalty on the
    minimum clearance, using L-BFGS-B on a smooth objective."""
    n = N
    start = time.time()
    
    def _smooth_obj(params):
        """Smooth objective: -sum(r) + penalty on soft-min of clearances."""
        x_p = params[0:n]
        y_p = params[n:2*n]
        r_p = params[2*n:3*n]
        
        # Wall clearances
        wall_clear = np.concatenate([
            x_p - r_p,
            w - x_p - r_p,
            y_p - r_p,
            h - y_p - r_p
        ])
        
        # Pairwise clearances
        dx = x_p[:, None] - x_p[None, :]
        dy = y_p[:, None] - y_p[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_p[:, None] + r_p[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        pair_clear = d_pairs - r_sum[iu]
        
        # Soft-min of all clearances (log-sum-exp approximation of min)
        all_clear = np.concatenate([wall_clear, pair_clear])
        # Use soft-min: -logsumexp(-clearances / tau) approximates min
        tau = 0.005
        # Clip to avoid overflow
        neg_clear = -all_clear / tau
        neg_clear_clipped = np.clip(neg_clear, -50, 50)
        m = np.max(neg_clear_clipped)
        soft_min_val = -(m + np.log(np.sum(np.exp(neg_clear_clipped - m)))) * tau
        
        obj = -np.sum(r_p) + 200.0 * np.maximum(-soft_min_val, 0.0) ** 2
        return obj
    
    x0 = np.concatenate([x, y, r])
    bounds = [(0.01, w - 0.01) for _ in range(n)] + \
             [(0.01, h - 0.01) for _ in range(n)] + \
             [(0.01, min(w, h) * 0.5) for _ in range(n)]
    
    maxiter = int(min(500, max(100, (time_budget - (time.time() - start)) * 30)))
    
    try:
        res = minimize(_smooth_obj, x0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100})
        x_new = res.x[0:n]
        y_new = res.x[n:2*n]
        r_new = res.x[2*n:3*n]
        
        # Validate and repair
        if _is_valid(x_new, y_new, r_new, w, h):
            return x_new, y_new, r_new
        else:
            for shrink in [0.995, 0.99, 0.98, 0.97]:
                r_repair = r_new * shrink
                if _is_valid(x_new, y_new, r_repair, w, h):
                    return x_new, y_new, r_repair
    except Exception:
        pass
    return None


def _gradient_expansion_polish(x, y, r, w, h, time_budget):
    """Simultaneous position and radius optimization using smooth clearance gradients.
    
    For each constraint (wall or pair), compute the gradient of the clearance
    with respect to positions and radii. Then do a projected gradient ascent
    on sum(r) subject to all clearances >= 0, using a smooth penalty.
    """
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    iu = np.triu_indices(n, k=1)
    
    lr = 0.002
    for it in range(2000):
        if time.time() > time_budget:
            break
        
        # Compute all clearances and their gradients
        # Wall clearances: c = x - r, w - x - r, y - r, h - y - r
        # Pair clearances: c = d - r_i - r_j where d = ||p_i - p_j||
        
        # Objective gradient: d(sum r)/dr_i = 1 for all i
        # Penalty gradient for violated constraints
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        pair_clear = d_pairs - r_sum_pairs
        
        # Wall clearances
        wc_left = x - r
        wc_right = w - x - r
        wc_bottom = y - r
        wc_top = h - y - r
        
        # Compute gradient of penalty
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Pairwise gradient contributions
        overlap = np.maximum(-pair_clear, 0.0)
        if np.any(overlap > 1e-12):
            # d(pair_clear)/dx_i = (x_i - x_j)/d, d/dx_j = (x_j - x_i)/d
            # d(pair_clear)/dr_i = -1, d/dr_j = -1
            ux = dx[iu] / d_pairs
            uy = dy[iu] / d_pairs
            
            # Gradient of -sum(overlap^2) w.r.t. positions
            # d/dx_i of -(-c)^2 = 2*c * d(c)/dx_i when c < 0
            # = 2*(-c) * (1/d) * (x_i - x_j) for i, and 2*(-c) * (1/d) * (x_j - x_i) for j
            # But we want to maximize clearance, so we push in direction of increasing clearance
            # Penalty = sum(overlap^2) where overlap = max(-clear, 0)
            # d(penalty)/dx_i = 2*overlap * d(-clear)/dx_i = -2*overlap * d(clear)/dx_i
            
            # For circle i in pair (i,j): d(clear)/dx_i = (x_i - x_j)/d = ux
            # For circle j: d(clear)/dx_j = (x_j - x_i)/d = -ux
            
            np.add.at(gx, iu[0], 2 * overlap * ux)  # push i away from j
            np.add.at(gy, iu[0], 2 * overlap * uy)
            np.add.at(gx, iu[1], -2 * overlap * ux)  # push j away from i
            np.add.at(gy, iu[1], -2 * overlap * uy)
            np.add.at(gr, iu[0], 2 * overlap)  # shrink r_i
            np.add.at(gr, iu[1], 2 * overlap)  # shrink r_j
        
        # Wall gradient contributions
        wc_left_v = np.maximum(-wc_left, 0.0)
        wc_right_v = np.maximum(-wc_right, 0.0)
        wc_bottom_v = np.maximum(-wc_bottom, 0.0)
        wc_top_v = np.maximum(-wc_top, 0.0)
        
        gx += 2 * wc_left_v   # push right if too close to left wall
        gx -= 2 * wc_right_v  # push left if too close to right wall
        gy += 2 * wc_bottom_v # push up if too close to bottom
        gy -= 2 * wc_top_v    # push down if too close to top
        gr += 2 * (wc_left_v + wc_right_v + wc_bottom_v + wc_top_v)
        
        # Combine: want to increase r (gradient +1) but avoid penalty (gradient -gr_penalty)
        # Net gradient for r: 1 - gr_penalty (scaled)
        penalty_scale = 50.0
        net_gx = gx * penalty_scale
        net_gy = gy * penalty_scale
        net_gr = np.ones(n) - gr * penalty_scale
        
        # Adaptive learning rate
        max_grad = max(np.max(np.abs(net_gx)), np.max(np.abs(net_gy)), np.max(np.abs(net_gr)), 1e-8)
        if max_grad > 10:
            lr = max(0.0005, lr * 0.8)
        elif max_grad < 1:
            lr = min(0.005, lr * 1.05)
        
        x += lr * net_gx
        y += lr * net_gy
        r += lr * net_gr
        
        # Project to feasible region (clip positions, ensure positive radii)
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        r = np.maximum(r, 0.005)
        
        # If all clearances are positive and improving, try to grow all radii
        if np.min(pair_clear) > 1e-6 and np.min(wc_left) > 1e-6 and np.min(wc_right) > 1e-6:
            if np.min(wc_bottom) > 1e-6 and np.min(wc_top) > 1e-6:
                r_trial = r * 1.001
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
    
    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_DEADLINE

    best_value = 0.0
    best_result = None

    # Phase 1: Delaunay-based construction for various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]

    for w0 in aspect_ratios:
        if time.time() > deadline - 35:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Try Delaunay construction
        result = _delaunay_construction(w0, h0, max_iter=100)
        if result is not None:
            x, y, r = result
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

        # Also try grid construction as fallback
        best_grid = None
        best_grid_r = 0
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            cell_w = w0 / cols
            cell_h = h0 / rows
            r_grid = min(cell_w, cell_h) * 0.48
            if r_grid > best_grid_r:
                best_grid_r = r_grid
                best_grid = (rows, cols)

        if best_grid is not None:
            rows, cols = best_grid
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)

            for i in range(N):
                row = i // cols
                col = i % cols
                x[i] = (col + 0.5) * (w0 / cols)
                y[i] = (row + 0.5) * (h0 / rows)
                r[i] = min(w0 / cols, h0 / rows) * 0.45

            x, y = _force_relax(x, y, r, w0, h0, max_iter=300)

            if not _is_valid(x, y, r, w0, h0):
                r *= 0.95
                if not _is_valid(x, y, r, w0, h0):
                    continue

            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

    # Phase 2: SLSQP joint optimization (try multiple times with different starting points)
    if best_result is not None and time.time() < deadline - 40:
        x_best, y_best, r_best, w_best, h_best = best_result
        opt_result = _slsqp_optimize(x_best, y_best, r_best, w_best, h_best, deadline - 20)
        if opt_result is not None:
            x_opt, y_opt, r_opt, w_opt, h_opt = opt_result
            val = float(np.sum(r_opt))
            if val > best_value:
                best_value = val
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)
        
        # Second SLSQP pass from the improved solution
        if best_result is not None and time.time() < deadline - 20:
            x_best, y_best, r_best, w_best, h_best = best_result
            opt_result2 = _slsqp_optimize(x_best, y_best, r_best, w_best, h_best, deadline - 10)
            if opt_result2 is not None:
                x_opt2, y_opt2, r_opt2, w_opt2, h_opt2 = opt_result2
                val2 = float(np.sum(r_opt2))
                if val2 > best_value:
                    best_value = val2
                    best_result = (x_opt2.copy(), y_opt2.copy(), r_opt2.copy(), w_opt2, h_opt2)

    # Phase 3: Vectorized force polish + uniform radius growth
    if best_result is not None and time.time() < deadline - 15:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 8
        polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_best, h_best)

    # Phase 4: Epigraph per-circle radius maximization (iterate multiple times)
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 7
        for _ in range(3):
            if time.time() > remaining:
                break
            r_ep = _epigraph_radius_max(x_best, y_best, r_best, w_best, h_best, remaining)
            if r_ep is not None:
                val = float(np.sum(r_ep))
                if val > best_value:
                    best_value = val
                    r_best = r_ep
                    best_result = (x_best.copy(), y_best.copy(), r_ep.copy(), w_best, h_best)
                # Re-polish positions after radius change
                if time.time() < remaining - 2:
                    polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining - 2)
                    if polished is not None:
                        x_p, y_p, r_p = polished
                        val2 = float(np.sum(r_p))
                        if val2 > best_value:
                            best_value = val2
                            x_best, y_best, r_best = x_p, y_p, r_p
                            best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_best, h_best)
            else:
                break

    # Phase 5: Gradient-based expansion polish (simultaneous pos+radius)
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 5
        grad_result = _gradient_expansion_polish(x_best, y_best, r_best, w_best, h_best, remaining)
        if grad_result is not None:
            x_g, y_g, r_g = grad_result
            val = float(np.sum(r_g))
            if val > best_value:
                best_value = val
                best_result = (x_g, y_g, r_g, w_best, h_best)

    # Phase 6: Smooth soft-min L-BFGS-B polish (high-precision final pass)
    if best_result is not None and time.time() < deadline - 5:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 2
        smooth_result = _smooth_softmin_polish(x_best, y_best, r_best, w_best, h_best, remaining)
        if smooth_result is not None:
            x_s, y_s, r_s = smooth_result
            val = float(np.sum(r_s))
            if val > best_value:
                best_value = val
                best_result = (x_s, y_s, r_s, w_best, h_best)

    # Phase 7: Final individual expansion
    if best_result is not None and time.time() < deadline - 2:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 1
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, remaining)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_best, h_best)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety: ensure strict validity
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
