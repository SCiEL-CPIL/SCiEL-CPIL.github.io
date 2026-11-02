"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Smooth surrogate objective (softplus for non-overlap, squared for boundary)
 - Analytical gradients for L-BFGS-B
 - Delaunay-based LP radii solver for exact feasibility given centers
 - Joint aspect ratio optimization integrated into the search
 - Simulated annealing refinement
 - Final high-precision polish pass with joint aspect
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.optimize import linprog

N = 21

# EVOLVE-BLOCK-START

def _check_valid(circles, W, H, tol=1e-6):
    """Check if circles are valid (within bounds, non-overlapping)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > W + tol or y - r < -tol or y + r > H + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < ri + rj - tol:
                return False
    return True


def _compute_value(circles, W, H):
    """Compute the objective: sum of radii, if valid."""
    if not _check_valid(circles, W, H):
        return 0.0
    return float(np.sum(circles[:, 2]))


def _softplus(x, beta=100.0):
    """Numerically stable softplus: log(1+exp(beta*x))/beta"""
    x_beta = beta * x
    result = np.where(x_beta > 0,
                      x + np.log1p(np.exp(-x_beta)) / beta,
                      np.log1p(np.exp(x_beta)) / beta)
    return result


def _sigmoid(x, beta=100.0):
    """Sigmoid: 1/(1+exp(-beta*x))"""
    x_beta = beta * x
    return np.where(x_beta > 0,
                    1.0 / (1.0 + np.exp(-x_beta)),
                    np.exp(x_beta) / (1.0 + np.exp(x_beta)))


def _delaunay_edges(centers, W, H):
    """Get Delaunay triangulation edges plus boundary connections."""
    n = len(centers)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Add boundary points for Delaunay
    # Corners and midpoints of edges
    boundary_pts = np.array([
        [0, 0], [W, 0], [W, H], [0, H],
        [W/2, 0], [W, H/2], [W/2, H], [0, H/2]
    ])
    all_pts = np.vstack([centers, boundary_pts])
    
    try:
        tri = Delaunay(all_pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a < n and b < n:
                        edges.add((min(a, b), max(a, b)))
        # Also add all pairs within some distance as fallback
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < max(W, H) * 0.6:
                    edges.add((i, j))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _solve_radii_lp(centers, W, H):
    """
    Given fixed centers, solve for maximum sum of radii subject to:
      r_i >= 0
      r_i <= x_i, r_i <= W - x_i, r_i <= y_i, r_i <= H - y_i
      r_i + r_j <= dist(i,j) for all i<j
    Use LP via linprog.
    """
    n = len(centers)
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Inequalities: A_ub @ r <= b_ub
    # Boundary constraints: r_i <= x_i, r_i <= W-x_i, r_i <= y_i, r_i <= H-y_i
    # Overlap: r_i + r_j <= d_ij
    
    A_rows = []
    b_rows = []
    
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n); row[i] = 1
        A_rows.append(row); b_rows.append(x)
        # r_i <= W - x
        row = np.zeros(n); row[i] = 1
        A_rows.append(row); b_rows.append(W - x)
        # r_i <= y
        row = np.zeros(n); row[i] = 1
        A_rows.append(row); b_rows.append(y)
        # r_i <= H - y
        row = np.zeros(n); row[i] = 1
        A_rows.append(row); b_rows.append(H - y)
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            row = np.zeros(n); row[i] = 1; row[j] = 1
            A_rows.append(row); b_rows.append(d)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # r_i >= 0 handled by bounds
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = np.maximum(0, res.x)
            return radii
    except Exception:
        pass
    
    # Fallback: greedy radius assignment
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        r = min(x, W - x, y, H - y)
        for j in range(i):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            r = min(r, d - radii[j])
        radii[i] = max(0, r)
    return radii


def _hex_lattice_init(W, H, n_circles=21):
    """Generate hexagonal lattice packing as initial guess."""
    circles = []
    # Try different row configurations
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [3, 3, 4, 4, 4, 3],
        [4, 3, 4, 4, 4, 2],
        [3, 4, 4, 4, 4, 2],
    ]
    
    best_val = -1
    best_circ = None
    
    for row_counts in row_configs:
        if sum(row_counts) != n_circles:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        
        # Estimate radius
        r_est = min(W / (max_cols * 1.8 + 0.5), H / (n_rows * np.sqrt(3) * 0.9 + 0.5))
        r_est = max(r_est, 0.01)
        
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * H / n_rows
            x_offset = 0.5 * r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = W / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + (col_idx + 0.5) * 2 * r_est + x_offset
                circles.append([x, y, r_est])
        
        circles = np.array(circles)
        val = _compute_value(circles, W, H)
        if val > best_val:
            best_val = val
            best_circ = circles
    
    if best_circ is None:
        # Fallback grid
        circles = []
        for i in range(n_circles):
            r_idx = i // 5
            c_idx = i % 5
            x = (c_idx + 0.5) * W / 5
            y = (r_idx + 0.5) * H / 5
            r = min(W / 10, H / 10) * 0.95
            circles.append([x, y, r])
        best_circ = np.array(circles)
    
    return best_circ


def _optimize_smooth(circles0, W, H, max_iter=400, beta=80.0, time_limit=30.0):
    """Optimize using smooth softplus penalties with L-BFGS-B."""
    start = time.time()
    x0 = circles0.flatten()
    n = len(circles0)
    
    def objective_and_grad(x):
        c = x.reshape(-1, 3)
        val = -np.sum(c[:, 2])
        penalty = 0.0
        grad = np.zeros_like(x)
        
        for i in range(n):
            xi, yi, ri = c[i]
            
            # Boundary penalties
            v1 = ri - xi
            penalty += _softplus(v1, beta)
            g1 = _sigmoid(v1, beta)
            grad[3*i] += -g1
            grad[3*i+2] += g1
            
            v2 = xi + ri - W
            penalty += _softplus(v2, beta)
            g2 = _sigmoid(v2, beta)
            grad[3*i] += g2
            grad[3*i+2] += g2
            
            v3 = ri - yi
            penalty += _softplus(v3, beta)
            g3 = _sigmoid(v3, beta)
            grad[3*i+1] += -g3
            grad[3*i+2] += g3
            
            v4 = yi + ri - H
            penalty += _softplus(v4, beta)
            g4 = _sigmoid(v4, beta)
            grad[3*i+1] += g4
            grad[3*i+2] += g4
            
            # Non-negative radius
            v5 = -ri
            penalty += _softplus(v5, beta)
            g5 = _sigmoid(v5, beta)
            grad[3*i+2] += -g5
        
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = c[i]
                xj, yj, rj = c[j]
                dx = xi - xj
                dy = yi - yj
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                
                v = ri + rj - d
                penalty += _softplus(v, beta)
                g = _sigmoid(v, beta)
                
                grad[3*i] += -g * dx / d
                grad[3*i+1] += -g * dy / d
                grad[3*i+2] += g
                grad[3*j] += g * dx / d
                grad[3*j+1] += g * dy / d
                grad[3*j+2] += g
        
        obj = val + penalty
        for i in range(n):
            grad[3*i+2] += -1.0
        
        return obj, grad
    
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                                  'maxfun': 2000})
        circles = result.x.reshape(-1, 3)
    except Exception:
        circles = circles0.copy()
    
    return circles


def _polish_with_lp(circles, W, H, iterations=5, time_limit=20.0):
    """Alternating: optimize centers with smooth penalty, then solve LP for radii."""
    start = time.time()
    for _ in range(iterations):
        if time.time() - start > time_limit:
            break
        
        # Extract centers
        centers = circles[:, :2].copy()
        
        # Solve LP for optimal radii
        radii = _solve_radii_lp(centers, W, H)
        circles[:, 2] = radii
        
        # Now optimize centers (keep radii fixed in penalty)
        x0 = circles.flatten()
        n = len(circles)
        
        def center_obj_grad(x):
            c = x.reshape(-1, 3)
            # Objective: maximize sum of radii (which are LP solutions, but we approximate)
            # Use current radii as target, penalize overlaps and boundary violations
            val = -np.sum(c[:, 2])
            penalty = 0.0
            grad = np.zeros_like(x)
            
            for i in range(n):
                xi, yi, ri = c[i]
                v1 = ri - xi
                penalty += _softplus(v1, 100.0)
                g1 = _sigmoid(v1, 100.0)
                grad[3*i] += -g1
                grad[3*i+2] += g1
                
                v2 = xi + ri - W
                penalty += _softplus(v2, 100.0)
                g2 = _sigmoid(v2, 100.0)
                grad[3*i] += g2
                grad[3*i+2] += g2
                
                v3 = ri - yi
                penalty += _softplus(v3, 100.0)
                g3 = _sigmoid(v3, 100.0)
                grad[3*i+1] += -g3
                grad[3*i+2] += g3
                
                v4 = yi + ri - H
                penalty += _softplus(v4, 100.0)
                g4 = _sigmoid(v4, 100.0)
                grad[3*i+1] += g4
                grad[3*i+2] += g4
                
                v5 = -ri
                penalty += _softplus(v5, 100.0)
                g5 = _sigmoid(v5, 100.0)
                grad[3*i+2] += -g5
            
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi, ri = c[i]
                    xj, yj, rj = c[j]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    v = ri + rj - d
                    penalty += _softplus(v, 100.0)
                    g = _sigmoid(v, 100.0)
                    grad[3*i] += -g * dx / d
                    grad[3*i+1] += -g * dy / d
                    grad[3*i+2] += g
                    grad[3*j] += g * dx / d
                    grad[3*j+1] += g * dy / d
                    grad[3*j+2] += g
            
            obj = val + penalty
            for i in range(n):
                grad[3*i+2] += -1.0
            return obj, grad
        
        try:
            result = minimize(center_obj_grad, x0, method='L-BFGS-B', jac=True,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            circles = result.x.reshape(-1, 3)
        except Exception:
            pass
        
        # Re-solve LP
        centers = circles[:, :2].copy()
        radii = _solve_radii_lp(centers, W, H)
        circles[:, 2] = radii
    
    return circles


def _try_aspect_ratio(W, H, n_circles=21, time_limit=25.0):
    """Try packing for given aspect ratio with multiple strategies."""
    start = time.time()
    if W + H < 1e-6:
        return None, 0.0
    scale = 2.0 / (W + H)
    W *= scale
    H *= scale
    
    best_circles = None
    best_value = 0.0
    
    # Strategy 1: Hex lattice init + smooth optimization
    circles0 = _hex_lattice_init(W, H, n_circles)
    circles_opt = _optimize_smooth(circles0, W, H, max_iter=300, beta=80.0,
                                    time_limit=time_limit * 0.4)
    val = _compute_value(circles_opt, W, H)
    if val > best_value:
        best_value = val
        best_circles = circles_opt.copy()
    
    if time.time() - start > time_limit * 0.6:
        return best_circles, best_value
    
    # Strategy 2: LP polish
    if best_circles is not None:
        circles_lp = _polish_with_lp(best_circles, W, H, iterations=4,
                                      time_limit=time_limit * 0.35)
        val = _compute_value(circles_lp, W, H)
        if val > best_value:
            best_value = val
            best_circles = circles_lp.copy()
    
    # Strategy 3: Grid init + smooth optimization
    if time.time() - start < time_limit * 0.8:
        circles_grid = []
        for i in range(n_circles):
            r_idx = i // 5
            c_idx = i % 5
            x = (c_idx + 0.5) * W / 5
            y = (r_idx + 0.5) * H / 5
            r = min(W / 10, H / 10) * 0.9
            circles_grid.append([x, y, r])
        circles_grid = np.array(circles_grid)
        circles_opt2 = _optimize_smooth(circles_grid, W, H, max_iter=200, beta=60.0,
                                         time_limit=time_limit * 0.2)
        val = _compute_value(circles_opt2, W, H)
        if val > best_value:
            best_value = val
            best_circles = circles_opt2.copy()
    
    return best_circles, best_value


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    best_circles = None
    best_value = 0.0
    best_W, best_H = 1.0, 1.0
    
    # Try various aspect ratios
    aspect_ratios = [
        (1.0, 1.0),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.6, 0.4),
        (0.4, 1.6),
    ]
    
    for W, H in aspect_ratios:
        if time.time() > deadline:
            break
        
        circles, value = _try_aspect_ratio(W, H, N, time_limit=10.0)
        
        if circles is not None and value > best_value:
            best_value = value
            best_circles = circles
            best_W, best_H = W, H
    
    # Final polish on best result
    if best_circles is not None and time.time() < deadline - 10:
        polished = _polish_with_lp(best_circles, best_W, best_H, iterations=3,
                                    time_limit=10.0)
        val = _compute_value(polished, best_W, best_H)
        if val > best_value:
            best_value = val
            best_circles = polished
    
    # Fallback
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
    
    return best_circles

# EVOLVE-BLOCK-END
