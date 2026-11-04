import numpy as np
import time
from scipy.optimize import minimize, LbfgsB

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds.
    pos_r: (n, 3) array of [x, y, r]
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    # Check radii are positive
    if np.any(rs < 1e-9):
        return False
    
    # Check bounds: circles must be inside [0, w] x [0, h]
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    # Check non-overlap
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _resolve_collisions(circles, w, h, max_iters=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    circles: (n, 3) array [x, y, r]
    Returns modified circles array.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints: push circles inside [0, w] x [0, h]
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        
        # 2. Pairwise collisions: push apart
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        # Random perturbation if exactly overlapping
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart by half the overlap each
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    # Final clamp to ensure boundary validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return np.column_stack((pos, rad))

def _generate_hex_init(w, h, n_circles):
    """
    Generate initial circle positions in a hexagonal pattern.
    """
    circles = np.zeros((n_circles, 3))
    count = 0
    
    # Estimate optimal radius for hex packing in this box
    # Approximate max radius for N circles in area A is ~ sqrt(A/N) * 0.5 / sqrt(2/sqrt(3))
    # But we start small and let optimizer grow them.
    # Let's just place them in a hex grid that fits.
    
    # Try to fit a hex grid
    # Rows of circles.
    # r is the radius.
    # Vertical spacing: sqrt(3)*r
    # Horizontal spacing: 2*r
    # Offset for odd rows: r
    
    # We want to find a scale factor s such that the grid fits.
    # Let's try a range of scales.
    best_scale = 0.0
    best_count = 0
    
    # We just need a valid initial configuration.
    # Let's pick a radius that allows at least N circles.
    # Area of box = w*h. Area of circle ~ pi*r^2.
    # N * pi * r^2 < w*h => r < sqrt(w*h/(N*pi))
    # Hex packing density ~ 0.9.
    # r < sqrt(w*h / (N * pi / 0.9))
    
    r_max = np.sqrt(w * h / (N * np.pi / 0.9)) * 0.8
    
    # Grid generation
    # We will generate a grid with spacing based on r_max
    # And then clamp/relax.
    
    # Let's just use a simple grid but with hex offset for better initial separation.
    # If w > h, more columns.
    cols = int(np.sqrt(n_circles * (w/h)))
    rows = int(np.ceil(n_circles / cols))
    
    # Adjust cols/rows to fit N
    while cols * rows < n_circles:
        if cols < rows:
            cols += 1
        else:
            rows += 1
        if cols > 10 or rows > 10: break # Safety

    cell_w = w / cols
    cell_h = h / rows
    
    # Hex offset
    # We can't easily do hex in a rectangular grid without varying x positions per row.
    # Let's stick to a jittered grid for simplicity and robustness, or a proper hex layout.
    
    # Proper Hex Layout
    # Let's try to fit a hex lattice.
    # Vertical distance dy = sqrt(3) * r
    # Horizontal distance dx = 2 * r
    # Number of rows R = floor(h / dy)
    # Number of cols C = floor(w / dx)
    
    # We want to maximize r such that we can fit N circles.
    # This is complex. Let's just use the grid init but with a random jitter to break symmetry.
    
    # Actually, the current `_generate_grid_init` is okay, but let's add a jitter.
    # Or better, let's implement a simple hex placement.
    
    # Let's just use the grid init but ensure it's well distributed.
    # The current `_generate_grid_init` uses 0.45 * min(cell_w, cell_h).
    # This is safe.
    
    # Let's return a jittered grid to avoid symmetric local minima.
    circles = _generate_grid_init(w, h, n_circles, cols, rows)
    # Add small random jitter to break symmetry
    jitter = 0.01 * np.random.rand(n_circles, 2)
    circles[:, :2] += jitter
    circles[:, :2] = np.clip(circles[:, :2], circles[:, 2], np.array([w, h]) - circles[:, 2])
    
    return circles

def _generate_grid_init(w, h, n_circles, cols, rows):
    """
    Generate initial circle positions in a grid pattern.
    """
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            # Initial radius: half the min cell size, slightly smaller
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
            
    return circles

def _soft_penalty(x, w, h, beta):
    """
    Compute soft penalty for overlaps and boundary violations.
    x: flattened array [x0, y0, r0, x1, y1, r1, ...]
    beta: temperature for log-sum-exp
    """
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    # Use smooth max: log(sum(exp(beta * val))) / beta
    # We want to penalize violations:
    # x - r < 0 -> x - r is negative. We want to penalize max(0, -(x-r)).
    # Let f = -(x-r). If f > 0, it's a violation.
    # Smooth max(0, f) ~ log(1 + exp(beta*f)) / beta
    
    # Left boundary
    f_left = -(xs - rs)
    # To avoid overflow in exp, shift f_left
    max_f_left = np.max(f_left)
    if max_f_left > 0:
        penalty += (1.0/beta) * np.log(np.sum(np.exp(beta * (f_left - max_f_left))) * np.exp(beta * max_f_left))
    # Right boundary
    f_right = -(w - xs - rs)
    max_f_right = np.max(f_right)
    if max_f_right > 0:
        penalty += (1.0/beta) * np.log(np.sum(np.exp(beta * (f_right - max_f_right))) * np.exp(beta * max_f_right))
    # Bottom boundary
    f_bottom = -(ys - rs)
    max_f_bottom = np.max(f_bottom)
    if max_f_bottom > 0:
        penalty += (1.0/beta) * np.log(np.sum(np.exp(beta * (f_bottom - max_f_bottom))) * np.exp(beta * max_f_bottom))
    # Top boundary
    f_top = -(h - ys - rs)
    max_f_top = np.max(f_top)
    if max_f_top > 0:
        penalty += (1.0/beta) * np.log(np.sum(np.exp(beta * (f_top - max_f_top))) * np.exp(beta * max_f_top))
        
    # Overlap penalties
    # dist < r_i + r_j -> r_i + r_j - dist > 0
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            f_overlap = rs[i] + rs[j] - dist
            # Smooth max(0, f_overlap)
            # If f_overlap is very negative, exp(beta*f) is 0.
            # We can just compute log-sum-exp of the positive parts?
            # Actually, log(1 + exp(beta*f)) / beta approximates max(0, f) if f is large positive, and 0 if f is large negative.
            # But for large negative f, exp(beta*f) underflows to 0, log(1+0)=0.
            # For large positive f, exp(beta*f) overflows.
            # Use stable log-sum-exp.
            
            # We only care if there is a violation.
            if f_overlap > -1.0: # Only compute if close to violation
                # Stable soft max
                # max(0, f) approx log(1 + exp(beta*f))/beta
                # If beta*f > 0, exp(beta*f) is large.
                # If beta*f < 0, exp(beta*f) is small.
                
                # Let's use a simpler smooth max:
                # soft_max(a, b) = log(exp(a) + exp(b))
                # We want max(0, f).
                # This is log(exp(0) + exp(f)) = log(1 + exp(f)).
                # With scaling beta: log(1 + exp(beta*f))/beta.
                
                # To prevent overflow:
                # If beta*f > 0: log(exp(beta*f) * (1 + exp(-beta*f))) / beta = beta*f + log(1 + exp(-beta*f))/beta
                # If beta*f < 0: log(1 + exp(beta*f))/beta
                
                bf = beta * f_overlap
                if bf > 0:
                    val = (bf + np.log1p(np.exp(-bf))) / beta
                else:
                    val = np.log1p(np.exp(bf)) / beta
                penalty += val
                
    return penalty

def _soft_penalty_grad(x, w, h, beta):
    """
    Compute gradient of soft penalty.
    """
    n = N
    xs = x[0::3].copy()
    ys = x[1::3].copy()
    rs = x[2::3].copy()
    
    grad = np.zeros_like(x)
    
    # Helper for soft max gradient
    # d/dx log(1 + exp(beta*x))/beta = sigmoid(beta*x)
    def sig(z):
        if z > 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            return ez / (1.0 + ez)
            
    # Boundaries
    # f_left = -(xs - rs) = rs - xs
    # grad_x: -sig(beta*(rs-xs))
    # grad_r: +sig(beta*(rs-xs))
    f_left = rs - xs
    g_left = sig(beta * f_left)
    grad[0::3] -= g_left
    grad[2::3] += g_left
    
    # f_right = -(w - xs - rs) = xs + rs - w
    f_right = xs + rs - w
    g_right = sig(beta * f_right)
    grad[0::3] += g_right
    grad[2::3] += g_right
    
    # f_bottom = -(ys - rs) = rs - ys
    f_bottom = rs - ys
    g_bottom = sig(beta * f_bottom)
    grad[1::3] -= g_bottom
    grad[2::3] += g_bottom
    
    # f_top = -(h - ys - rs) = ys + rs - h
    f_top = ys + rs - h
    g_top = sig(beta * f_top)
    grad[1::3] += g_top
    grad[2::3] += g_top
    
    # Overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 1e-12:
                dist = 1e-6
                nx = 0
                ny = 0
            else:
                dist = np.sqrt(dist_sq)
                nx = dx / dist
                ny = dy / dist
                
            f_overlap = rs[i] + rs[j] - dist
            # Gradient of log(1 + exp(beta*f))/beta wrt f is sig(beta*f)
            g_overlap = sig(beta * f_overlap)
            
            # d f_overlap / d xs[i] = -nx
            # d f_overlap / d xs[j] = +nx
            # d f_overlap / d ys[i] = -ny
            # d f_overlap / d ys[j] = +ny
            # d f_overlap / d rs[i] = 1
            # d f_overlap / d rs[j] = 1
            
            # Apply to gradient
            idx_i_x = i*3
            idx_i_y = i*3+1
            idx_i_r = i*3+2
            idx_j_x = j*3
            idx_j_y = j*3+1
            idx_j_r = j*3+2
            
            grad[idx_i_x] += g_overlap * (-nx)
            grad[idx_i_y] += g_overlap * (-ny)
            grad[idx_i_r] += g_overlap
            
            grad[idx_j_x] += g_overlap * (nx)
            grad[idx_j_y] += g_overlap * (ny)
            grad[idx_j_r] += g_overlap
            
    return grad

def _soft_objective(x, w, h, beta, weight):
    """
    Objective: -sum(r) + weight * penalty
    """
    r = x[2::3]
    pen = _soft_penalty(x, w, h, beta)
    return -np.sum(r) + weight * pen

def _soft_objective_grad(x, w, h, beta, weight):
    """
    Gradient of objective.
    """
    n = N
    grad = np.zeros_like(x)
    # d(-sum(r))/dx = -1 for r components
    grad[2::3] = -1.0
    
    # Add penalty gradient
    p_grad = _soft_penalty_grad(x, w, h, beta)
    grad += weight * p_grad
    
    return grad

def _optimize_soft(x0, w, h, deadline):
    """
    Optimize using L-BFGS-B with soft penalty.
    """
    if time.time() > deadline - 5.0:
        return None

    n = N
    # Bounds for L-BFGS-B
    # x in [r, w-r] is hard to define as bounds because r is variable.
    # We can use loose bounds and rely on penalty.
    # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
    bounds = []
    for i in range(n):
        bounds.append((0.0, w)) # x
        bounds.append((0.0, h)) # y
        bounds.append((1e-4, min(w, h)/2)) # r

    # Multi-stage beta
    betas = [10.0, 50.0, 100.0, 200.0]
    weights = [100.0, 1000.0, 10000.0, 100000.0]
    
    x_current = x0.copy()
    
    for i, beta in enumerate(betas):
        if time.time() > deadline - 10.0:
            break
            
        weight = weights[i]
        
        # Run L-BFGS-B
        try:
            res = minimize(
                _soft_objective,
                x_current,
                jac=_soft_objective_grad,
                args=(w, h, beta, weight),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.success or res.fun < -np.sum(x_current[2::3]) + weight * _soft_penalty(x_current, w, h, beta):
                x_current = res.x
        except Exception:
            pass
            
    return x_current.reshape(-1, 3)

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio.
    1. Generate grid init.
    2. Relax.
    3. Soft optimization.
    4. Final SLSQP polish.
    """
    # Try a few grid configurations
    best_init = None
    best_init_score = 0.0
    
    ar = w / h
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6), (5, 4)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4), (4, 5)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4), (3, 7)]
        
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        # Add jitter
        init_circles[:, :2] += 0.01 * np.random.rand(N, 2)
        init_circles[:, :2] = np.clip(init_circles[:, :2], init_circles[:, 2], np.array([w, h]) - init_circles[:, 2])
        
        # Quick relax
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=100)
        if _validate_solution(relaxed, w, h, N):
            score = np.sum(relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = relaxed

    if best_init is None:
        best_init = _generate_grid_init(w, h, N, 5, 5)
        best_init = _resolve_collisions(best_init, w, h, max_iters=100)
        if not _validate_solution(best_init, w, h, N):
            return None

    # Soft optimization
    if time.time() < deadline - 20.0:
        x0 = best_init.flatten()
        optimized_flat = _optimize_soft(x0, w, h, deadline)
        if optimized_flat is not None:
            optimized = optimized_flat.reshape(-1, 3)
            # Validate
            if not _validate_solution(optimized, w, h, N):
                # Try to fix
                fixed = _resolve_collisions(optimized, w, h, max_iters=500)
                if _validate_solution(fixed, w, h, N):
                    optimized = fixed
                else:
                    # Fall back to best_init if soft opt failed
                    pass
            
            if _validate_solution(optimized, w, h, N):
                # Final Polish with SLSQP
                if time.time() < deadline - 5.0:
                    # Construct SLSQP constraints
                    n = N
                    x0_polish = optimized.flatten()
                    
                    def objective(x):
                        return -np.sum(x[2::3])
                        
                    constraints = []
                    # Boundary
                    for i in range(n):
                        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
                        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
                        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
                        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
                        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})
                        
                    # Overlap
                    for i in range(n):
                        for j in range(i+1, n):
                            constraints.append({
                                'type': 'ineq',
                                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
                            })
                            
                    bounds = []
                    for i in range(n):
                        bounds.append((1e-6, w))
                        bounds.append((1e-6, h))
                        bounds.append((1e-6, min(w, h)/2))
                        
                    try:
                        res = minimize(
                            objective,
                            x0_polish,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 200, 'ftol': 1e-12}
                        )
                        if res.success:
                            new_circles = res.x.reshape(-1, 3)
                            if _validate_solution(new_circles, w, h, N):
                                optimized = new_circles
                    except Exception:
                        pass
                        
                return optimized
    
    # If soft opt not run or failed, return best_init
    if _validate_solution(best_init, w, h, N):
        return best_init
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin for 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3, 0.65, 1.35, 0.6, 1.4]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        # Time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 15:
            continue
            
        try:
            result = _solve_aspect(w, h, deadline)
            if result is not None:
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass

    if best_result is None:
        # Fallback: Simple grid
        w, h = 1.0, 1.0
        positions = []
        radii = []
        cols = 5
        rows = 5
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.9
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])
        
    return best_result
