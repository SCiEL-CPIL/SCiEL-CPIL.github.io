"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

H = np.sqrt(3) / 2.0

def _all_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.array(areas)

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles."""
    return float(np.min(_all_areas(points)))

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > H+tol:
        return False
    if y > 2*H*x + tol:
        return False
    if y > 2*H*(1-x) + tol:
        return False
    return True

def _all_inside(points, tol=1e-6):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p, tol) for p in points)

def _boundary_penalty(points):
    """Penalty for points outside the triangle."""
    pen = 0.0
    for p in points:
        x, y = p[0], p[1]
        if x < 0: pen += 5.0 * abs(x)
        if x > 1: pen += 5.0 * abs(x-1)
        if y < 0: pen += 5.0 * abs(y)
        if y > H: pen += 5.0 * abs(y-H)
        if y > 2*H*x: pen += 5.0 * abs(y - 2*H*x)
        if y > 2*H*(1-x): pen += 5.0 * abs(y - 2*H*(1-x))
    return pen

def _smoothed_min_area(params, beta):
    """
    Smoothed (log-sum-exp) approximation of the minimum triangle area.
    Lower value = better (we minimize negative smoothed min area).
    beta controls sharpness: large beta -> closer to true min.
    """
    points = params.reshape(-1, 2)
    areas = _all_areas(points)
    # Log-sum-exp for min: -logsumexp(-areas/beta) * beta
    # This is a smooth approximation of min(areas)
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smoothed_min = -lse * beta
    pen = _boundary_penalty(points)
    return -smoothed_min + pen

def _smoothed_min_area_grad(params, beta):
    """
    Gradient of smoothed min area objective (negative smoothed min + penalty).
    Uses finite differences on the smoothed objective for reliability.
    """
    points = params.reshape(-1, 2)
    areas = _all_areas(points)
    
    # Compute the effective weights for the log-sum-exp
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp  # weights for each triangle
    
    # Gradient of smoothed_min = -lse*beta w.r.t. each coordinate
    # d(smoothed_min)/d(coord) = -beta * d(-lse)/d(coord) = beta * d(lse)/d(coord)
    # lse = max_scaled + log(sum_exp)
    # d(lse)/d(area_i) = -weights[i] / beta  (from the -areas/beta term)
    # Actually let me just use finite differences for the gradient
    
    eps = 1e-7
    grad = np.zeros_like(params)
    for idx in range(len(params)):
        params_plus = params.copy()
        params_plus[idx] += eps
        f_plus = _smoothed_min_area(params_plus, beta)
        params_minus = params.copy()
        params_minus[idx] -= eps
        f_minus = _smoothed_min_area(params_minus, beta)
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def _gradient_polish(points, deadline, beta_start=0.05, beta_end=0.005, max_iter=200):
    """
    Gradient-based local optimization using smoothed min area.
    Anneals beta from smooth to sharp for precision.
    """
    points = points.copy()
    params = points.flatten()
    current_area = _min_triangle_area(points)
    
    for pass_num in range(4):
        if time.time() > deadline - 3:
            break
        
        # Anneal beta
        beta = beta_start * (beta_end / beta_start) ** (pass_num / 3.0)
        
        try:
            result = minimize(
                _smoothed_min_area,
                params,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2*N),
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_params = result.x
            new_points = new_params.reshape(-1, 2)
            
            # Project back inside triangle
            for i in range(N):
                x, y = new_points[i]
                # Clip to triangle
                if x < 0: x = 0.001
                if x > 1: x = 0.999
                if y < 0: y = 0.001
                if y > H: y = H - 0.001
                if y > 2*H*x: y = 2*H*x - 0.001
                if y > 2*H*(1-x): y = 2*H*(1-x) - 0.001
                new_points[i] = [x, y]
            
            if _all_inside(new_points, tol=1e-4):
                new_area = _min_triangle_area(new_points)
                if new_area > current_area:
                    points = new_points
                    current_area = new_area
        except Exception:
            pass
    
    return points, current_area

def _simulated_annealing(points, deadline, n_iter=5000, T_start=0.05, T_end=0.001):
    """
    Simulated annealing for global exploration in 22D.
    Uses adaptive step size.
    """
    points = points.copy()
    current_area = _min_triangle_area(points)
    best_points = points.copy()
    best_area = current_area
    
    rng = np.random.RandomState(42)
    n_params = 2 * N
    step_size = 0.05
    
    for it in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        # Anneal temperature
        progress = it / max(n_iter - 1, 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Adaptive step size
        step_size = 0.05 * (1.0 - 0.8 * progress)
        
        # Generate perturbation
        delta = rng.randn(n_params) * step_size
        new_params = points.flatten() + delta
        
        # Clip to bounds
        new_params = np.clip(new_params, 0, 1)
        new_points = new_params.reshape(-1, 2)
        
        # Project inside triangle
        for i in range(N):
            x, y = new_points[i]
            if x < 0: x = 0.001
            if x > 1: x = 0.999
            if y < 0: y = 0.001
            if y > H: y = H - 0.001
            if y > 2*H*x: y = 2*H*x - 0.001
            if y > 2*H*(1-x): y = 2*H*(1-x) - 0.001
            new_points[i] = [x, y]
        
        if not _all_inside(new_points, tol=1e-4):
            continue
        
        new_area = _min_triangle_area(new_points)
        
        # Accept if better or with probability exp(-delta/T)
        if new_area >= current_area:
            points = new_points
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_points = new_points.copy()
        else:
            delta_obj = current_area - new_area  # positive
            if delta_obj < T:
                prob = np.exp(-delta_obj / max(T, 1e-10))
                if rng.random() < prob:
                    points = new_points
                    current_area = new_area
    
    return best_points, best_area

def _structured_init(rng):
    """Generate a structured initial configuration."""
    h = H
    cx, cy = 0.5, h/3
    
    # Try different structured patterns
    pattern = rng.randint(3)
    
    if pattern == 0:
        # Hexagonal packing
        dx = 0.22
        dy = dx * np.sqrt(3) / 2
        points = []
        # 2-3-3-2 rows
        for row, (y_off, x_offsets) in enumerate([
            (1.5*dy, [-dx/2, dx/2]),
            (0.5*dy, [-dx, 0, dx]),
            (-0.5*dy, [-dx, 0, dx]),
            (-1.5*dy, [-dx/2, dx/2])
        ]):
            for x_off in x_offsets:
                px = cx + x_off + rng.randn()*0.01
                py = cy + y_off + rng.randn()*0.01
                if _inside_triangle([px, py]):
                    points.append([px, py])
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.02 and v > 0.02:
                px = u + 0.5 * v
                py = h * v
                points.append([px, py])
    
    elif pattern == 1:
        # Ring-based
        points = [[cx, cy]]
        for ring, (r, n_pts) in enumerate([(0.15, 5), (0.28, 5)]):
            angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + rng.uniform(0, 0.3)
            for a in angles:
                px = cx + r*np.cos(a) + rng.randn()*0.005
                py = cy + r*np.sin(a)*0.85 + rng.randn()*0.005
                if _inside_triangle([px, py]):
                    points.append([px, py])
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.02 and v > 0.02:
                px = u + 0.5 * v
                py = h * v
                points.append([px, py])
    
    else:
        # Random with minimum distance
        points = []
        min_dist = 0.15
        attempts = 0
        while len(points) < N and attempts < 500:
            attempts += 1
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.02 and v > 0.02:
                px = u + 0.5 * v
                py = h * v
                p = [px, py]
                if all(np.linalg.norm(np.array(p) - np.array(q)) > min_dist for q in points):
                    points.append(p)
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.02 and v > 0.02:
                px = u + 0.5 * v
                py = h * v
                points.append([px, py])
    
    return np.array(points[:N])

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    rng = np.random.RandomState(7)
    
    best_points = None
    best_min_area = 0.0
    
    # Phase 1: Multiple SA runs from different structured initializations
    for init_idx in range(5):
        if time.time() > deadline - 40:
            break
        
        points = _structured_init(rng)
        polished, area = _gradient_polish(points, deadline, beta_start=0.08, beta_end=0.02, max_iter=100)
        
        if area > best_min_area:
            best_min_area = area
            best_points = polished.copy()
    
    # Phase 2: SA global search from best point
    if best_points is not None and time.time() < deadline - 30:
        sa_points, sa_area = _simulated_annealing(best_points, deadline, n_iter=8000, T_start=0.04, T_end=0.0005)
        if sa_area > best_min_area:
            best_min_area = sa_area
            best_points = sa_points.copy()
    
    # Phase 3: Gradient polish after SA
    if best_points is not None and time.time() < deadline - 15:
        polished, area = _gradient_polish(best_points, deadline, beta_start=0.05, beta_end=0.005, max_iter=300)
        if area > best_min_area:
            best_min_area = area
            best_points = polished.copy()
    
    # Phase 4: Short SA re-explore + final polish
    if best_points is not None and time.time() < deadline - 10:
        sa_points2, sa_area2 = _simulated_annealing(best_points, deadline, n_iter=3000, T_start=0.02, T_end=0.0003)
        if sa_area2 > best_min_area:
            best_min_area = sa_area2
            best_points = sa_points2.copy()
    
    # Final gradient polish
    if best_points is not None and time.time() < deadline - 3:
        polished, area = _gradient_polish(best_points, deadline, beta_start=0.03, beta_end=0.002, max_iter=200)
        if area > best_min_area:
            best_min_area = area
            best_points = polished.copy()
    
    return best_points
# EVOLVE-BLOCK-END
