"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _min_triangle_area_vec(points):
    """Vectorized: compute minimum area among all C(n,3) triangles."""
    n = len(points)
    # points shape (n, 2)
    # For each triple (i,j,k), area = 0.5*|cross(pj-pi, pk-pi)|
    # We'll compute all pairwise differences and cross products
    # More efficient: use the formula with all pairs
    # For n=11, C(11,3)=165, so we can just enumerate
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return min_area
    return min_area

def _min_triangle_area_fast(points):
    """Faster version using numpy for the inner loop."""
    n = len(points)
    min_area = float('inf')
    # Pre-compute all pairwise cross products
    # For points i, j, k: area = 0.5 * |x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)|
    xs = points[:, 0]
    ys = points[:, 1]
    for i in range(n):
        for j in range(i+1, n):
            # For all k > j
            # area(i,j,k) = 0.5 * |xs[i]*(ys[j]-ys[k]) + xs[j]*(ys[k]-ys[i]) + xs[k]*(ys[i]-ys[j])|
            # = 0.5 * |xs[i]*ys[j] - xs[i]*ys[k] + xs[j]*ys[k] - xs[j]*ys[i] + xs[k]*ys[i] - xs[k]*ys[j]|
            # Let's just compute directly
            for k in range(j+1, n):
                area = 0.5 * abs(xs[i]*(ys[j]-ys[k]) + xs[j]*(ys[k]-ys[i]) + xs[k]*(ys[i]-ys[j]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return min_area
    return min_area

def _inside_triangle_vec(points, tol=1e-6):
    """Vectorized check if all points are inside the equilateral triangle."""
    h = np.sqrt(3) / 2
    xs = points[:, 0]
    ys = points[:, 1]
    # y >= 0
    if np.any(ys < -tol):
        return False
    # y <= 2*h*x  (edge from (0,0) to (0.5,h))
    if np.any(ys > 2*h*xs + tol):
        return False
    # y <= 2*h*(1-x)  (edge from (1,0) to (0.5,h))
    if np.any(ys > 2*h*(1-xs) + tol):
        return False
    # x >= 0, x <= 1
    if np.any(xs < -tol) or np.any(xs > 1+tol):
        return False
    return True

def _project_to_triangle(points, tol=1e-4):
    """Project points that are outside the triangle back inside."""
    h = np.sqrt(3) / 2
    result = points.copy()
    for i in range(len(result)):
        x, y = result[i]
        # Check constraints and project
        if y < 0:
            y = 0
        if y > 2*h*x:
            # Project onto line y = 2*h*x
            # Move toward the line
            t = y / (2*h*x) if x > 1e-10 else 0.5
            y = 2*h*x * min(t, 1.0) if x > 1e-10 else 0
        if y > 2*h*(1-x):
            t = y / (2*h*(1-x)) if (1-x) > 1e-10 else 0.5
            y = 2*h*(1-x) * min(t, 1.0) if (1-x) > 1e-10 else 0
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        result[i] = [x, y]
    return result

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    h = np.sqrt(3) / 2
    
    # Best known Heilbronn configuration for 11 points
    # Start with a good initial configuration based on known results
    # The optimal value for 11 points is approximately 0.0397 (normalized by triangle area)
    # Triangle area = sqrt(3)/4 ≈ 0.4330
    
    # Initial configurations to try
    configs = []
    
    # Config 1: Based on symmetric pattern
    # Place points in a structured way
    # Use the known near-optimal arrangement
    # For 11 points, a good starting point is a perturbed grid
    
    # Symmetric hexagonal-like arrangement
    cx, cy = 0.5, h/3
    base_points = np.array([
        [0.5, h/3],           # center
        [0.25, h/6],          # ring 1
        [0.75, h/6],
        [0.5, h*0.75],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h*0.625],
        [0.625, h*0.625],
        [0.5, h*0.125],
        [0.1875, h*0.3125],
        [0.8125, h*0.3125],
    ])
    configs.append(base_points)
    
    # Config 2: Uniform grid
    grid_points = []
    for row in range(3):
        for col in range(4):
            x = 0.15 + 0.7 * col / 3
            y = h * (0.15 + 0.7 * row / 2)
            if _inside_triangle_vec(np.array([[x, y]])):
                grid_points.append([x, y])
    if len(grid_points) >= N:
        configs.append(np.array(grid_points[:N]))
    
    # Config 3: Random with good spread
    rng = np.random.RandomState(42)
    for seed in range(10):
        rng = np.random.RandomState(seed)
        pts = []
        attempts = 0
        while len(pts) < N and attempts < 1000:
            u, v = rng.random(), rng.random()
            if u + v < 0.9:
                px = u + 0.5*v
                py = h*v
                # Check minimum distance to existing points
                if len(pts) == 0 or all(np.linalg.norm([px,py] - p) > 0.08 for p in pts):
                    pts.append([px, py])
            attempts += 1
        if len(pts) == N:
            configs.append(np.array(pts))
    
    # Evaluate initial configs
    best_points = None
    best_min_area = 0.0
    
    for cfg in configs:
        if not _inside_triangle_vec(cfg):
            cfg = _project_to_triangle(cfg)
        ma = _min_triangle_area_fast(cfg)
        if ma > best_min_area:
            best_min_area = ma
            best_points = cfg.copy()
    
    if best_points is None:
        # Fallback
        best_points = np.array([[0.5, h/3]] * N)
        best_min_area = 0.0
    
    # Stage 1: Local optimization using Nelder-Mead on the 22D parameter space
    # Objective: maximize min triangle area = minimize negative min area
    def objective_flat(flat_params):
        pts = flat_params.reshape(N, 2)
        if not _inside_triangle_vec(pts, tol=1e-3):
            # Penalty for being outside
            return 1.0 - _min_triangle_area_fast(pts) * 0.1
        return -_min_triangle_area_fast(pts)
    
    x0 = best_points.flatten()
    
    # Try Nelder-Mead with multiple restarts
    for restart in range(3):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb initial point
            x_init = x0 + np.random.normal(0, 0.02, 2*N)
        
        result = minimize(
            objective_flat, x_init, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        pts = result.fun  # This is -min_area
        # Actually result.fun is the objective value, which is -min_area
        # Let's recompute
        pts = result.x.reshape(N, 2)
        if _inside_triangle_vec(pts, tol=1e-4):
            ma = _min_triangle_area_fast(pts)
            if ma > best_min_area:
                best_min_area = ma
                best_points = pts.copy()
    
    # Stage 2: Simulated annealing with multiple restarts
    if time.time() < deadline - 10:
        current_points = best_points.copy()
        current_min_area = best_min_area
        best_sa_points = current_points.copy()
        best_sa_min_area = current_min_area
        
        T = 0.005
        T_min = 1e-7
        cooling = 0.9998
        max_iter = 30000
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick a random point to move
            idx = np.random.randint(N)
            old_point = current_points[idx].copy()
            
            # Propose a move with adaptive step size
            step = T * 0.8
            new_point = old_point + np.random.normal(0, step, 2)
            
            # Check if inside triangle
            if not _inside_triangle_vec(np.array([new_point]), tol=1e-5):
                # Try a smaller step
                step = T * 0.3
                new_point = old_point + np.random.normal(0, step, 2)
                if not _inside_triangle_vec(np.array([new_point]), tol=1e-5):
                    continue
            
            current_points[idx] = new_point
            new_min_area = _min_triangle_area_fast(current_points)
            
            delta = new_min_area - current_min_area
            
            # Accept if better, or with probability exp(delta/T) if worse
            if delta > 0 or (T > 0 and np.random.random() < np.exp(delta / T)):
                current_min_area = new_min_area
                if new_min_area > best_sa_min_area:
                    best_sa_min_area = new_min_area
                    best_sa_points = current_points.copy()
            else:
                current_points[idx] = old_point
            
            T *= cooling
            if T < T_min:
                break
        
        if best_sa_min_area > best_min_area:
            best_points = best_sa_points
            best_min_area = best_sa_min_area
    
    # Stage 3: Final fine-tuning with Nelder-Mead on the best found configuration
    if time.time() < deadline - 5:
        x0 = best_points.flatten()
        result = minimize(
            objective_flat, x0, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        pts = result.x.reshape(N, 2)
        if _inside_triangle_vec(pts, tol=1e-4):
            ma = _min_triangle_area_fast(pts)
            if ma > best_min_area:
                best_points = pts
                best_min_area = ma
    
    # Final validation and projection
    if not _inside_triangle_vec(best_points, tol=1e-5):
        best_points = _project_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END

