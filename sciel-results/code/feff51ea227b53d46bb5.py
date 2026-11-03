"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _normalize(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms

def _covering_radius(X):
    """
    Compute the exact covering radius in degrees.
    The covering radius is the maximum over y on S^2 of min_i arccos(y . x_i).
    This maximum occurs at Voronoi vertices, which are equidistant to >= 3 points.
    For each triple (i,j,k), the Voronoi vertex is parallel to (x_i - x_j) x (x_i - x_k).
    We check both antipodal directions.
    """
    X = _normalize(X)
    n = len(X)
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = X[i] - X[j]
                v2 = X[i] - X[k]
                y = np.cross(v1, v2)
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-10:
                    continue
                y = y / y_norm
                
                for sign in [1.0, -1.0]:
                    yc = sign * y
                    dots = X @ yc
                    max_dot = np.max(dots)
                    # Check that i, j, k are among the nearest points
                    if (abs(dots[i] - max_dot) < 1e-7 and 
                        abs(dots[j] - max_dot) < 1e-7 and 
                        abs(dots[k] - max_dot) < 1e-7):
                        dist = np.arccos(np.clip(max_dot, -1.0, 1.0))
                        if dist > max_dist:
                            max_dist = dist
    
    return np.degrees(max_dist)

def _facet_circumradii(X):
    """
    Compute the circumradius (angular) for each facet of the convex hull.
    For each triple (i,j,k), compute the circumcenter and check if it's a valid
    Voronoi vertex (empty cap). Return the maximum angular distance.
    """
    X = _normalize(X)
    n = len(X)
    max_ang = 0.0
    worst_triple = None
    
    for i, j, k in combinations(range(n), 3):
        # Compute circumcenter of the triangle formed by X[i], X[j], X[k]
        # The circumcenter is parallel to (X[j]-X[i]) x (X[k]-X[i])
        v1 = X[j] - X[i]
        v2 = X[k] - X[i]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            continue
        c = c / c_norm
        
        for sign in [1.0, -1.0]:
            yc = sign * c
            dots = X @ yc
            max_dot = np.max(dots)
            # Check empty cap: i, j, k must be among the nearest
            if (abs(dots[i] - max_dot) < 1e-7 and 
                abs(dots[j] - max_dot) < 1e-7 and 
                abs(dots[k] - max_dot) < 1e-7):
                ang = np.arccos(np.clip(max_dot, -1.0, 1.0))
                if ang > max_ang:
                    max_ang = ang
                    worst_triple = (i, j, k, sign)
    
    return max_ang, worst_triple

def _smooth_surrogate(X, beta=100.0, M=4000):
    """
    Smooth approximation of the covering radius using soft-max over a grid.
    """
    X = _normalize(X)
    
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = Y @ X.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))  # (M,)
    
    max_d = np.max(distances)
    exp_terms = np.exp(beta * (distances - max_d))
    sum_exp = np.sum(exp_terms)
    soft_max = max_d + np.log(sum_exp) / beta
    
    return soft_max

def _smooth_surrogate_grad(X, beta=100.0, M=4000):
    """
    Smooth approximation of the covering radius with analytic gradient.
    Returns (value, gradient) where gradient has shape (N, 3).
    """
    X = _normalize(X)
    n = len(X)
    
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = Y @ X.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))  # (M,)
    
    max_d = np.max(distances)
    exp_terms = np.exp(beta * (distances - max_d))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # (M,)
    soft_max = max_d + np.log(sum_exp) / beta
    
    # Gradient with respect to X
    # d/dX_j soft_max = (1/beta) * sum_m w_m * d(dist_m)/dX_j
    # dist_m = arccos(max_i y_m . x_i)
    # d(dist_m)/dX_j = -1/sqrt(1 - (y_m . x_j)^2) * y_m * (indicator that j is the argmax)
    # But we use soft assignment: weight for each j is proportional to exp(beta * dist_m) * indicator
    
    # For each grid point m, the nearest point is argmax(dots[m])
    # d(dist_m)/dX_j = -y_m / sqrt(1 - dots[m,j]^2) if j is nearest
    # Using soft assignment:
    # grad_j = (1/beta) * sum_m w_m * (-y_m / sqrt(1 - dots[m,j]^2)) * P(j is nearest | m)
    
    # For efficiency, use the hard assignment (argmax) as an approximation
    nearest = np.argmax(dots, axis=1)  # (M,)
    
    grad = np.zeros((n, 3))
    for m in range(M):
        j = nearest[m]
        dot_val = dots[m, j]
        sqrt_term = np.sqrt(max(1.0 - dot_val * dot_val, 1e-12))
        grad[j] += weights[m] * (-Y[m] / sqrt_term) / beta
    
    return soft_max, grad

def _fibonacci_points(n, jitter=0.0):
    """Generate points using Fibonacci spiral with optional jitter."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    if jitter > 0:
        phi += np.random.randn(n) * jitter
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(X)

def _random_points(n):
    """Generate random points on the sphere."""
    X = np.random.randn(n, 3)
    return _normalize(X)

def _icosahedron_based_points(n):
    """Generate points starting from icosahedron vertices and adding more."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    if n <= 12:
        return verts[:n]
    
    # Add more points by subdividing and projecting
    X = verts.copy()
    while len(X) < n:
        # Find the pair with maximum distance and add midpoint
        max_dist = 0
        best_pair = (0, 1)
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                d = np.linalg.norm(X[i] - X[j])
                if d > max_dist:
                    max_dist = d
                    best_pair = (i, j)
        mid = (X[best_pair[0]] + X[best_pair[1]]) / 2
        mid = mid / np.linalg.norm(mid)
        X = np.vstack([X, mid])
    
    return X[:n]

def _spherical_lloyd_iteration(X, iters=30, M=5000):
    """
    Lloyd's algorithm for spherical k-means / covering.
    """
    X = _normalize(X)
    n = len(X)
    
    for _ in range(iters):
        i = np.arange(M) + 0.5
        z = 1.0 - 2.0 * i / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        dots = Y @ X.T  # (M, N)
        nearest = np.argmax(dots, axis=1)  # (M,)
        
        new_X = np.zeros_like(X)
        for k in range(n):
            mask = (nearest == k)
            if np.sum(mask) > 0:
                centroid = np.mean(Y[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_X[k] = centroid / norm
                else:
                    new_X[k] = X[k]
            else:
                new_X[k] = X[k]
        
        X = _normalize(new_X)
    
    return X

def _lbfgs_smooth_minimize(X, maxiter=300, beta=100.0, M=3000, deadline=None):
    """
    Refine using L-BFGS-B with the soft-max covering radius objective and gradient.
    """
    X = _normalize(X)
    
    def objective_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _normalize(X_local)
        return _smooth_surrogate(X_local, beta=beta, M=M)
    
    def objective_grad_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _normalize(X_local)
        val, grad = _smooth_surrogate_grad(X_local, beta=beta, M=M)
        return val, grad.flatten()
    
    try:
        result = minimize(
            objective_grad_flat,
            X.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        
        if deadline is not None and time.time() > deadline:
            return X
        
        X_opt = _normalize(result.x.reshape(N, 3))
        return X_opt
    except Exception:
        return X

def _nelder_mead_optimize(X, maxiter=500, deadline=None):
    """
    Nelder-Mead optimization on the sphere using the smooth surrogate.
    """
    X = _normalize(X)
    n = len(X)
    
    def objective_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _normalize(X_local)
        return _smooth_surrogate(X_local, beta=200.0, M=3000)
    
    try:
        result = minimize(
            objective_flat,
            X.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if deadline is not None and time.time() > deadline:
            return X
        
        X_opt = _normalize(result.x.reshape(N, 3))
        return X_opt
    except Exception:
        return X

def _simulated_annealing(X, start_temp=0.08, end_temp=0.001, n_iter=3000, deadline=None):
    """
    Simulated annealing to escape local minima.
    """
    X = _normalize(X)
    n = len(X)
    best_X = X.copy()
    best_score = _covering_radius(X)
    
    current_temp = start_temp
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point and perturb it
        idx = np.random.randint(n)
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_X = X.copy()
        new_X[idx] = new_X[idx] + current_temp * direction
        new_X = _normalize(new_X)
        
        # Evaluate using fast surrogate
        score = _smooth_surrogate(new_X, beta=50.0, M=1500)
        best_score_surr = _smooth_surrogate(best_X, beta=50.0, M=1500)
        
        if score < best_score_surr or np.random.rand() < np.exp(-(score - best_score_surr) / current_temp):
            X = new_X
        
        # Cool down
        current_temp = start_temp * (end_temp / start_temp) ** (it / n_iter)
        
        # Periodically evaluate exact score
        if it % 300 == 0:
            exact_score = _covering_radius(X)
            if exact_score < best_score:
                best_score = exact_score
                best_X = X.copy()
    
    # Final exact evaluation
    final_score = _covering_radius(X)
    if final_score < best_score:
        best_score = final_score
        best_X = X.copy()
    
    return best_X, best_score

def _high_precision_polish(X, maxiter=500, deadline=None):
    """
    Final high-precision polish using high-beta L-BFGS.
    """
    X = _normalize(X)
    
    def objective_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _normalize(X_local)
        return _smooth_surrogate(X_local, beta=1000.0, M=6000)
    
    def objective_grad_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _normalize(X_local)
        val, grad = _smooth_surrogate_grad(X_local, beta=1000.0, M=6000)
        return val, grad.flatten()
    
    try:
        result = minimize(
            objective_grad_flat,
            X.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
        )
        
        if deadline is not None and time.time() > deadline:
            return X
        
        X_opt = _normalize(result.x.reshape(N, 3))
        return X_opt
    except Exception:
        return X

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    np.random.seed(42)
    
    best_X = None
    best_score = np.inf
    
    # Try multiple diverse initializations
    initializations = [
        _fibonacci_points(N, jitter=0.05),
        _fibonacci_points(N, jitter=0.1),
        _random_points(N),
        _icosahedron_based_points(N),
        _fibonacci_points(N, jitter=0.2),
        _random_points(N),
    ]
    
    for init_idx, X_init in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        X = X_init.copy()
        
        # Phase 1: Lloyd's algorithm
        X = _spherical_lloyd_iteration(X, iters=20, M=4000)
        score = _covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()
        
        # Phase 2: L-BFGS refinement with moderate beta
        if time.time() < deadline - 10:
            X = _lbfgs_smooth_minimize(X, maxiter=200, beta=100.0, M=3000, deadline=deadline - 8)
            score = _covering_radius(X)
            if score < best_score:
                best_score = score
                best_X = X.copy()
        
        # Phase 3: Simulated annealing if time permits
        if time.time() < deadline - 20 and init_idx < 3:
            X_sa, score_sa = _simulated_annealing(
                best_X, 
                start_temp=0.05, 
                end_temp=0.001, 
                n_iter=1500,
                deadline=deadline - 10
            )
            if score_sa < best_score:
                best_score = score_sa
                best_X = X_sa.copy()
    
    # Phase 4: High-precision polish
    if time.time() < deadline - 5:
        X = _high_precision_polish(best_X, maxiter=300, deadline=deadline - 2)
        score = _covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()
    
    # Phase 5: Final Nelder-Mead polish if time permits
    if time.time() < deadline - 3:
        X = _nelder_mead_optimize(best_X, maxiter=200, deadline=deadline - 1)
        score = _covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()
    
    return _normalize(best_X)

# EVOLVE-BLOCK-END
