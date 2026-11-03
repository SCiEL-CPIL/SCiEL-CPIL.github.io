"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _project_to_sphere(X):
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
    X = _project_to_sphere(X)
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

def _soft_max_covering_radius(X, beta=100.0, M=3000):
    """
    Smooth approximation of the covering radius using soft-max over a grid.
    """
    X = _project_to_sphere(X)
    
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

def _lloyd_iteration(X, iters=50):
    """
    Lloyd's algorithm for spherical k-means / covering.
    Each point moves to the centroid of its Voronoi cell (approximated by nearest-neighbor regions).
    """
    X = _project_to_sphere(X)
    n = len(X)
    
    for _ in range(iters):
        # Assign each point on a fine grid to its nearest X point
        M = 5000
        i = np.arange(M) + 0.5
        z = 1.0 - 2.0 * i / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Find nearest X point for each Y
        dots = Y @ X.T  # (M, N)
        nearest = np.argmax(dots, axis=1)  # (M,)
        
        # Compute centroid for each cluster
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
        
        X = _project_to_sphere(new_X)
    
    return X

def _simulated_annealing(X, start_temp=0.1, end_temp=0.001, n_iter=2000, deadline=None):
    """
    Simulated annealing to escape local minima.
    """
    X = _project_to_sphere(X)
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
        new_X = _project_to_sphere(new_X)
        
        # Evaluate
        # Use a fast surrogate for most iterations
        score = _soft_max_covering_radius(new_X, beta=50.0, M=1000)
        best_score_surr = _soft_max_covering_radius(best_X, beta=50.0, M=1000)
        
        if score < best_score_surr or np.random.rand() < np.exp(-(score - best_score_surr) / current_temp):
            X = new_X
        
        # Cool down
        current_temp = start_temp * (end_temp / start_temp) ** (it / n_iter)
        
        # Periodically evaluate exact score
        if it % 200 == 0:
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

def _lbfgs_refine(X, maxiter=300, beta=100.0, M=3000, deadline=None):
    """
    Refine using L-BFGS-B with the soft-max covering radius objective.
    """
    X = _project_to_sphere(X)
    
    def objective_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _project_to_sphere(X_local)
        return _soft_max_covering_radius(X_local, beta=beta, M=M)
    
    try:
        result = minimize(
            objective_flat,
            X.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        
        if deadline is not None and time.time() > deadline:
            return X
        
        X_opt = _project_to_sphere(result.x.reshape(N, 3))
        return X_opt
    except Exception:
        return X

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    np.random.seed(12345)
    
    # Start with jittered Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.randn(N) * 0.1
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = _project_to_sphere(X)
    
    best_X = X.copy()
    best_score = _covering_radius(best_X)
    
    # Phase 1: Lloyd's algorithm to get a good starting point
    X = _lloyd_iteration(X, iters=30)
    score = _covering_radius(X)
    if score < best_score:
        best_score = score
        best_X = X.copy()
    
    # Phase 2: L-BFGS refinement with moderate beta
    if time.time() < deadline - 30:
        X = _lbfgs_refine(X, maxiter=200, beta=50.0, M=2000, deadline=deadline - 20)
        score = _covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()
    
    # Phase 3: Simulated annealing to escape local minima
    if time.time() < deadline - 40:
        X_sa, score_sa = _simulated_annealing(
            best_X, 
            start_temp=0.05, 
            end_temp=0.001, 
            n_iter=1000,
            deadline=deadline - 30
        )
        if score_sa < best_score:
            best_score = score_sa
            best_X = X_sa.copy()
    
    # Phase 4: High-precision L-BFGS polish with high beta
    if time.time() < deadline - 15:
        X = _lbfgs_refine(best_X, maxiter=400, beta=500.0, M=5000, deadline=deadline - 5)
        score = _covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()
    
    # Phase 5: Another round of Lloyd + L-BFGS if time permits
    if time.time() < deadline - 20:
        X = _lloyd_iteration(best_X, iters=20)
        X = _lbfgs_refine(X, maxiter=300, beta=200.0, M=4000, deadline=deadline - 5)
        score = _covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()
    
    # Final: one more high-beta polish
    if time.time() < deadline - 5:
        X = _lbfgs_refine(best_X, maxiter=200, beta=1000.0, M=6000, deadline=deadline - 1)
        score = _covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()
    
    return _project_to_sphere(best_X)

# EVOLVE-BLOCK-END
