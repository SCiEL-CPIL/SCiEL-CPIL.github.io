"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True
    
    def compute_radii_from_centers(centers, margin=0.0):
        """
        Given circle centers, compute maximum non-overlapping radii using
        Delaunay triangulation and boundary distances.
        Returns array of radii.
        """
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)
        
        # Compute boundary distances
        for i in range(n):
            d_left = centers[i, 0]
            d_right = 1.0 - centers[i, 0]
            d_bottom = centers[i, 1]
            d_top = 1.0 - centers[i, 1]
            radii[i] = min(d_left, d_right, d_bottom, d_top)
        
        # Build Delaunay triangulation to find nearest neighbors
        try:
            tri = Delaunay(centers)
            # For each edge in the triangulation, enforce non-overlap
            for simplex in tri.simplices:
                for i_idx in range(3):
                    for j_idx in range(i_idx+1, 3):
                        i = simplex[i_idx]
                        j = simplex[j_idx]
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        # Each circle can have at most half the distance to neighbor
                        max_r_i = dist / 2.0
                        max_r_j = dist / 2.0
                        radii[i] = min(radii[i], max_r_i)
                        radii[j] = min(radii[j], max_r_j)
        except Exception:
            # If Delaunay fails (e.g., collinear points), use brute force
            for i in range(n):
                for j in range(i+1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r = dist / 2.0
                    radii[i] = min(radii[i], max_r)
                    radii[j] = min(radii[j], max_r)
        
        # Apply margin if needed
        if margin > 0:
            radii = np.maximum(0.0, radii - margin)
        
        return radii
    
    def objective_and_gradient(centers_flat):
        """
        Objective: negative sum of radii (to maximize sum of radii).
        Also returns gradient for L-BFGS-B.
        """
        centers = centers_flat.reshape(-1, 2)
        
        # Compute radii
        radii = compute_radii_from_centers(centers)
        
        # Objective: -sum(radii)
        obj = -np.sum(radii)
        
        # Compute gradient using finite differences or analytical
        # For efficiency, use a simple finite difference approach
        grad = np.zeros_like(centers_flat)
        eps = 1e-6
        for k in range(len(centers_flat)):
            center_k = centers_flat.copy()
            center_k[k] += eps
            c_plus = center_k.reshape(-1, 2)
            r_plus = compute_radii_from_centers(c_plus)
            
            center_k = centers_flat.copy()
            center_k[k] -= eps
            c_minus = center_k.reshape(-1, 2)
            r_minus = compute_radii_from_centers(c_minus)
            
            grad[k] = (-np.sum(r_plus) + np.sum(r_minus)) / (2 * eps)
        
        return obj, grad
    
    def objective_only(centers_flat):
        """Just the objective for L-BFGS-B."""
        centers = centers_flat.reshape(-1, 2)
        radii = compute_radii_from_centers(centers)
        return -np.sum(radii)
    
    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])
    
    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers
    
    # Strategy: L-BFGS-B optimization with multiple restarts
    best_result = None
    best_score = np.inf
    
    # Multiple initializations
    inits = []
    
    # Grid initialization
    inits.append(initial_grid_centers())
    
    # Random initializations with different seeds
    for seed in [1, 7, 13, 23, 37, 53, 71, 89]:
        inits.append(random_centers(seed))
    
    # Also try some structured patterns
    # Hexagonal-like pattern
    hex_centers = []
    for i in range(6):
        for j in range(5):
            if len(hex_centers) < N:
                x = 0.1 + i * 0.16 + (0.08 if j % 2 else 0)
                y = 0.1 + j * 0.17
                if x < 0.9 and y < 0.9:
                    hex_centers.append([x, y])
    if len(hex_centers) >= N:
        inits.append(np.array(hex_centers[:N]))
    
    # Optimize each initialization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                objective_only,
                init_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(0.05, 0.95)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                opt_centers = result.x.reshape(-1, 2)
                radii = compute_radii_from_centers(opt_centers)
                
                # Check validity
                x_full = np.column_stack([opt_centers, radii])
                if is_valid(x_full, tol=1e-6):
                    score = -np.sum(radii)
                    if score < best_score:
                        best_score = score
                        best_result = x_full.copy()
        except Exception as e:
            continue
    
    # If no valid result found, fall back to grid
    if best_result is None:
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers)
        best_result = np.column_stack([centers, radii])
    
    # Final refinement with smaller steps
    if best_result is not None and time.time() < deadline - 10:
        x_ref = best_result.copy()
        # Try small perturbations and re-optimize
        rng = np.random.RandomState(123)
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            # Perturb centers slightly
            perturbed = x_ref[:, :2].copy()
            perturbed += rng.normal(0, 0.01, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.05, 0.95)
            
            try:
                result = minimize(
                    objective_only,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0.05, 0.95)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success:
                    opt_centers = result.x.reshape(-1, 2)
                    radii = compute_radii_from_centers(opt_centers)
                    x_try = np.column_stack([opt_centers, radii])
                    
                    if is_valid(x_try, tol=1e-6):
                        score = -np.sum(radii)
                        if score < best_score:
                            best_score = score
                            best_result = x_try.copy()
            except Exception:
                continue
    
    # Ensure final validity
    if best_result is None or not is_valid(best_result, tol=1e-6):
        # Last resort: very conservative grid
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers, margin=0.001)
        best_result = np.column_stack([centers, radii])
    
    return best_result

# EVOLVE-BLOCK-END
