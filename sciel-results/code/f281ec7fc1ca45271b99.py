"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 120s limit
    
    # Precompute all triples
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    
    def _compute_areas(x):
        """Vectorized triangle area computation."""
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)
    
    def _hull_area(x):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(x)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(x_flat):
        """
        Returns the minimum triangle area divided by convex hull area.
        Higher is better.
        """
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        min_area = np.min(areas)
        return min_area / hull_area
    
    def soft_min_score(x_flat, beta):
        """
        Soft-min version of the score for smoother optimization.
        Uses log-sum-exp trick for numerical stability.
        beta controls sharpness: large beta -> exact min, small beta -> smoother.
        """
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        
        # Soft min using -logsumexp(-areas/beta)/beta
        # This approximates min(areas) when beta is small
        scaled = -areas / beta
        # Shift for numerical stability
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -beta * log_sum_exp
        
        return soft_min / hull_area
    
    def annealed_objective(x_flat, beta):
        """
        Objective for minimization with annealing.
        """
        return -soft_min_score(x_flat, beta)
    
    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        """Generate points on a regular polygon."""
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])
    
    def random_ring(n=N, r_min=0.8, r_max=1.0):
        """Generate random points on a ring."""
        angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        radii = np.random.uniform(r_min, r_max, n)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    def two_ring(n=N, n_inner=5, r_inner=0.5, r_outer=1.0):
        """Generate points on two concentric rings."""
        n_outer = n - n_inner
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + np.pi/n_outer
        x_inner = np.column_stack([r_inner*np.cos(angles_inner), r_inner*np.sin(angles_inner)])
        x_outer = np.column_stack([r_outer*np.cos(angles_outer), r_outer*np.sin(angles_outer)])
        return np.vstack([x_inner, x_outer])
    
    def optimize_with_annealing(x_start, beta_start=1.0, beta_end=0.01, 
                                  n_anneal_steps=20, max_iter_per_step=500):
        """
        Perform optimization with annealing of the soft-min parameter.
        """
        x_curr = x_start.copy()
        beta_curr = beta_start
        
        # Linear annealing schedule
        for step in range(n_anneal_steps):
            if time.time() > deadline - 5:
                break
            
            # Interpolate beta
            t = step / max(n_anneal_steps - 1, 1)
            beta = beta_start * (beta_end / beta_start) ** t
            
            def obj(x_flat):
                return annealed_objective(x_flat, beta)
            
            try:
                res = optimize.minimize(
                    obj,
                    x_curr,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_per_step, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                x_curr = res.x
            except Exception:
                break
        
        return x_curr
    
    def _affine_score(params):
        """
        Score for affine-transformed grid. params = [a, b, c, d] for matrix [[a,b],[c,d]].
        Uses a 4x4 grid minus one corner = 15 points, then we pick 13.
        Actually, let's use a different approach: start with a known good configuration
        and optimize its affine transform.
        """
        # We'll use the regular 13-gon as base and optimize affine transform
        # But that's degenerate (affine transform of regular polygon is just affine polygon)
        # Instead, use a structured 13-point set
        
        # Let's use: 5 points on inner ring, 8 points on outer ring
        # Or better: use a perturbed grid
        
        # Actually, for the affine approach, we need a fixed point set
        # Let's use a 4x4 grid minus 3 corners = 13 points
        # Grid positions: (-1.5,-1.5) to (1.5,1.5) step 1
        # That's 16 points, remove 3 to get 13
        
        # Define base grid (13 points from 4x4 minus 3 corners)
        # Full 4x4 grid at positions {-1.5, -0.5, 0.5, 1.5} x {-1.5, -0.5, 0.5, 1.5}
        # Remove corners: (-1.5,-1.5), (1.5,-1.5), (-1.5,1.5) -> 13 points
        base_points = np.array([
            [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
            [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
            [-1.5, 0.5], [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5],
            [-0.5, 1.5], [0.5, 1.5], [1.5, 1.5]
        ])
        
        # Apply affine transform: x' = A @ x + t (but translation doesn't affect areas or hull area)
        # So we only need the 2x2 matrix A
        a, b, c, d = params
        
        # Apply transform
        transformed = base_points.copy()
        transformed[:, 0] = a * base_points[:, 0] + b * base_points[:, 1]
        transformed[:, 1] = c * base_points[:, 0] + d * base_points[:, 1]
        
        # Check for degenerate transform
        det = a * d - b * c
        if abs(det) < 1e-10:
            return -1.0
        
        x_flat = transformed.flatten()
        return score(x_flat)
    
    def _optimize_affine():
        """Optimize the affine transformation of the grid configuration."""
        # Multiple restarts for the affine optimization
        best_affine_score = -np.inf
        best_affine_params = None
        
        # Initial guesses for [a, b, c, d]
        initial_guesses = [
            [1.0, 0.0, 0.0, 1.0],   # Identity
            [1.0, 0.2, 0.0, 1.0],   # Slight shear
            [0.9, 0.0, 0.0, 1.1],   # Anisotropic scale
            [1.0, 0.3, -0.2, 1.0],  # Shear both ways
            [0.8, 0.1, 0.1, 1.2],   # Mixed
            [1.1, -0.2, 0.2, 0.9],  # Negative shear
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline - 15:
                break
            
            try:
                res = optimize.minimize(
                    lambda p: -_affine_score(p),
                    guess,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                s = -res.fun
                if s > best_affine_score:
                    best_affine_score = s
                    best_affine_params = res.x.copy()
            except Exception:
                pass
        
        return best_affine_score, best_affine_params
    
    def _get_affine_points(params):
        """Get the 13 points from affine-transformed grid."""
        base_points = np.array([
            [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
            [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
            [-1.5, 0.5], [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5],
            [-0.5, 1.5], [0.5, 1.5], [1.5, 1.5]
        ])
        
        a, b, c, d = params
        transformed = base_points.copy()
        transformed[:, 0] = a * base_points[:, 0] + b * base_points[:, 1]
        transformed[:, 1] = c * base_points[:, 0] + d * base_points[:, 1]
        return transformed
    
    best_x = None
    best_score = -np.inf
    
    # Phase 0: Try affine optimization of grid (fast, 4D search)
    try:
        aff_score, aff_params = _optimize_affine()
        if aff_score > best_score and aff_params is not None:
            best_score = aff_score
            best_x = _get_affine_points(aff_params).flatten()
    except Exception:
        pass
    
    # Phase 1: Annealed optimization from regular polygon
    x0 = regular_polygon(N).flatten()
    try:
        x_annealed = optimize_with_annealing(
            x0, beta_start=0.5, beta_end=0.005, 
            n_anneal_steps=15, max_iter_per_step=300
        )
        s = score(x_annealed)
        if s > best_score:
            best_score = s
            best_x = x_annealed.copy()
    except Exception:
        pass
    
    # Phase 2: Direct optimization from the best result
    if time.time() < deadline - 10 and best_x is not None:
        try:
            res = optimize.minimize(
                lambda x: -score(x),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Multiple restarts with annealing
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        # Generate diverse starting points
        if i == 0:
            # Perturbed regular polygon
            x_start = (regular_polygon(N, phase=np.random.uniform(0, 2*np.pi/N)) + 
                       np.random.randn(N, 2) * 0.05).flatten()
        elif i == 1:
            # Two-ring structure
            x_start = two_ring(N, n_inner=5, r_inner=0.4, r_outer=1.0).flatten()
        elif i == 2:
            # Two-ring with different parameters
            x_start = two_ring(N, n_inner=6, r_inner=0.6, r_outer=1.0).flatten()
        elif i == 3:
            # Random ring
            x_start = random_ring(N, r_min=0.85, r_max=1.0).flatten()
        elif i == 4:
            # Perturbed best solution
            if best_x is not None:
                x_start = (best_x.reshape(-1, 2) + np.random.randn(N, 2) * 0.03).flatten()
            else:
                continue
        elif i == 5:
            # Slightly irregular polygon
            angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.randn(N) * 0.1
            radii = 1.0 + np.random.randn(N) * 0.05
            x_start = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]).flatten()
        elif i == 6:
            # Grid-like perturbation
            base = regular_polygon(N)
            # Add some radial perturbation
            r = np.linalg.norm(base, axis=1)
            theta = np.arctan2(base[:, 1], base[:, 0])
            r_pert = r * (1 + 0.1 * np.sin(3*theta + np.random.uniform(0, 2*np.pi)))
            x_start = np.column_stack([r_pert*np.cos(theta), r_pert*np.sin(theta)]).flatten()
        else:
            # Random points in disk
            angles = np.sort(np.random.uniform(0, 2*np.pi, N))
            radii = np.sqrt(np.random.uniform(0.3, 1.0, N))
            x_start = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]).flatten()
        
        # Annealed optimization from this start
        try:
            x_annealed = optimize_with_annealing(
                x_start, beta_start=0.3, beta_end=0.01,
                n_anneal_steps=10, max_iter_per_step=200
            )
            s = score(x_annealed)
            if s > best_score:
                best_score = s
                best_x = x_annealed.copy()
        except Exception:
            pass
        
        # Quick direct refinement if time permits
        if time.time() < deadline - 5 and best_x is not None and s > best_score - 0.01:
            try:
                res = optimize.minimize(
                    lambda x: -score(x),
                    x_annealed,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                s2 = -res.fun
                if s2 > best_score:
                    best_score = s2
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 3 and best_x is not None:
        try:
            res = optimize.minimize(
                lambda x: -score(x),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: regular polygon
        best_x = regular_polygon(N).flatten()
    
    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
