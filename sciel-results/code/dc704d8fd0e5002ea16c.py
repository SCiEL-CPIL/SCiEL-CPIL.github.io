import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def score_points(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d) ** 2

    def prepare_params(pts):
        """
        Reduces dimensionality by fixing the first two points (closest pair)
        to (0,0) and (1,0).
        Returns (params, permutation) or (None, None) if invalid.
        """
        d = pdist(pts, 'euclidean')
        min_idx = np.argmin(d)
        iu = np.triu_indices(N, 1)
        i_min, j_min = iu[0][min_idx], iu[1][min_idx]
        
        p0 = pts[i_min]
        p1 = pts[j_min]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return None, None

        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so i_min is 0 and j_min is 1
        perm = list(range(N))
        perm[0] = i_min
        perm[1] = j_min
        remaining = [i for i in range(N) if i != i_min and i != j_min]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # transformed[0] should be [0,0]
        # transformed[1] should be [1,0]
        params = transformed[2:].flatten()
        return params, perm

    def objective(params):
        """
        Objective function for optimization.
        Reconstructs points from params (p0=0, p1=1, rest=params) and computes score.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        return score_points(pts)

    def soft_objective(params, beta=50.0):
        """
        Smooth surrogate for the objective function to allow gradient-based optimization
        of the min/max terms.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        return soft_max_d_sq / soft_min_d_sq

    def epigraph_objective(params, t):
        """
        Epigraph formulation: minimize t subject to all pairwise distances in [1/t, t].
        This creates a smooth constrained problem that can be solved with SLSQP.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-8:
            return 1e10
            
        ratio_sq = max_d_sq / min_d_sq
        return t + 0.1 * (ratio_sq - t**2) if ratio_sq > t**2 else t

    def generate_hex_seed():
        """
        Generate a hexagonal lattice subset of 16 points.
        This is a strong geometric prior for 2D packing.
        """
        R = 5
        hex_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        
        # Select 16 points closest to the origin to form a compact cluster
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        return hex_pts[idx]

    def generate_bubble_relax_seed():
        """
        Generate a seed using bubble relaxation: start with random points,
        then iteratively push overlapping points apart to minimize sum of
        squared overlap penalties.
        """
        rng = np.random.default_rng(999)
        pts = rng.uniform(-1, 1, size=(N, 2))
        
        # Bubble relaxation: minimize sum of max(0, 1 - dist_ij)^2
        for iteration in range(200):
            if time.time() > deadline - 20:
                break
            forces = np.zeros((N, 2))
            for i in range(N):
                for j in range(i+1, N):
                    d = pts[j] - pts[i]
                    dist = np.linalg.norm(d)
                    if dist < 1.0 and dist > 1e-8:
                        # Force pushing apart
                        force_mag = (1.0 - dist)
                        forces[i] -= force_mag * d / dist
                        forces[j] += force_mag * d / dist
            pts += 0.1 * forces
        return pts

    def generate_sqrtradial_seed():
        """
        Generate a square-root radial distribution:
        Points placed at radii proportional to sqrt(index) in a grid-like pattern.
        This mitigates corner-to-center distance disparity.
        """
        # Create a 4x4 grid of indices
        grid_idx = np.arange(16).reshape(4, 4)
        
        # Square root scaling for radial distribution
        sqrt_idx = np.sqrt(grid_idx + 1)  # +1 to avoid zero
        
        # Normalize to unit square
        sqrt_norm = sqrt_idx / sqrt_idx.max()
        
        # Map to polar-like coordinates with radial bias
        pts = []
        for i in range(4):
            for j in range(4):
                r = sqrt_norm[i, j]
                theta = (i + 0.5) * np.pi / 4 + (j + 0.5) * np.pi / 8
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pts.append([x, y])
        
        return np.array(pts)

    def generate_circular_seed():
        """
        Generate points on concentric circles with alternating angles.
        """
        pts = []
        # Inner ring: 4 points
        for k in range(4):
            theta = k * np.pi / 2
            pts.append([0.3 * np.cos(theta), 0.3 * np.sin(theta)])
        
        # Middle ring: 6 points
        for k in range(6):
            theta = k * 2 * np.pi / 6 + np.pi / 6
            pts.append([0.7 * np.cos(theta), 0.7 * np.sin(theta)])
        
        # Outer ring: 6 points
        for k in range(6):
            theta = k * 2 * np.pi / 6
            pts.append([1.2 * np.cos(theta), 1.2 * np.sin(theta)])
        
        return np.array(pts)

    def generate_other_seeds():
        """Generate alternative initial configurations."""
        seeds = []
        
        # Square Grid 4x4
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        # Scale to have reasonable spacing
        grid = grid * 0.8
        seeds.append(grid)
        
        # Two concentric rings
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_inner = 0.5
        r_outer = 1.5
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        ring_pts = np.vstack([inner_pts, outer_pts])
        seeds.append(ring_pts)
        
        # Random Uniform
        rng = np.random.default_rng(42)
        rand_pts = rng.uniform(-1, 1, size=(16, 2))
        seeds.append(rand_pts)
        
        # Bubble relaxation seed
        bubble_seed = generate_bubble_relax_seed()
        seeds.append(bubble_seed)
        
        # Three concentric rings (4-6-6)
        pts3 = []
        for k in range(4):
            theta = k * np.pi / 2 + np.pi/4
            pts3.append([0.4 * np.cos(theta), 0.4 * np.sin(theta)])
        for k in range(6):
            theta = k * 2 * np.pi / 6
            pts3.append([0.9 * np.cos(theta), 0.9 * np.sin(theta)])
        for k in range(6):
            theta = k * 2 * np.pi / 6 + np.pi/6
            pts3.append([1.4 * np.cos(theta), 1.4 * np.sin(theta)])
        seeds.append(np.array(pts3))
        
        return seeds

    best_score = 1e10
    best_params = None
    best_perm = None

    # 1. Primary Seeds
    hex_seed = generate_hex_seed()
    sqrtradial_seed = generate_sqrtradial_seed()
    circular_seed = generate_circular_seed()
    
    # 2. Secondary Seeds
    other_seeds = generate_other_seeds()
    
    all_seeds = [hex_seed, sqrtradial_seed, circular_seed] + other_seeds
    
    # Add perturbed versions of key seeds to escape potential local minima
    rng = np.random.default_rng(123)
    for seed in [hex_seed, sqrtradial_seed, circular_seed]:
        for scale in [0.02, 0.05, 0.10]:
            perturbed = seed + rng.normal(0, scale, size=(16, 2))
            all_seeds.append(perturbed)

    def try_optimize(seed_pts, tag=""):
        """Try to optimize from a given seed point configuration."""
        nonlocal best_score, best_params, best_perm
        
        if time.time() > deadline - 15:
            return
            
        params, perm = prepare_params(seed_pts)
        if params is None:
            return
            
        # Initial score
        curr_score = objective(params)
        if curr_score < best_score:
            best_score = curr_score
            best_params = params.copy()
            best_perm = perm.copy()
            
        # Optimization Phase 1: Powell (Derivative-free, good for non-smooth start)
        try:
            res = minimize(objective, params, method='Powell', 
                           options={'maxiter': 200, 'maxfev': 2000, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                best_perm = perm.copy()
                
        except Exception:
            pass

        # Optimization Phase 2: L-BFGS-B on Soft Objective (Gradient-based polish)
        if time.time() < deadline - 10:
            try:
                # Increase beta for sharper approximation to exact min/max
                res2 = minimize(soft_objective, res.x, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
                
                # Check exact score of the result
                exact_score = objective(res2.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_params = res2.x.copy()
                    best_perm = perm.copy()
                    
            except Exception:
                pass
        
        # Optimization Phase 3: SLSQP with epigraph formulation
        if time.time() < deadline - 8:
            try:
                # Initialize t with current best ratio
                t_init = np.sqrt(best_score) if best_score < 1e9 else 2.0
                
                def epigraph_full(params_t):
                    p = params_t[:-1]
                    t = params_t[-1]
                    return epigraph_objective(p, t)
                
                # Constraints: t >= 1 (ratio >= 1)
                constraints = [{'type': 'ineq', 'fun': lambda x: x[-1] - 1.0}]
                
                bounds = [(None, None)] * (N-2)*2 + [(1.0, 10.0)]
                
                res3 = minimize(epigraph_full, np.append(best_params, t_init), 
                               method='SLSQP', bounds=bounds, constraints=constraints,
                               options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
                
                if res3.fun < best_score:
                    best_score = res3.fun
                    best_params = res3.x[:-1].copy()
                    best_perm = perm.copy()
                    
            except Exception:
                pass
        
        # Optimization Phase 4: Local search with small perturbations
        if time.time() < deadline - 5:
            try:
                local_rng = np.random.default_rng(hash(tag) % 2**32)
                for _ in range(3):
                    perturb_scale = 0.005
                    perturbed_params = best_params + local_rng.normal(0, perturb_scale, size=best_params.shape)
                    
                    res4 = minimize(soft_objective, perturbed_params, method='L-BFGS-B',
                                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False})
                    
                    exact_score = objective(res4.x)
                    if exact_score < best_score:
                        best_score = exact_score
                        best_params = res4.x.copy()
                        best_perm = perm.copy()
                        
            except Exception:
                pass

    # Process all seeds
    for idx, seed_pts in enumerate(all_seeds):
        try_optimize(seed_pts, tag=f"seed_{idx}")

    # Final Polish: If we have time, do one more very tight L-BFGS-B pass on the best found
    if best_params is not None and time.time() < deadline - 3:
        try:
            # Use a very high beta for near-exact gradient
            def final_soft_objective(params, beta=200.0):
                return soft_objective(params, beta)
                
            res3 = minimize(final_soft_objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False})
            
            exact_score = objective(res3.x)
            if exact_score < best_score:
                best_score = exact_score
                best_params = res3.x.copy()
                best_perm = best_perm.copy()
        except Exception:
            pass
        
        # Second final polish with even tighter settings
        if best_params is not None and time.time() < deadline - 1:
            try:
                def ultra_soft_objective(params, beta=500.0):
                    return soft_objective(params, beta)
                
                res4 = minimize(ultra_soft_objective, best_params, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False})
                
                exact_score = objective(res4.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_params = res4.x.copy()
                    best_perm = best_perm.copy()
            except Exception:
                pass

    # Reconstruct final points from best_params and best_perm
    if best_params is not None:
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = best_params.reshape(-1, 2)
        
        # Reverse permutation
        inv_perm = np.argsort(best_perm)
        final_pts = pts[inv_perm]
        
        # Center the points for numerical stability (optional, but good practice)
        final_pts -= final_pts.mean(axis=0)
        
        return final_pts
    else:
        # Fallback: return hex seed centered
        pts = generate_hex_seed()
        pts -= pts.mean(axis=0)
        return pts
