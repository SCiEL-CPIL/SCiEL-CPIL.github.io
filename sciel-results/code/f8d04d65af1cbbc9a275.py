"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    def pairwise_dists_sq(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        iu = np.triu_indices(N, k=1)
        return d2[iu]

    def exact_score(pts):
        d2 = pairwise_dists_sq(pts)
        min_d2 = np.min(d2)
        max_d2 = np.max(d2)
        if min_d2 < 1e-12:
            return 1e10
        return max_d2 / min_d2

    def soft_objective(params, temp=0.01):
        """
        Soft objective using log-sum-exp for smooth min/max.
        Returns soft_max / soft_min of squared distances.
        """
        pts = params.reshape(N, D)
        d2 = pairwise_dists_sq(pts)
        d2 = np.maximum(d2, 1e-12)
        
        m_max = np.max(d2)
        # Soft max: m + temp * log(sum(exp((d - m)/temp)))
        exp_terms = (d2 - m_max) / temp
        exp_terms = np.clip(exp_terms, -500, 500)
        soft_max = m_max + temp * np.log(np.sum(np.exp(exp_terms)))
        
        m_min = np.min(d2)
        # Soft min: m - temp * log(sum(exp(-(d - m)/temp)))
        exp_terms_min = -(d2 - m_min) / temp
        exp_terms_min = np.clip(exp_terms_min, -500, 500)
        soft_min = m_min - temp * np.log(np.sum(np.exp(exp_terms_min)))
        
        if soft_min < 1e-12:
            return 1e10
        return soft_max / soft_min

    def bubble_relax(pts, n_iter=200, strength=0.05):
        """
        Iteratively adjust points to equalize pairwise distances.
        Push apart pairs closer than median, pull together pairs farther than median.
        """
        pts = pts.copy()
        for _ in range(n_iter):
            d2 = pairwise_dists_sq(pts)
            median_d2 = np.median(d2)
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            forces = np.zeros_like(pts)
            
            for i in range(N):
                for j in range(i+1, N):
                    d_ij2 = d2[np.triu_indices(N, 1)[0][np.triu_indices(N,1)[1] == j] if False else 0]
                    # Simpler: recompute
            # Recompute properly
            iu_i, iu_j = np.triu_indices(N, 1)
            for idx in range(len(iu_i)):
                i, j = iu_i[idx], iu_j[idx]
                d_ij2 = d2[idx]
                if d_ij2 < 1e-12:
                    continue
                d_ij = np.sqrt(d_ij2)
                # Force magnitude: push if too close, pull if too far
                if d_ij2 < median_d2:
                    # Push apart
                    f = strength * (1.0 - d_ij2 / median_d2)
                else:
                    # Pull together
                    f = -strength * 0.5 * (d_ij2 / median_d2 - 1.0)
                
                v = pts[i] - pts[j]
                v = v / (d_ij + 1e-12)
                forces[i] += f * v
                forces[j] -= f * v
            
            pts += forces
        
        # Remove any duplicate points
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                    pts[j] += 1e-4 * np.array([np.cos(i*0.7), np.sin(j*1.3)])
        
        return pts

    def generate_hex_seed():
        """Generate 16 points from hexagonal lattice."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        
        hex_points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                p = i * v1 + j * v2
                d_sq = np.sum(p**2)
                if d_sq <= 4.0 + 1e-9:
                    hex_points.append(p)
        
        hex_points = np.array(hex_points)
        
        if len(hex_points) != 19:
            return hex_points[:16]
        
        # Remove 3 from the outer ring (distance 2) to get 16
        d_sqs = np.sum(hex_points**2, axis=1)
        outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
        
        best_combo = None
        best_score = float('inf')
        from itertools import combinations
        for remove_idx in combinations(outer_indices, 3):
            keep_idx = [i for i in range(19) if i not in remove_idx]
            pts = hex_points[keep_idx]
            score = exact_score(pts)
            if score < best_score:
                best_score = score
                best_combo = pts
        
        return best_combo

    def generate_grid_seed():
        """Generate 4x4 grid."""
        gx = np.array([0, 1, 2, 3], dtype=float)
        gy = np.array([0, 1, 2, 3], dtype=float)
        X, Y = np.meshgrid(gx, gy)
        return np.column_stack([X.flatten(), Y.flatten()])

    def generate_compressed_grid():
        """4x4 grid with corners moved toward center."""
        pts = generate_grid_seed()
        center = np.array([1.5, 1.5])
        for idx in [0, 3, 12, 15]:
            pts[idx] = pts[idx] + 0.25 * (center - pts[idx])
        return pts

    def generate_spiral_seed():
        """Golden angle spiral."""
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        pts = np.zeros((N, 2))
        for i in range(N):
            r = np.sqrt(i + 1)
            theta = i * golden_angle
            pts[i, 0] = r * np.cos(theta)
            pts[i, 1] = r * np.sin(theta)
        return pts

    def generate_concentric_circles():
        """1 center + 5 inner + 10 outer."""
        pts = [np.array([0.0, 0.0])]
        for i in range(5):
            angle = 2 * np.pi * i / 5
            pts.append(np.array([0.5 * np.cos(angle), 0.5 * np.sin(angle)]))
        for i in range(10):
            angle = 2 * np.pi * i / 10
            pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
        return np.array(pts)

    def refine_annealed(init_pts, t0, deadline):
        """
        Refine using annealed soft objective with L-BFGS-B.
        Returns (best_pts, best_score).
        """
        best_pts = init_pts.copy()
        best_score = exact_score(init_pts)
        
        # Annealing schedule
        temps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        current_params = init_pts.flatten()
        
        for temp in temps:
            if time.time() - t0 > deadline - 8:
                break
            
            try:
                res = minimize(
                    lambda p, t=temp: soft_objective(p, t),
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                current_params = res.x
                pts_check = current_params.reshape(N, D)
                score_check = exact_score(pts_check)
                
                if score_check < best_score:
                    best_score = score_check
                    best_pts = pts_check.copy()
                    
            except Exception:
                pass
        
        return best_pts, best_score

    def final_polish(pts, t0, deadline):
        """Final high-precision polish with very low temperature and exact Nelder-Mead."""
        if time.time() - t0 > deadline - 3:
            return pts
        
        try:
            # Very low temperature soft objective
            res = minimize(
                lambda p: soft_objective(p, 0.0005),
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            pts = res.x.reshape(N, D)
        except Exception:
            pass
        
        if time.time() - t0 > deadline - 2:
            return pts
        
        # Exact Nelder-Mead polish
        try:
            res2 = minimize(
                lambda p: exact_score(p.reshape(N, D)),
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'maxfev': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            pts2 = res2.x.reshape(N, D)
            if exact_score(pts2) < exact_score(pts):
                pts = pts2
        except Exception:
            pass
        
        return pts

    # Generate diverse seeds
    seeds = []
    
    hex_seed = generate_hex_seed()
    if hex_seed is not None:
        seeds.append(hex_seed)
    
    seeds.append(generate_grid_seed())
    seeds.append(generate_compressed_grid())
    seeds.append(generate_spiral_seed())
    seeds.append(generate_concentric_circles())
    
    # Random perturbations of grid
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = generate_grid_seed() + rng.randn(N, D) * 0.1
        seeds.append(perturbed)
    
    # Random points
    for _ in range(2):
        random_pts = rng.random((N, 2)) * 2 - 1
        seeds.append(random_pts)
    
    best_pts = None
    best_score = float('inf')
    
    for i, init_pts in enumerate(seeds):
        if time.time() - t0 > deadline - 15:
            break
        
        # Apply bubble relaxation as a pre-processing step
        relaxed_pts = bubble_relax(init_pts, n_iter=100, strength=0.03)
        
        # Annealed refinement
        refined_pts, score = refine_annealed(relaxed_pts, t0, deadline)
        
        if score < best_score:
            best_score = score
            best_pts = refined_pts
    
    # Final polish on the best
    if best_pts is not None:
        best_pts = final_polish(best_pts, t0, deadline)
        final_score = exact_score(best_pts)
        if final_score < best_score:
            best_score = final_score
    
    if best_pts is None:
        best_pts = seeds[0]
    
    # Ensure no duplicate points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += 1e-6 * np.array([1.0, 0.0])
    
    return best_pts

# EVOLVE-BLOCK-END
