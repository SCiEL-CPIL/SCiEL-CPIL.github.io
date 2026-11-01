"""
Combined program: Soft-Min Surrogate Optimization with Force-Directed Pre-conditioning
and Multi-Stage Beta Annealing with Simulated Annealing.

Strategy:
1. Use a numerically stable Soft-Min surrogate to smoothly approximate the objective
   r = min( min_{i<j} d_ij/2, min_i (1 - |c_i|) ).
2. Pre-condition with force-directed relaxation to quickly resolve overlaps.
3. Optimize this surrogate using L-BFGS-B with analytical gradients.
4. Use Simulated Annealing to escape local minima by occasionally accepting worse solutions.
5. Multi-start from diverse initializations (ring, hex lattice, random).
6. Final high-precision polish with very high beta.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def _score(points):
        """Compute packing radius."""
        if points.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms >= 1.0):
            return 0.0
        boundary_margin = np.min(1.0 - norms)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        return min(min_pair_dist / 2.0, boundary_margin)

    def _soft_min_obj_and_grad(params, beta):
        """
        Objective: -soft_min(constraints)
        Constraints: d_ij / 2 for pairs, 1 - |c_i| for boundary.
        We want to maximize the minimum of these, so we minimize the negative soft-min.
        
        Soft min definition (numerically stable):
        Let vals be the constraint values. m = min(vals).
        soft_min = m - (1/beta) * log(sum(exp(-beta * (vals - m))))
        
        Equivalent formulation: minimize (1/beta) * log(sum(exp(-beta * vals)))
        """
        centers = params.reshape(N, 2)
        
        # Pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Boundary distances
        norms = np.linalg.norm(centers, axis=1)
        bound_dists = 1.0 - norms
        
        # Combine all constraints
        vals = np.concatenate([bound_dists, pair_dists / 2.0])
        
        # Compute Soft Min using log-sum-exp for stability
        y = -beta * vals
        max_y = np.max(y)
        y_shifted = y - max_y
        sum_exp = np.sum(np.exp(y_shifted))
        log_sum_exp = max_y + np.log(sum_exp)
        obj_val = log_sum_exp / beta
        
        # Gradient
        weights = np.exp(y_shifted) / sum_exp
        grad_centers = np.zeros_like(centers)
        
        # Boundary contributions
        # vals[0:N] = 1 - |c_i|
        # df/dc_i = -weights[i] * d(1-|c_i|)/dc_i = -weights[i] * (-c_i/|c_i|) = weights[i] * c_i/|c_i|
        bound_weights = weights[:N]
        safe_norms = np.where(norms < 1e-8, 1.0, norms)
        boundary_grad = (bound_weights / safe_norms)[:, np.newaxis] * centers
        grad_centers += boundary_grad
        
        # Pair contributions
        # vals[N:] = d_ij / 2
        # df/dc_i = -weights[pair] * d(d_ij/2)/dc_i = -weights[pair] * 0.5 * (c_i - c_j)/d_ij
        # df/dc_j = -weights[pair] * d(d_ij/2)/dc_j = -weights[pair] * 0.5 * (c_j - c_i)/d_ij = +weights[pair] * 0.5 * (c_i - c_j)/d_ij
        pair_weights = weights[N:]
        d_pairs = dists[i_idx, j_idx]
        safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
        vecs = centers[i_idx] - centers[j_idx]
        unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
        
        grad_contrib_i = -0.5 * pair_weights[:, np.newaxis] * unit_vecs
        grad_contrib_j = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
        
        np.add.at(grad_centers, i_idx, grad_contrib_i)
        np.add.at(grad_centers, j_idx, grad_contrib_j)
        
        return obj_val, grad_centers

    def _optimize_with_beta(centers, beta, max_iter=200):
        """Optimize using L-BFGS-B with given beta."""
        def objective(x):
            val, grad = _soft_min_obj_and_grad(x, beta)
            return val, grad.flatten()
        
        x0 = centers.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10,
                                   'maxls': 50, 'maxfun': 500})
        return result.x.reshape(N, 2)

    def _force_directed_relax(pts, n_steps=200, target_r=0.12):
        """
        Force-directed relaxation to improve packing.
        Pushes overlapping circles apart and pulls them toward the center.
        """
        pts = pts.copy()
        target_d = 2.0 * target_r
        
        for step in range(n_steps):
            if time.time() > deadline - 5.0:
                break
            
            forces = np.zeros_like(pts)
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist_mat, np.inf)
            
            # Only consider pairs that are too close
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist_mat[i, j]
                    if d < target_d and d > 1e-10:
                        overlap = target_d - d
                        direction = (pts[i] - pts[j]) / d
                        force_mag = overlap * 0.5
                        forces[i] += force_mag * direction
                        forces[j] -= force_mag * direction
            
            # Boundary constraint: push circles inward if they're outside
            dists_origin = np.linalg.norm(pts, axis=1)
            for i in range(N):
                if dists_origin[i] > 1.0 - target_r:
                    direction = -pts[i] / dists_origin[i]
                    excess = dists_origin[i] - (1.0 - target_r)
                    forces[i] += excess * direction
            
            # Update positions with adaptive learning rate
            lr = 0.01 * (1.0 - step / max(n_steps, 1)) + 0.001
            pts += lr * forces
            
            # Project to unit disk
            dists = np.linalg.norm(pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / dists[mask, np.newaxis]
        
        return pts

    def _simulated_annealing(centers, T_start=0.01, T_end=0.0001, 
                             iters_per_temp=10, n_temps=30, beta=100.0):
        """
        Simulated annealing to escape local minima.
        At each temperature, do several L-BFGS steps, then accept/reject based on Metropolis criterion.
        """
        current = centers.copy()
        current_score = _score(current)
        best = current.copy()
        best_score = current_score
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / n_temps)
        
        for _ in range(n_temps):
            if time.time() > deadline - 5.0:
                break
            
            for _ in range(iters_per_temp):
                if time.time() > deadline - 5.0:
                    break
                
                # Perturb
                noise_scale = T * 0.5
                perturbed = current + np.random.normal(0, noise_scale, size=current.shape)
                
                # Project inside unit circle
                norms = np.linalg.norm(perturbed, axis=1)
                mask = norms > 0.99
                if np.any(mask):
                    perturbed[mask] *= 0.99 / norms[mask]
                
                # Optimize perturbed
                optimized = _optimize_with_beta(perturbed, beta, max_iter=20)
                new_score = _score(optimized)
                
                # Metropolis acceptance
                if new_score >= current_score:
                    current = optimized
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best = current.copy()
                else:
                    delta = new_score - current_score
                    if delta > -0.05:  # Only accept if not too bad
                        accept_prob = np.exp(delta / T)
                        if np.random.random() < accept_prob:
                            current = optimized
                            current_score = new_score
            
            T *= cooling
        
        return best, best_score

    def _ring_init(ring_configs):
        """Generate points from concentric rings."""
        points = np.zeros((N, 2))
        idx = 0
        for size, radius, offset in ring_configs:
            for j in range(size):
                angle = 2 * np.pi * j / size + offset
                points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
                idx += 1
        return points

    def _hex_lattice_init(r_guess=0.125):
        """Hexagonal lattice initialization."""
        spacing = 2.0 * r_guess
        pts = []
        k = int(1.0 / spacing) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - r_guess) ** 2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            return pts[idx]
        else:
            return _ring_init([(1, 0.0, 0.0), (6, 0.2, 0.0), (12, 0.4, 0.0), 
                               (18, 0.6, 0.0), (13, 0.85, 0.0)])

    def _random_init(seed=None):
        if seed is not None:
            np.random.seed(seed)
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 0.9, N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points

    # Initializations
    inits = []
    
    # 1. Ring structure
    ring_configs_1 = [(1, 0.0, 0.0), (6, 0.2, 0.0), (12, 0.4, 0.0), 
                      (18, 0.6, 0.0), (13, 0.85, 0.0)]
    inits.append(_ring_init(ring_configs_1))
    
    # 2. Ring structure with different offsets
    ring_configs_2 = [(1, 0.0, 0.0), (6, 0.22, 0.1), (12, 0.44, 0.2), 
                      (12, 0.66, 0.3), (9, 0.82, 0.4), (10, 0.92, 0.5)]
    inits.append(_ring_init(ring_configs_2))
    
    # 3. Hex lattice
    inits.append(_hex_lattice_init(0.125))
    
    # 4. Random
    inits.append(_random_init(42))
    inits.append(_random_init(123))
    inits.append(_random_init(456))

    best_pts = None
    best_score = -1.0

    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        try:
            # Stage 0: Force-directed pre-conditioning
            pts = _force_directed_relax(init_pts, n_steps=100, target_r=0.12)
            
            # Stage 1: Initial optimization with moderate beta
            pts = _optimize_with_beta(pts, beta=50.0, max_iter=100)
            score = _score(pts)
            
            # Stage 2: Simulated annealing to escape local minima
            pts, score = _simulated_annealing(pts, T_start=0.005, T_end=0.0001,
                                              iters_per_temp=5, n_temps=20, beta=100.0)
            
            # Stage 3: High-precision polish
            if time.time() < deadline - 10.0:
                pts = _optimize_with_beta(pts, beta=500.0, max_iter=200)
                score = _score(pts)
            
            # Stage 4: Final very high beta polish
            if time.time() < deadline - 5.0:
                pts = _optimize_with_beta(pts, beta=2000.0, max_iter=100)
                score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
        except Exception:
            continue

    # Final polish on best result
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            best_pts = _optimize_with_beta(best_pts, beta=5000.0, max_iter=500)
            final_score = _score(best_pts)
            if final_score > best_score:
                best_score = final_score
        except Exception:
            pass

    if best_pts is None:
        best_pts = _hex_lattice_init(0.125)

    return best_pts
