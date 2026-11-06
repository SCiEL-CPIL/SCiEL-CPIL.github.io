"""
Fifty equal circles in a circle - direct structural approach.
Uses a ring-based configuration with known optimal ring sizes,
then refines with gradient-based optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit, 20s buffer)
    
    best_score = 0.0
    best_centers = None
    
    def _score_raw(centers):
        """Compute the actual score (minimum radius) for a set of centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms >= 1.0):
            return 0.0
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_edge_dist = np.min(1.0 - norms)
        
        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(r, 0.0)
    
    def _objective(centers, beta=300.0):
        """Smooth approximation of negative score."""
        centers = centers.reshape(-1, 2)
        
        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta
        
        return -soft_min_val
    
    def _gradient(centers, beta=300.0):
        """Compute gradient of the objective - vectorized."""
        centers = centers.reshape(-1, 2)
        grad = np.zeros_like(centers)
        
        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-15
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-15
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Edge gradient contribution: -weights[i] * beta * (-c_i / |c_i|)
        edge_w = weights[:N] * beta
        grad += edge_w[:, np.newaxis] * (centers / norms[:, np.newaxis])
        
        # Pair gradient contribution (vectorized)
        pair_w = weights[N:] * beta  # shape (n_pairs,)
        i_idx = iu[0]
        j_idx = iu[1]
        
        # Direction vectors: (c_i - c_j) / |c_i - c_j|
        pair_dirs = (centers[i_idx] - centers[j_idx]) / dists[i_idx, j_idx][:, np.newaxis]
        
        # For each pair k with weight w_k:
        #   grad[i_k] += -w_k * dir_k
        #   grad[j_k] += +w_k * dir_k
        # Use np.add.at for correct accumulation with repeated indices
        np.add.at(grad, i_idx, -pair_w[:, np.newaxis] * pair_dirs)
        np.add.at(grad, j_idx, pair_w[:, np.newaxis] * pair_dirs)
        
        return grad.reshape(-1)
    
    # Direct structural approach: use ring-based configuration
    # Known good configuration: 1 center, 6, 12, 18, 13 in rings
    def _make_ring_config(ring_radii, ring_counts, angles_offset=0.0, angle_jitter=0.0):
        centers = []
        
        # Center
        if ring_counts[0] > 0:
            centers.append([0.0, 0.0])
        
        for ring_idx in range(1, len(ring_radii)):
            r = ring_radii[ring_idx]
            count = ring_counts[ring_idx]
            for k in range(count):
                theta = 2 * np.pi * k / count + angles_offset * ring_idx
                if angle_jitter > 0:
                    theta += np.random.uniform(-angle_jitter, angle_jitter)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        return centers
    
    # Try multiple diverse ring configurations
    ring_configs = []
    
    # Base ring configurations with varied radii
    base_radii_sets = [
        [0.0, 0.25, 0.50, 0.75, 0.95],
        [0.0, 0.22, 0.48, 0.72, 0.94],
        [0.0, 0.28, 0.52, 0.74, 0.93],
        [0.0, 0.24, 0.51, 0.76, 0.96],
        [0.0, 0.26, 0.49, 0.73, 0.92],
    ]
    
    for radii_set in base_radii_sets:
        for ang_off in [0.0, 0.15, 0.3, 0.45, 0.6]:
            ring_configs.append(_make_ring_config(radii_set, [1, 6, 12, 18, 13], angles_offset=ang_off))
    
    # Hexagonal lattice-based with varied spacings
    def _hexagonal_config(spacing):
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x*x + y*y)
                if r <= 1.0 - 0.12 - 1e-9:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            return None
        
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]
    
    for spacing in [0.24, 0.25, 0.26, 0.27, 0.28]:
        hex_cfg = _hexagonal_config(spacing)
        if hex_cfg is not None:
            ring_configs.append(hex_cfg)
    
    # Random configurations with different seeds and ranges
    for seed in range(8):
        np.random.seed(42 + seed * 100)
        range_val = 0.8 + 0.05 * seed
        cfg = np.random.uniform(-range_val, range_val, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        if np.any(mask):
            cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        ring_configs.append(cfg)
    
    # Optimize each configuration with adaptive beta strategy
    # Keep top 5 solutions for refinement
    top_solutions = []  # List of (score, centers)
    
    for idx, x0 in enumerate(ring_configs):
        if time.time() > deadline - 30.0:
            break
        
        # Start with moderate beta, then increase if improving
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 25.0:
                break
            
            try:
                result = minimize(
                    _objective,
                    x0.flatten(),
                    args=(beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                
                if score > best_score:
                    best_score = score
                    best_centers = centers
                
                # Add to top solutions if it's good
                if score > 0.12:
                    top_solutions.append((score, centers.copy()))
            except Exception:
                continue
    
    # Refine top solutions with higher precision
    if time.time() < deadline - 20.0:
        # Sort top solutions by score
        top_solutions.sort(key=lambda x: x[0], reverse=True)
        
        for score, centers in top_solutions[:5]:
            if time.time() > deadline - 10.0:
                break
            
            # Perturb and refine
            perturbed = centers + np.random.normal(0, 0.002, size=(N, 2))
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.98
            if np.any(mask):
                perturbed[mask] *= 0.98 / norms[mask, np.newaxis]
            
            try:
                result = minimize(
                    _objective,
                    perturbed.flatten(),
                    args=(1000.0,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
                )
                
                refined_centers = result.x.reshape(N, 2)
                refined_score = _score_raw(refined_centers)
                if refined_score > best_score:
                    best_score = refined_score
                    best_centers = refined_centers
            except Exception:
                continue
    
    # High-precision polish on best solution
    if best_centers is not None and time.time() < deadline - 8.0:
        try:
            result = minimize(
                _objective,
                best_centers.flatten(),
                args=(2000.0,),
                jac=lambda x, b: _gradient(x, b),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 80}
            )
            
            centers = result.x.reshape(N, 2)
            score = _score_raw(centers)
            if score > best_score:
                best_score = score
                best_centers = centers
        except Exception:
            pass
    
    # Final validation and return
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers
    
    # Fallback
    return _make_ring_config([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13])
