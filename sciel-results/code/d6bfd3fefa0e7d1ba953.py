import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    def _raw_score(centers):
        """
        Calculate the maximum radius r such that all circles fit.
        r = min( min_pair_dist/2, min_edge_dist )
        """
        if not np.all(np.isfinite(centers)):
            return 0.0
        centers = centers.reshape(N, 2)
        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms > 1.0 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        min_edge_dist = np.min(1.0 - norms)
        
        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(0.0, r)

    def _soft_min_objective(centers, beta=100.0):
        """
        Objective function for L-BFGS-B: minimize -soft_min(constraints)
        """
        centers = centers.reshape(N, 2)
        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta
        
        return -soft_min_val

    def _soft_min_gradient(centers, beta=100.0):
        """
        Gradient of the soft-min objective.
        """
        centers = centers.reshape(N, 2)
        grad = np.zeros_like(centers)
        
        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Edge gradient contribution
        edge_w = weights[:N]
        grad += edge_w[:, np.newaxis] * (centers / norms[:, np.newaxis])
        
        # Pair gradient contribution
        n_pairs = len(pair_constraints)
        i_idx = iu[0]
        j_idx = iu[1]
        w_pair = weights[N:N+n_pairs]
        
        diff_pairs = centers[i_idx] - centers[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
        
        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_vecs)
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_vecs)
        
        return grad.reshape(-1)

    def _refine(centers, beta=300.0, maxiter=200):
        """
        Local refinement using L-BFGS-B with soft-min objective.
        """
        try:
            result = minimize(
                _soft_min_objective,
                centers.flatten(),
                args=(beta,),
                jac=lambda x, b: _soft_min_gradient(x, b),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            refined = result.x.reshape(N, 2)
            # Ensure points stay inside
            norms = np.sqrt(np.sum(refined**2, axis=1))
            mask = norms > 1.0
            if np.any(mask):
                refined[mask] *= (1.0 - 1e-6) / norms[mask, np.newaxis]
            return refined
        except Exception:
            return centers

    def _simulated_annealing(centers, steps=1500, T_init=0.015, T_final=0.0003):
        """
        Simulated Annealing for global exploration.
        """
        current_centers = centers.copy()
        current_score = _raw_score(current_centers)
        best_centers = current_centers.copy()
        best_score = current_score
        
        iu = np.triu_indices(N, 1)
        
        for i in range(steps):
            if time.time() > deadline - 10.0:
                break
                
            T = T_init * (T_final / T_init) ** (i / steps)
            
            # Move 1-3 random points
            num_moves = np.random.randint(1, 4)
            indices = np.random.choice(N, num_moves, replace=False)
            
            new_centers = current_centers.copy()
            step_size = max(T * 0.5, 0.003)
            moves = np.random.normal(0, step_size, size=(num_moves, 2))
            new_centers[indices] += moves
            
            # Keep inside unit circle
            norms = np.sqrt(np.sum(new_centers**2, axis=1))
            out_mask = norms > 0.9995
            if np.any(out_mask):
                new_centers[out_mask] *= 0.9995 / norms[out_mask, np.newaxis]
                
            new_score = _raw_score(new_centers)
            delta = new_score - current_score
            
            if delta > 0:
                current_centers = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = new_centers.copy()
            else:
                clamped_delta = max(delta, -10.0 * T)
                if np.random.rand() < np.exp(clamped_delta / T):
                    current_centers = new_centers
                    current_score = new_score
        
        return best_centers, best_score

    # --- Generate diverse initializations ---
    
    initializations = []
    
    # 1. Ring-based initializations with various configurations
    def make_rings(radii, counts, offsets=None):
        centers = []
        centers.append([0.0, 0.0])
        start_idx = 1
        for r, c in zip(radii, counts):
            base_offset = offsets[start_idx-1] if offsets else 0.0
            for k in range(c):
                theta = 2 * np.pi * k / c + base_offset
                centers.append([r * np.cos(theta), r * np.sin(theta)])
            start_idx += 1
        return np.array(centers[:N])

    # Standard ring configuration
    r1 = [0.0, 0.24, 0.46, 0.68, 0.88]
    c1 = [1, 6, 12, 18, 13]
    initializations.append(make_rings(r1, c1))
    
    # Perturbed rings
    for seed in range(10):
        np.random.seed(seed)
        perturbed_radii = [r1[0]] + [r + np.random.uniform(-0.02, 0.02) for r in r1[1:]]
        offsets = [0.0] * len(r1)
        for s in range(1, len(r1)):
            offsets[s] = np.random.uniform(-0.15, 0.15)
        initializations.append(make_rings(perturbed_radii, c1, offsets))
    
    # Different ring count distributions
    ring_configs = [
        ([0.0, 0.25, 0.5, 0.75, 0.9], [1, 6, 12, 18, 13]),
        ([0.0, 0.22, 0.44, 0.66, 0.85], [1, 7, 14, 14, 14]),
        ([0.0, 0.2, 0.4, 0.6, 0.8, 0.95], [1, 5, 10, 15, 10, 9]),
        ([0.0, 0.18, 0.38, 0.58, 0.78, 0.92], [1, 6, 11, 16, 10, 6]),
        ([0.0, 0.2, 0.45, 0.7, 0.9], [1, 8, 16, 14, 11]),
    ]
    for radii, counts in ring_configs:
        total = sum(counts)
        if total < N:
            counts[-1] += N - total
        elif total > N:
            counts[-1] -= total - N
            if counts[-1] <= 0:
                continue
        cfg = make_rings(radii, counts)
        if cfg is not None and len(cfg) == N:
            initializations.append(cfg)
    
    # 2. Hexagonal lattice initializations
    def make_hex_lattice(spacing):
        points = []
        for j in range(-15, 16):
            for i in range(-15, 16):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x*x + y*y) <= 1.0 - spacing/2 - 1e-9:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            return None
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]
    
    for spacing in [0.23, 0.24, 0.25, 0.26, 0.27]:
        hex_cfg = make_hex_lattice(spacing)
        if hex_cfg is not None:
            initializations.append(hex_cfg)
    
    # 3. Random initializations
    np.random.seed(42)
    for _ in range(8):
        cfg = np.random.uniform(-0.95, 0.95, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.95
        if np.any(mask):
            cfg[mask] *= 0.95 / norms[mask, np.newaxis]
        initializations.append(cfg)

    # --- Phase 1: Refine all initializations ---
    best_score = 0.0
    best_centers = initializations[0]
    top_solutions = []
    
    for idx, x0 in enumerate(initializations):
        if time.time() > deadline - 80.0:
            break
        
        refined = _refine(x0, beta=200.0, maxiter=150)
        score = _raw_score(refined)
        
        if score > best_score:
            best_score = score
            best_centers = refined.copy()
        
        top_solutions.append((score, refined.copy()))
        top_solutions.sort(key=lambda x: x[0], reverse=True)
        top_solutions = top_solutions[:5]

    # --- Phase 2: Simulated Annealing from top solutions ---
    for sa_round in range(min(4, len(top_solutions))):
        if time.time() > deadline - 30.0:
            break
        
        start_sol = top_solutions[sa_round][1]
        sa_centers, sa_score = _simulated_annealing(
            start_sol, 
            steps=2000, 
            T_init=0.012, 
            T_final=0.0002
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
            top_solutions.append((sa_score, sa_centers.copy()))
            top_solutions.sort(key=lambda x: x[0], reverse=True)
            top_solutions = top_solutions[:5]
        
        # Refine SA result
        if time.time() < deadline - 15.0:
            refined_sa = _refine(best_centers, beta=400.0, maxiter=250)
            score = _raw_score(refined_sa)
            if score > best_score:
                best_score = score
                best_centers = refined_sa.copy()
                top_solutions.append((score, refined_sa.copy()))
                top_solutions.sort(key=lambda x: x[0], reverse=True)
                top_solutions = top_solutions[:5]

    # --- Phase 3: Final high-precision polish ---
    if time.time() < deadline - 10.0:
        final_refined = _refine(best_centers, beta=600.0, maxiter=350)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined

    if time.time() < deadline - 5.0:
        final_refined = _refine(best_centers, beta=1000.0, maxiter=400)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined

    # Ensure valid return
    if not np.all(np.isfinite(best_centers)):
        return initializations[0]
    
    # Final validity check
    norms = np.sqrt(np.sum(best_centers**2, axis=1))
    if np.any(norms > 1.0):
        # Scale down if any point is outside
        max_norm = np.max(norms)
        best_centers *= (1.0 - 1e-6) / max_norm

    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(f"Resulting shape: {centers.shape}")
