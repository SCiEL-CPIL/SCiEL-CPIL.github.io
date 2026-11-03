import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import least_squares

N = 300

# Precompute pair indices at module level for speed
_i_idx, _j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
_NUM_PAIRS = len(_i_idx)

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill

    i_idx = _i_idx
    j_idx = _j_idx
    num_pairs = _NUM_PAIRS

    def compute_score(pts):
        """Compute the packing radius score using squared distances for speed."""
        x = pts[:, 0]
        y = pts[:, 1]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dists_sq = dx**2 + dy**2
        np.fill_diagonal(dists_sq, np.inf)
        
        rad_sq = x**2 + y**2
        rad = np.sqrt(rad_sq)
        bound_dists = 1.0 - rad
        
        pair_dists_sq = dists_sq[i_idx, j_idx]
        min_pair_dists_sq = np.min(pair_dists_sq)
        min_pair_dists = np.sqrt(min_pair_dists_sq) / 2.0
        
        min_bound = np.min(bound_dists)
        
        return min(min_pair_dists, min_bound)

    def compute_all_dists(pts):
        """Compute all pairwise distances and boundary distances."""
        x = pts[:, 0]
        y = pts[:, 1]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dists_sq = dx**2 + dy**2
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        rad = np.sqrt(x**2 + y**2)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        return dists, rad, bound_dists, all_vals

    def soft_min_weights(vals, k, current_min):
        """Compute soft-min weights using log-sum-exp for stability."""
        shifted = vals - current_min
        mask_close = shifted < 50.0
        if not np.any(mask_close):
            return np.ones_like(vals) / len(vals)
        close_shifted = shifted[mask_close]
        max_c = np.max(close_shifted)
        if max_c > 0:
            close_shifted = close_shifted - max_c
        exp_vals = np.exp(-k * close_shifted)
        weights = np.zeros_like(vals)
        weights[mask_close] = exp_vals
        sum_w = np.sum(weights)
        if sum_w > 1e-15:
            weights /= sum_w
        else:
            weights = np.ones_like(vals) / len(vals)
        return weights

    def compute_grads(pts, weights, dists, rad, bound_dists):
        """Compute gradient of soft-min objective."""
        grads = np.zeros_like(pts)
        
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-10
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            idx_m = mask_nonzero
            diff_m = pts[i_idx[idx_m]] - pts[j_idx[idx_m]]
            vecs[idx_m] = (diff_m / d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads

    def spectral_regularization_grad(pts, weight=1.0):
        """
        Compute gradient of the variance of Delaunay edge lengths.
        Minimizing this variance promotes a more uniform lattice structure.
        We maximize uniformity, so we return -gradient of variance.
        """
        if weight < 1e-6:
            return np.zeros_like(pts)
        try:
            tri = Delaunay(pts)
            edges = tri.edges
            if len(edges) == 0:
                return np.zeros_like(pts)
            
            p1 = pts[edges[:, 0]]
            p2 = pts[edges[:, 1]]
            diffs = p1 - p2
            lengths = np.linalg.norm(diffs, axis=1)
            
            mean_len = np.mean(lengths)
            # We want to minimize variance, so we push points to equalize lengths.
            # grad_var = sum(2*(L_i - mean) / N * dL_i/dx)
            # We want to move in direction of -grad_var to minimize variance.
            # So return -grad_var * weight
            
            safe_lengths = np.where(lengths > 1e-9, lengths, 1.0)
            edge_coeffs = 2.0 * (lengths - mean_len) / len(lengths)
            
            forces = (edge_coeffs / safe_lengths)[:, np.newaxis] * diffs
            
            grad_var = np.zeros_like(pts)
            np.add.at(grad_var, edges[:, 0], forces)
            np.add.at(grad_var, edges[:, 1], -forces)
            
            return -weight * grad_var
        except Exception:
            return np.zeros_like(pts)

    def optimize(pts, alpha_init, k_init, k_end, cooling, time_budget,
                 stall_threshold=150, noise_scale_base=0.01, use_spectral=True, spectral_weight_init=0.5):
        """Main optimization loop using soft-min gradient ascent with spectral regularization."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        end_time = start_time + time_budget
        spectral_weight = spectral_weight_init
        
        while time.time() < end_time:
            dists, rad, bound_dists, all_vals = compute_all_dists(pts)
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            weights = soft_min_weights(all_vals, k, current_min)
            
            grads = compute_grads(pts, weights, dists, rad, bound_dists)
            
            if use_spectral and spectral_weight > 1e-6:
                # Scale spectral weight: decay as we get closer to optimum or as k increases
                # High k means we are focusing on local minima, spectral regularizer can be reduced
                # But early on, it helps global structure.
                # Let's decay it slowly.
                spec_grad = spectral_regularization_grad(pts, weight=spectral_weight)
                grads += spec_grad
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-10:
                grads /= grad_norm
            
            pts += alpha * grads
            
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            alpha *= cooling
            progress = min(1.0, (time.time() - start_time) / time_budget)
            k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Decay spectral weight
            spectral_weight *= 0.9995
            
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005)
                # Re-boost spectral weight on restart to help escape local minima via global structure
                spectral_weight = max(spectral_weight, 0.2)
        
        return best_pts, best_score

    def active_set_newton(pts, time_budget):
        """
        Active-set Newton refinement on binding constraints.
        """
        if time.time() >= start_time + time_budget:
            return pts, compute_score(pts)
        
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        max_iters = 80
        tol = 1e-14
        
        for iteration in range(max_iters):
            if time.time() >= start_time + time_budget:
                break
            
            dists, rad, bound_dists, all_vals = compute_all_dists(pts)
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
            
            binding_threshold = current_min + 1e-8
            binding_mask = all_vals <= binding_threshold
            num_binding = np.sum(binding_mask)
            
            if num_binding == 0:
                break
            
            if num_binding > 40:
                binding_indices = np.argsort(all_vals)[:40]
            else:
                binding_indices = np.where(binding_mask)[0]
            
            num_constraints = len(binding_indices)
            A = np.zeros((num_constraints, 2*N))
            
            for idx, ci in enumerate(binding_indices):
                if ci < num_pairs:
                    i, j = i_idx[ci], j_idx[ci]
                    d = dists[i, j]
                    if d < 1e-10:
                        continue
                    vec = (pts[i] - pts[j]) / d
                    A[idx, 2*i:2*i+2] = vec
                    A[idx, 2*j:2*j+2] = -vec
                else:
                    k = ci - num_pairs
                    r = rad[k]
                    if r < 1e-10:
                        continue
                    vec = -pts[k] / r
                    A[idx, 2*k:2*k+2] = vec
            
            try:
                delta_x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(num_constraints), rcond=None)
            except:
                break
            
            step_size = 5e-5
            for _ in range(5):
                pts_trial = pts + step_size * delta_x.reshape(N, 2)
                rad_new = np.linalg.norm(pts_trial, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts_trial[mask_out] /= rad_new[mask_out, np.newaxis]
                
                trial_score = compute_score(pts_trial)
                if trial_score > best_score:
                    pts = pts_trial
                    best_score = trial_score
                    best_pts = pts.copy()
                    step_size *= 1.5
                else:
                    break
            
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            new_score = compute_score(pts)
            if abs(new_score - current_min) < tol:
                break
        
        return best_pts, best_score

    def anneal_restart(pts, time_budget):
        """
        Anneal restart: perturb the best solution and re-optimize.
        """
        if time.time() >= start_time + time_budget:
            return pts, compute_score(pts)
        
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        num_restarts = 4
        time_per_restart = time_budget / num_restarts
        
        for r in range(num_restarts):
            if time.time() >= start_time + time_budget:
                break
            
            noise_scale = 0.004 * (1 - r / num_restarts) + 0.001
            pts_perturbed = pts + np.random.normal(0, noise_scale, size=pts.shape)
            rad_new = np.linalg.norm(pts_perturbed, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts_perturbed[mask_out] /= rad_new[mask_out, np.newaxis]
            
            pts_opt, score = optimize(pts_perturbed,
                                       alpha_init=0.002,
                                       k_init=150.0,
                                       k_end=600.0,
                                       cooling=0.9997,
                                       time_budget=time_per_restart,
                                       stall_threshold=80,
                                       noise_scale_base=0.001,
                                       use_spectral=True,
                                       spectral_weight_init=0.1)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
        
        return best_pts, best_score

    def sharp_soft_min_polish(pts, time_budget):
        """
        Final polish with very sharp soft-min and tiny steps.
        """
        if time.time() >= start_time + time_budget:
            return pts, compute_score(pts)
        
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = 0.0005
        k = 800.0
        stall_count = 0
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            dists, rad, bound_dists, all_vals = compute_all_dists(pts)
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            weights = soft_min_weights(all_vals, k, current_min)
            grads = compute_grads(pts, weights, dists, rad, bound_dists)
            
            # No spectral regularization in final polish, focus on local precision
            # if use_spectral: ... (disabled)
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-10:
                grads /= grad_norm
            
            pts += alpha * grads
            
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            alpha *= 0.998
            k *= 1.005
            
            if stall_count > 200:
                pts += np.random.normal(0, 0.0005, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.0003)
                k = max(k, 1500.0)
        
        return best_pts, best_score

    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts

    def random_init():
        """Random initialization with rejection sampling."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)

    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.93, 0.96, 0.99, 1.0, 1.01, 1.03]
    
    for mult in spacing_multipliers:
        if time.time() - start_time > time_limit * 0.45:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        time_budget = min(6.0, time_limit * 0.06)
        pts_opt, score = optimize(pts,
                                   alpha_init=0.012,
                                   k_init=2.0,
                                   k_end=150.0,
                                   cooling=0.9998,
                                   time_budget=time_budget,
                                   stall_threshold=80,
                                   noise_scale_base=0.006,
                                   use_spectral=True,
                                   spectral_weight_init=0.5)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.55:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.55:
                break
            
            pts = random_init()
            time_budget = min(5.0, time_limit * 0.05)
            pts_opt, score = optimize(pts,
                                       alpha_init=0.008,
                                       k_init=1.5,
                                       k_end=100.0,
                                       cooling=0.9997,
                                       time_budget=time_budget,
                                       stall_threshold=60,
                                       noise_scale_base=0.008,
                                       use_spectral=True,
                                       spectral_weight_init=0.3)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish with gradient-based optimization (sharper)
    if best_pts is not None and time.time() - start_time < time_limit * 0.60:
        remaining_time = time_limit * 0.70 - time.time()
        if remaining_time > 3.0:
            pts_polished, score_polished = optimize(best_pts,
                                                     alpha_init=0.002,
                                                     k_init=100.0,
                                                     k_end=500.0,
                                                     cooling=0.9995,
                                                     time_budget=min(remaining_time, 8.0),
                                                     stall_threshold=150,
                                                     noise_scale_base=0.001,
                                                     use_spectral=True,
                                                     spectral_weight_init=0.2)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
    
    # Phase 4: Active-set Newton refinement
    if best_pts is not None and time.time() - start_time < time_limit * 0.70:
        remaining_time = time_limit * 0.78 - time.time()
        if remaining_time > 2.0:
            pts_newton, score_newton = active_set_newton(best_pts, min(remaining_time, 8.0))
            
            if score_newton > best_score:
                best_score = score_newton
                best_pts = pts_newton
    
    # Phase 5: Anneal restart from perturbed best solution
    if best_pts is not None and time.time() - start_time < time_limit * 0.78:
        remaining_time = time_limit * 0.88 - time.time()
        if remaining_time > 3.0:
            pts_anneal, score_anneal = anneal_restart(best_pts, min(remaining_time, 12.0))
            
            if score_anneal > best_score:
                best_score = score_anneal
                best_pts = pts_anneal
    
    # Phase 6: Active-set Newton again after anneal
    if best_pts is not None and time.time() - start_time < time_limit * 0.88:
        remaining_time = time_limit * 0.94 - time.time()
        if remaining_time > 1.0:
            pts_newton2, score_newton2 = active_set_newton(best_pts, min(remaining_time, 4.0))
            
            if score_newton2 > best_score:
                best_score = score_newton2
                best_pts = pts_newton2
    
    # Phase 7: Sharp soft-min final polish
    if best_pts is not None and time.time() - start_time < time_limit * 0.94:
        remaining_time = time_limit * 0.995 - time.time()
        if remaining_time > 1.0:
            pts_sharp, score_sharp = sharp_soft_min_polish(best_pts, min(remaining_time, 4.0))
            
            if score_sharp > best_score:
                best_score = score_sharp
                best_pts = pts_sharp
    
    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
