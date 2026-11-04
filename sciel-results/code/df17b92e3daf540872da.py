import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

N = 300
# Precompute pair indices at module level for efficiency
_I_IDX, _J_IDX = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
_NUM_PAIRS = len(_I_IDX)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    i_idx = _I_IDX
    j_idx = _J_IDX
    num_pairs = _NUM_PAIRS
    
    def compute_score_fast(pts):
        """Compute the packing radius score efficiently using only pair distances."""
        pi = pts[i_idx]
        pj = pts[j_idx]
        diff = pi - pj
        sq_dists = np.sum(diff**2, axis=1)
        d_pair = np.sqrt(sq_dists) / 2.0
        
        # Boundary distances
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        all_vals = np.concatenate([d_pair, bound_dists])
        return np.min(all_vals)
    
    def soft_min_weights(vals, k, current_min):
        """Compute soft-min weights using log-sum-exp for stability."""
        shifted = vals - current_min
        mask_close = shifted < 30.0
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
    
    def optimize(pts, alpha_init, k_init, k_end, cooling, time_budget, 
                 stall_threshold=150, noise_scale_base=0.01, polish=False):
        """
        Main optimization loop using soft-min gradient ascent with fast distance computation.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score_fast(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        end_time = start_time + time_budget
        
        if polish:
            stall_threshold = 200
            cooling = 0.9994
        
        while time.time() < end_time:
            pi = pts[i_idx]
            pj = pts[j_idx]
            diff = pi - pj
            sq_dists = np.sum(diff**2, axis=1)
            d_pair = np.sqrt(sq_dists)
            pair_vals = d_pair / 2.0
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            weights = soft_min_weights(all_vals, k, current_min)
            
            grads = np.zeros_like(pts)
            
            pair_weights = weights[:num_pairs]
            mask_nonzero = d_pair > 1e-10
            if np.any(mask_nonzero):
                idx_m = mask_nonzero
                vecs = (diff[idx_m] / d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
                
                np.add.at(grads, i_idx, vecs)
                np.add.at(grads, j_idx, -vecs)
            
            bound_weights = weights[num_pairs:]
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
            grads += bound_grads
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1.0:
                grads /= grad_norm
            
            if polish:
                step_alpha = alpha * 0.4
            else:
                step_alpha = alpha
            
            pts += step_alpha * grads
            
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            alpha *= cooling
            
            progress = min(1.0, (time.time() - start_time) / time_budget)
            if polish:
                k = k_init + (k_end - k_init) * (progress ** 2.0)
            else:
                k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                if polish:
                    noise_scale *= 0.2
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.004 if not polish else 0.0015)
        
        return best_pts, best_score
    
    def least_squares_residuals(x, target_r):
        """Residuals for least_squares."""
        pts = x.reshape(N, 2)
        pi = pts[i_idx]
        pj = pts[j_idx]
        diff = pi - pj
        sq_dists = np.sum(diff**2, axis=1)
        d_pair = np.sqrt(sq_dists) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        all_dists = np.concatenate([d_pair, bound_dists])
        residuals = np.maximum(0, target_r - all_dists)
        return residuals
    
    def least_squares_jacobian(x, target_r):
        """Analytical Jacobian for least_squares_residuals (vectorized)."""
        pts = x.reshape(N, 2)
        pi = pts[i_idx]
        pj = pts[j_idx]
        diff = pi - pj
        sq_dists = np.sum(diff**2, axis=1)
        d_pair = np.sqrt(sq_dists) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        all_dists = np.concatenate([d_pair, bound_dists])
        
        mask = all_dists < target_r
        if not np.any(mask):
            return np.zeros((len(all_dists), N * 2))
        
        jac = np.zeros((len(all_dists), N * 2))
        
        pair_mask = mask[:num_pairs]
        if np.any(pair_mask):
            idx_m = np.where(pair_mask)[0]
            d_pair_m = d_pair[idx_m]
            safe_d = np.where(d_pair_m > 1e-10, d_pair_m, 1e-10)
            
            pi_m = pts[i_idx[idx_m]]
            pj_m = pts[j_idx[idx_m]]
            vecs = (pi_m - pj_m) / (2.0 * safe_d[:, np.newaxis])
            
            ii = i_idx[idx_m]
            jj = j_idx[idx_m]
            
            jac[idx_m, 2*ii] = -vecs[:, 0]
            jac[idx_m, 2*ii + 1] = -vecs[:, 1]
            jac[idx_m, 2*jj] = vecs[:, 0]
            jac[idx_m, 2*jj + 1] = vecs[:, 1]
        
        bound_mask = mask[num_pairs:]
        if np.any(bound_mask):
            idx_m = np.where(bound_mask)[0]
            rad_m = rad[idx_m]
            safe_rad = np.where(rad_m > 1e-10, rad_m, 1e-10)
            
            pi_m = pts[idx_m]
            vecs = pi_m / safe_rad[:, np.newaxis]
            
            rows = num_pairs + idx_m
            jac[rows, 2*idx_m] = vecs[:, 0]
            jac[rows, 2*idx_m + 1] = vecs[:, 1]
        
        return jac
    
    def run_least_squares(pts_init, target_r, time_budget, max_nfev=1000):
        """Run scipy least_squares to optimize packing with a target radius."""
        x0 = pts_init.flatten()
        end_time = time.time() + time_budget
        
        def time_check(res):
            if time.time() > end_time:
                raise RuntimeError("Time limit exceeded")
        
        try:
            result = least_squares(
                least_squares_residuals,
                x0,
                args=(target_r,),
                jac=least_squares_jacobian,
                bounds=([-1.0, -1.0] * N, [1.0, 1.0] * N),
                method='trf',
                max_nfev=max_nfev,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                callback=time_check
            )
            best_pts = result.x.reshape(N, 2)
            rad = np.linalg.norm(best_pts, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                best_pts[mask_out] /= rad[mask_out, np.newaxis]
            best_score = compute_score_fast(best_pts)
            return best_pts, best_score, True
        except RuntimeError:
            return pts_init, compute_score_fast(pts_init), False
        except Exception:
            return pts_init, compute_score_fast(pts_init), False
    
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
    
    def delaunay_rearrange(pts, n_swaps=50, time_budget=5.0):
        """
        Use Delaunay triangulation to identify local neighborhoods and perform
        topological rearrangements to escape local minima.
        """
        pts = pts.copy()
        best_score = compute_score_fast(pts)
        best_pts = pts.copy()
        end_time = time.time() + time_budget
        
        for swap_round in range(n_swaps):
            if time.time() > end_time:
                break
            
            try:
                tri = Delaunay(pts)
            except Exception:
                break
            
            # Get neighbors for each point from the triangulation
            # For each simplex, all 3 points are neighbors
            neighbors = [set() for _ in range(N)]
            for simplex in tri.simplices:
                a, b, c = simplex
                neighbors[a].add(b)
                neighbors[a].add(c)
                neighbors[b].add(a)
                neighbors[b].add(c)
                neighbors[c].add(a)
                neighbors[c].add(b)
            
            # Identify pairs of points that share a common neighbor (2-hop neighbors)
            # and are not already direct neighbors - these are candidates for swapping
            # to change the local topology
            
            # For each point, find a random neighbor and try to swap positions
            # with a point that is a 2-hop neighbor (not direct neighbor)
            for _ in range(3):
                if time.time() > end_time:
                    break
                
                # Pick a random point
                idx = np.random.randint(0, N)
                
                if len(neighbors[idx]) == 0:
                    continue
                
                # Get 2-hop neighbors (neighbors of neighbors, excluding self and direct neighbors)
                two_hop = set()
                for n1 in neighbors[idx]:
                    for n2 in neighbors[n1]:
                        if n2 != idx and n2 not in neighbors[idx]:
                            two_hop.add(n2)
                
                if len(two_hop) == 0:
                    continue
                
                # Pick a random 2-hop neighbor to swap with
                swap_idx = np.random.choice(list(two_hop))
                
                # Try the swap
                pts_new = pts.copy()
                pts_new[idx], pts_new[swap_idx] = pts[swap_idx], pts[idx]
                
                # Project back inside circle
                rad_new = np.linalg.norm(pts_new, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts_new[mask_out] /= rad_new[mask_out, np.newaxis]
                
                # Check if the swap improves or maintains the score
                score_new = compute_score_fast(pts_new)
                if score_new >= best_score - 1e-10:
                    pts = pts_new
                    if score_new > best_score:
                        best_score = score_new
                        best_pts = pts_new.copy()
            
            # After each round of swaps, do a short optimization to settle
            if time.time() < end_time - 2.0:
                pts_opt, score_opt = optimize(pts,
                                              alpha_init=0.003,
                                              k_init=100.0,
                                              k_end=300.0,
                                              cooling=0.9995,
                                              time_budget=min(2.0, max(0.5, end_time - time.time() - 1.0)),
                                              stall_threshold=50,
                                              noise_scale_base=0.001,
                                              polish=True)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt.copy()
                    pts = pts_opt
        
        return best_pts, best_score
    
    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03]
    
    time_per_init = min(12.0, time_limit * 0.15)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.55:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        if 0.97 <= mult <= 1.03:
            time_budget = time_per_init
        else:
            time_budget = time_per_init * 0.6
        
        pts_opt, score = optimize(pts, 
                                   alpha_init=0.012, 
                                   k_init=5.0, 
                                   k_end=200.0, 
                                   cooling=0.9997,
                                   time_budget=time_budget,
                                   stall_threshold=120,
                                   noise_scale_base=0.006)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.65:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.65:
                break
            
            pts = random_init()
            time_budget = min(10.0, time_limit * 0.10)
            pts_opt, score = optimize(pts,
                                       alpha_init=0.008,
                                       k_init=4.0,
                                       k_end=150.0,
                                       cooling=0.9996,
                                       time_budget=time_budget,
                                       stall_threshold=100,
                                       noise_scale_base=0.008)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Delaunay-based rearrangement to escape local minima
    if best_pts is not None and time.time() - start_time < time_limit * 0.80:
        remaining_time = time_limit * 0.85 - time.time()
        if remaining_time > 8.0:
            pts_delaunay, score_delaunay = delaunay_rearrange(best_pts, n_swaps=30, time_budget=min(remaining_time, 15.0))
            if score_delaunay > best_score:
                best_score = score_delaunay
                best_pts = pts_delaunay
    
    # Phase 4: Polish the best result found
    if best_pts is not None and time.time() - start_time < time_limit * 0.88:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            # First polish pass with soft-min gradient ascent
            pts_polished, score_polished = optimize(best_pts,
                                                     alpha_init=0.004,
                                                     k_init=150.0,
                                                     k_end=400.0,
                                                     cooling=0.9994,
                                                     time_budget=min(remaining_time * 0.25, 6.0),
                                                     stall_threshold=100,
                                                     noise_scale_base=0.002,
                                                     polish=True)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Second: Use scipy least_squares for high-precision polish
            remaining_time2 = time_limit * 0.97 - time.time()
            if remaining_time2 > 4.0:
                target_r = best_score * 1.00005
                ls_budget = min(remaining_time2 * 0.45, 12.0)
                pts_ls, score_ls, success = run_least_squares(best_pts, target_r, ls_budget, max_nfev=2000)
                
                if score_ls > best_score:
                    best_score = score_ls
                    best_pts = pts_ls
            
            # Third: Another polish pass after least_squares
            remaining_time3 = time_limit * 0.97 - time.time()
            if remaining_time3 > 3.0:
                pts_polished2, score_polished2 = optimize(best_pts,
                                                           alpha_init=0.003,
                                                           k_init=200.0,
                                                           k_end=500.0,
                                                           cooling=0.9993,
                                                           time_budget=min(remaining_time3 * 0.3, 5.0),
                                                           stall_threshold=80,
                                                           noise_scale_base=0.001,
                                                           polish=True)
                
                if score_polished2 > best_score:
                    best_score = score_polished2
                    best_pts = pts_polished2
            
            # Fourth: Final high-precision least_squares
            remaining_time4 = time_limit * 0.97 - time.time()
            if remaining_time4 > 2.5:
                target_r4 = best_score * 1.00003
                ls_budget4 = min(remaining_time4 * 0.8, 8.0)
                pts_ls4, score_ls4, _ = run_least_squares(best_pts, target_r4, ls_budget4, max_nfev=1500)
                
                if score_ls4 > best_score:
                    best_score = score_ls4
                    best_pts = pts_ls4
    
    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    # Final high-precision polish with scipy if time allows
    final_remaining = time_limit * 0.98 - time.time()
    if final_remaining > 2.0:
        target_r_final = best_score * 1.00002
        pts_final, score_final, _ = run_least_squares(best_pts, target_r_final, final_remaining, max_nfev=1000)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    # Final safety check
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
