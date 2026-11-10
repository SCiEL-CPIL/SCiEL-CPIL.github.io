# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Gram-matrix optimization to find a configuration with high symmetry and equalized distances,
    followed by coordinate-space polishing.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds
    safety_margin = 30  # seconds before limit to stop
    
    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def exact_objective(x):
        """Exact (d_max/d_min)^2 for final polish."""
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max == 0:
            return 1e10
        return (d_max / d_min) ** 2

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        x: flat array of length 42
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    # --- Gram Matrix Optimization ---
    
    def gram_objective(x_flat, beta=0.05, w_psd=100.0, w_rank=100.0, w_diag=100.0):
        """
        Objective for optimizing the Gram matrix G.
        x_flat: vector of length n*n representing the flattened symmetric matrix G.
        We only optimize off-diagonal elements, but for simplicity in L-BFGS-B, 
        we pass the full vector and ignore diagonal updates or fix them.
        Actually, L-BFGS-B optimizes all variables. We will penalize diagonal deviations.
        
        Terms:
        1. -SoftMin(d_ij^2) : Maximize minimum distance.
        2. w_diag * sum((G_ii - 1)^2) : Enforce unit norm.
        3. w_psd * sum(lambda_neg^2) : Enforce PSD.
        4. w_rank * sum(lambda_i^2 for i>3) : Enforce rank <= 3.
        """
        G = x_flat.reshape((n, n))
        
        # Symmetrize
        G = (G + G.T) / 2.0
        
        # Compute eigenvalues for penalties
        try:
            eigvals = np.linalg.eigvalsh(G)
        except np.linalg.LinAlgError:
            return 1e10
            
        # 1. Diagonal penalty
        diag = np.diag(G)
        diag_penalty = w_diag * np.sum((diag - 1.0) ** 2)
        
        # 2. PSD penalty
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = w_psd * np.sum(neg_eigs ** 2) if len(neg_eigs) > 0 else 0.0
        
        # 3. Rank penalty (keep top 3 eigenvalues)
        # Sort eigenvalues descending
        eigvals_sorted = np.sort(eigvals)[::-1]
        # Penalize eigenvalues from index 3 onwards (0-indexed, so 3, 4, ..., 13)
        rank_penalty = w_rank * np.sum(eigvals_sorted[3:] ** 2)
        
        # 4. Main objective: Maximize min distance
        # Extract off-diagonal distances squared
        # d_ij^2 = 2 - 2 G_ij
        # We need a mask for i < j
        iu = np.triu_indices(n, k=1)
        g_offdiag = G[iu]
        d2_offdiag = 2.0 - 2.0 * g_offdiag
        
        # Ensure d2 is positive for log/exp stability in SoftMin
        # If d2 is negative, it means G_ij > 1, which is invalid for unit vectors.
        # The diagonal penalty and PSD penalty should prevent this, but we clip for safety.
        d2_safe = np.maximum(d2_offdiag, 1e-6)
        
        # SoftMin: -beta * log(sum(exp(-d2/beta)))
        a = -d2_safe / beta
        max_a = np.max(a)
        log_sum = max_a + np.log(np.sum(np.exp(a - max_a)))
        soft_min_d2 = -beta * log_sum
        
        # We want to MAXIMIZE soft_min_d2, so we MINIMIZE -soft_min_d2
        main_obj = -soft_min_d2
        
        total_obj = main_obj + diag_penalty + psd_penalty + rank_penalty
        
        if not np.isfinite(total_obj):
            return 1e10
            
        return total_obj

    def gram_to_points(x_flat):
        """
        Convert optimized Gram matrix to 3D points.
        """
        G = x_flat.reshape((n, n))
        G = (G + G.T) / 2.0
        
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        
        # Sort by eigenvalue descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Keep top 3 components
        k = min(3, n)
        v = eigvecs[:, :k]
        w = eigvals[:k]
        
        # Ensure w is non-negative for sqrt
        w = np.maximum(w, 0.0)
        
        # P = sqrt(w) * v.T
        # v is (n, k), w is (k,)
        # P should be (n, k)
        P = v * np.sqrt(w)
        
        return P

    # Initial guess: Icosahedron vertices + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    points_init = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    points_init = points_init - np.mean(points_init, axis=0)
    # Normalize to unit sphere for Gram matrix consistency
    norms_init = np.linalg.norm(points_init, axis=1)
    points_init = points_init / norms_init[:, np.newaxis]
    
    best_points = points_init.copy()
    best_score = get_score(best_points)
    
    # 1. Gram Matrix Optimization
    if time.time() - start_time < time_limit - safety_margin:
        try:
            # Initialize G from points_init
            G_init = points_init @ points_init.T
            x_init = G_init.ravel()
            
            # Optimize G
            # Use L-BFGS-B
            res_gram = minimize(
                gram_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_gram.fun < 1e9:
                cand_points = gram_to_points(res_gram.x)
                # Normalize candidate points to unit sphere
                norms_cand = np.linalg.norm(cand_points, axis=1)
                cand_points = cand_points / norms_cand[:, np.newaxis]
                
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception as e:
            pass

    # 2. Local Refinement using L-BFGS-B on coordinates
    candidates_to_polish = [best_points.copy(), points_init.copy()]
    
    # Add random perturbations
    np.random.seed(123)
    for _ in range(5):
        perturb = best_points + np.random.randn(n, d) * 0.1
        # Re-normalize to unit sphere
        norms_p = np.linalg.norm(perturb, axis=1)
        perturb = perturb / norms_p[:, np.newaxis]
        candidates_to_polish.append(perturb)
    
    for init_pts in candidates_to_polish:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            # Polish with smooth objective
            res_local = minimize(
                lambda x: smooth_objective(x, beta=0.05), 
                init_pts.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < 1e9:
                opt_points = res_local.x.reshape((n, d))
                # Re-normalize
                norms_opt = np.linalg.norm(opt_points, axis=1)
                opt_points = opt_points / norms_opt[:, np.newaxis]
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
                    
                    # Second polish with smaller beta
                    if time.time() - start_time < time_limit - safety_margin:
                        res_local2 = minimize(
                            lambda x: smooth_objective(x, beta=0.01), 
                            opt_points.ravel(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if res_local2.fun < 1e9:
                            opt_points2 = res_local2.x.reshape((n, d))
                            norms_opt2 = np.linalg.norm(opt_points2, axis=1)
                            opt_points2 = opt_points2 / norms_opt2[:, np.newaxis]
                            opt_score2 = get_score(opt_points2)
                            if opt_score2 > best_score:
                                best_points = opt_points2
                                best_score = opt_score2
        except Exception:
            pass

    # 3. Final high-precision polish with exact objective
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    # 4. Additional Gram optimization from the best coordinate point
    if time.time() - start_time < time_limit - safety_margin:
        try:
            G_curr = best_points @ best_points.T
            x_curr = G_curr.ravel()
            
            res_gram2 = minimize(
                gram_objective,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res_gram2.fun < 1e9:
                cand_points2 = gram_to_points(res_gram2.x)
                norms_cand2 = np.linalg.norm(cand_points2, axis=1)
                cand_points2 = cand_points2 / norms_cand2[:, np.newaxis]
                
                cand_score2 = get_score(cand_points2)
                if cand_score2 > best_score:
                    best_points = cand_points2
                    best_score = cand_score2
                    
                    # Final coordinate polish on this new candidate
                    if time.time() - start_time < time_limit - 10:
                        res_final2 = minimize(
                            exact_objective, 
                            best_points.ravel(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
                        )
                        if res_final2.fun < 1e9:
                            fp2 = res_final2.x.reshape((n, d))
                            fs2 = get_score(fp2)
                            if fs2 > best_score:
                                best_points = fp2
                                best_score = fs2
        except Exception:
            pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        return points_init

    return best_points

# EVOLVE-BLOCK-END
