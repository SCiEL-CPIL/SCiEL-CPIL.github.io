import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def get_radius(centers):
        """Compute the exact radius for given centers."""
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        r_wall = np.min(wall_dists)
        
        # Pairwise distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_wall, r_pair)

    def soft_min_obj(centers_flat, beta):
        """
        Log-sum-exp based soft minimum objective for L-BFGS-B.
        We want to maximize min(r_wall, r_pair).
        Equivalent to minimizing -min(r_wall, r_pair).
        We approximate -min with a soft max of the negative terms, or soft min of the positive terms.
        
        Let r_i be the set of all constraints (wall distances and half-pair distances).
        We want to maximize min(r_i).
        min(r_i) ≈ -1/beta * log(sum(exp(-beta * r_i)))
        
        Objective to minimize: -min(r_i) ≈ 1/beta * log(sum(exp(-beta * r_i)))
        
        To avoid overflow/underflow, we use the log-sum-exp trick:
        sum(exp(-beta * r_i)) = exp(-beta * r_max) * sum(exp(-beta * (r_i - r_max)))
        log(sum(exp(-beta * r_i))) = -beta * r_max + log(sum(exp(-beta * (r_i - r_max))))
        
        So objective = 1/beta * (-beta * r_max + log(sum(exp(-beta * (r_i - r_max)))))
                     = -r_max + (1/beta) * log(sum(exp(-beta * (r_i - r_max))))
        
        This is a smooth approximation of -min(r_i). Minimizing this maximizes min(r_i).
        """
        centers = centers_flat.reshape(N, 3)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        
        # Pairwise half-distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Combine all constraints into a single array
        # Shape: (N + N*(N-1)) roughly, but we can just flatten the lower triangle and walls
        # Actually, easier to just stack them
        # Walls: N values
        # Pairs: N*(N-1)/2 values
        # For efficiency, let's just compute the soft min directly on the combined set.
        
        # We need to be careful with memory for N=100. 100*99/2 = 4950 pairs.
        # Total constraints ~ 5050. This is small enough to stack.
        
        # Extract upper triangle of pair distances
        iu = np.triu_indices(N, k=1)
        pair_vals = pair_dists[iu]
        
        # Stack all constraints
        all_vals = np.concatenate([wall_dists.ravel(), pair_vals])
        
        # Log-sum-exp trick
        v_max = np.max(all_vals)
        # If v_max is very negative, exp(-beta * (v - v_max)) might still be okay if v-v_max is small
        # But if beta is large, we might have issues.
        # Standard trick:
        # LSE(v) = v_max + log(sum(exp(v - v_max)))
        # Here we have log(sum(exp(-beta * v)))
        # = -beta * v_max + log(sum(exp(-beta * (v - v_max))))
        
        # Let u = -beta * (v - v_max) = beta * (v_max - v)
        # Since v <= v_max, u >= 0.
        # exp(u) can overflow if u is large.
        # However, v_max - v is 0 for the max, and positive for others.
        # If beta is large, even small differences can cause overflow.
        # To be safe, we can clip or use a different approach.
        # Alternatively, use the formula:
        # soft_min(v) = -1/beta * log(sum(exp(-beta * v)))
        # = v_max - 1/beta * log(sum(exp(beta * (v_max - v))))
        
        # Let diff = v_max - v >= 0
        # sum(exp(beta * diff))
        # If beta * diff is large, exp overflows.
        # We can use the property that only terms close to v_max contribute significantly.
        # But for robustness, let's just use numpy's logaddexp.reduce if available, or implement carefully.
        
        # Actually, for optimization, we often just use a simpler approximation or handle scaling.
        # Let's try to compute:
        # obj = v_max - (1/beta) * log(sum(exp(beta * (v_max - v))))
        
        # To prevent overflow in exp, we can subtract the max of the exponent (which is 0, since diff>=0).
        # Wait, max of beta*(v_max-v) is 0. So exp(0)=1.
        # The other terms are exp(negative_or_zero)? No, v_max-v >= 0, so beta*(v_max-v) >= 0.
        # So the max is 0? No, v_max is the MAX of v. So v_max - v_min is the largest positive.
        # So beta * (v_max - v) can be large positive.
        
        # Let's re-evaluate.
        # We want S = sum(exp(-beta * v))
        # Let M = max(v).
        # S = exp(-beta * M) * sum(exp(-beta * (v - M)))
        # Since v - M <= 0, let d = M - v >= 0.
        # S = exp(-beta * M) * sum(exp(beta * d))
        # log(S) = -beta * M + log(sum(exp(beta * d)))
        
        # The term sum(exp(beta * d)) can overflow if beta * d_max is large.
        # We can compute log(sum(exp(x))) using the log-sum-exp trick:
        # log(sum(exp(x))) = x_max + log(sum(exp(x - x_max)))
        # Here x = beta * d. x_max = beta * d_max.
        # So log(sum(exp(beta * d))) = beta * d_max + log(sum(exp(beta * (d - d_max))))
        # Note d - d_max <= 0. So exp(beta * (d - d_max)) <= 1. No overflow.
        
        # So:
        # log(S) = -beta * M + beta * d_max + log(sum(exp(beta * (d - d_max))))
        # obj = -1/beta * log(S) = M - d_max - (1/beta) * log(sum(exp(beta * (d - d_max))))
        # Note M - d_max = M - (M - v_min) = v_min.
        # So obj = v_min - (1/beta) * log(sum(exp(beta * (d - d_max))))
        
        # As beta -> inf, the term in log approaches log(1) = 0 (only the max d contributes).
        # So obj -> v_min. Correct.
        
        # Implementation:
        d = v_max - all_vals  # d >= 0, d_max = d[argmax(d)] = d[v_min_idx]
        d_max = np.max(d)
        
        # Exponents: e = beta * (d - d_max) <= 0
        e = beta * (d - d_max)
        
        # log_sum_exp(e)
        # max_e = 0 (since d - d_max <= 0, max is 0)
        # sum_exp = sum(exp(e))
        # log_sum_exp = log(sum_exp)
        
        # To be safe against underflow (exp(e) -> 0 for very negative e),
        # the sum is dominated by terms where e is close to 0.
        # np.sum(np.exp(e)) should be fine as long as at least one term is not 0.
        # Since d_max corresponds to d - d_max = 0, exp(0) = 1.
        # So sum_exp >= 1. log(sum_exp) >= 0.
        
        sum_exp = np.sum(np.exp(e))
        log_sum_exp = np.log(sum_exp)
        
        obj = v_max - d_max - (1.0 / beta) * log_sum_exp
        
        return obj

    def gradient_obj(centers_flat, beta):
        """
        Approximate gradient of the soft-min objective.
        Since L-BFGS-B can use analytic gradients, it's faster.
        But implementing the full gradient of the log-sum-exp of pairwise distances is complex.
        We can use finite differences or just let L-BFGS-B approximate it.
        For N=100, finite differences might be slow.
        Let's stick with numerical gradient first, or use a simpler objective if possible.
        
        Alternative: Use the fact that for large beta, the objective is close to -min.
        We can just minimize the sum of squared penalties for violations of a target radius r?
        
        Let's try a different objective:
        Minimize sum( max(0, target_r - r_i)^2 ) + lambda * (target_r)^2 ?
        This is non-smooth.
        
        Let's stick to the soft-min but use numerical gradient.
        L-BFGS-B approximates gradient if not provided.
        """
        return None # Use numerical gradient

    # Initial Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    base_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    best_centers = base_centers.copy()
    best_radius = get_radius(best_centers)
    
    np.random.seed(42)
    
    # Adaptive beta schedule
    # Start with low beta (smooth), increase to high beta (sharp)
    betas = [10, 20, 50, 100, 200]
    
    current_centers = base_centers.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Flatten
        x0 = current_centers.ravel()
        
        # Bounds: [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Minimize
        try:
            res = minimize(
                fun=lambda x: soft_min_obj(x, beta),
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 1000}
            )
            
            if res.success or res.fun < 0: # Check if it improved
                new_centers = res.x.reshape(N, 3)
                # Ensure validity
                if np.all(np.isfinite(new_centers)):
                    # Clip to be safe
                    new_centers = np.clip(new_centers, -0.5, 0.5)
                    r_new = get_radius(new_centers)
                    
                    if r_new > best_radius:
                        best_radius = r_new
                        best_centers = new_centers.copy()
                    
                    # Update current center for next beta
                    # We should start the next beta from the improved position
                    current_centers = new_centers.copy()
                    
        except Exception as e:
            pass
            
    # Final polish with a higher beta if time permits
    if time.time() < deadline - 5:
        beta_final = 500
        x0 = best_centers.ravel()
        try:
            res = minimize(
                fun=lambda x: soft_min_obj(x, beta_final),
                x0=x0,
                bounds=[(-0.5, 0.5)] * (N * 3),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 500}
            )
            if res.success or res.fun < 0:
                new_centers = res.x.reshape(N, 3)
                if np.all(np.isfinite(new_centers)):
                    new_centers = np.clip(new_centers, -0.5, 0.5)
                    r_new = get_radius(new_centers)
                    if r_new > best_radius:
                        best_radius = r_new
                        best_centers = new_centers.copy()
        except Exception:
            pass
            
    return best_centers
