import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    n = 14

    def sph_to_cart(sph):
        """Convert spherical coordinates (theta, phi) to Cartesian on unit sphere."""
        theta = sph[0::2]  # Polar angle [0, pi]
        phi = sph[1::2]    # Azimuthal angle [0, 2*pi]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])

    def exact_metric(sph):
        """Calculate the actual metric (d_min / d_max)^2."""
        cart = sph_to_cart(sph)
        d = pdist(cart)
        if len(d) == 0:
            return 0.0
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_metric_neg(sph, alpha=100.0):
        """
        Negative soft metric for optimization.
        Uses log-sum-exp to approximate min and max distances smoothly.
        """
        cart = sph_to_cart(sph)
        d = pdist(cart)
        if len(d) == 0:
            return 0.0
        
        # Avoid numerical issues in exp
        # Clip distances to prevent overflow/underflow in exp
        d_clipped = np.clip(d, 1e-12, 10.0)
        
        # Soft min: -log(sum(exp(-alpha*d))) / alpha
        # Soft max: log(sum(exp(alpha*d))) / alpha
        # We want to maximize (soft_min / soft_max)^2
        # So we minimize - (soft_min / soft_max)^2
        
        # For stability, we can compute log_sum_exp manually or use scipy if available, 
        # but simple np.log(np.sum(np.exp(...))) is fine if we manage the range.
        
        # To avoid overflow in exp(alpha*d) for large alpha*d:
        # max_d = np.max(d_clipped)
        # min_d = np.min(d_clipped)
        
        # Let's use a stable version.
        # soft_min = -1/alpha * log(sum(exp(-alpha * d)))
        # Let m = -alpha * max(d). Then exp(-alpha*d) = exp(m) * exp(-alpha*(d-max(d)))
        # sum(exp(-alpha*d)) = exp(m) * sum(exp(-alpha*(d-max(d))))
        # log(sum(exp(-alpha*d))) = m + log(sum(exp(-alpha*(d-max(d)))))
        
        max_d = np.max(d_clipped)
        min_d = np.min(d_clipped)
        
        # For soft min
        # terms_min = np.exp(-alpha * (d_clipped - max_d))
        # log_sum_min = -alpha * max_d + np.log(np.sum(terms_min))
        # soft_min = -1/alpha * log_sum_min
        
        # For soft max
        # terms_max = np.exp(alpha * (d_clipped - min_d))
        # log_sum_max = alpha * min_d + np.log(np.sum(terms_max))
        # soft_max = 1/alpha * log_sum_max
        
        # Actually, standard stable log-sum-exp:
        # LSE(x) = max(x) + log(sum(exp(x - max(x))))
        
        x_min = -alpha * d_clipped
        max_x_min = np.max(x_min)
        log_sum_min = max_x_min + np.log(np.sum(np.exp(x_min - max_x_min)))
        soft_min = -log_sum_min / alpha
        
        x_max = alpha * d_clipped
        max_x_max = np.max(x_max)
        log_sum_max = max_x_max + np.log(np.sum(np.exp(x_max - max_x_max)))
        soft_max = log_sum_max / alpha
        
        if soft_max < 1e-12:
            return 0.0
            
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    def fib_sphere(n):
        k = np.arange(n)
        # theta is polar angle
        theta = np.arccos(1 - 2 * (k + 0.5) / n)
        golden = np.pi * (1 + np.sqrt(5))
        phi = golden * k
        # Interleave theta and phi
        sph = np.empty(2 * n)
        sph[0::2] = theta
        sph[1::2] = phi
        return sph

    # Initializations
    inits = []
    inits.append(fib_sphere(n))
    
    # Random inits
    for seed in range(10):
        np.random.seed(seed)
        theta = np.random.uniform(0, np.pi, n)
        phi = np.random.uniform(0, 2 * np.pi, n)
        sph = np.empty(2 * n)
        sph[0::2] = theta
        sph[1::2] = phi
        inits.append(sph)

    # Icosahedron based init
    def icosahedron_init():
        t = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        # Add 2 more points, e.g., poles
        extra = np.array([[0, 0, 1], [0, 0, -1]])
        all_pts = np.vstack([verts, extra])
        theta = np.arccos(np.clip(all_pts[:, 2], -1, 1))
        phi = np.arctan2(all_pts[:, 1], all_pts[:, 0])
        phi = np.mod(phi, 2 * np.pi)
        sph = np.empty(2 * n)
        sph[0::2] = theta
        sph[1::2] = phi
        return sph

    inits.append(icosahedron_init())

    best_sph = None
    best_score = -1.0

    # 1. Local optimization from multiple starts using L-BFGS-B on soft metric
    # We use a moderate alpha initially for smoothness
    alpha_start = 50.0
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(soft_metric_neg, init, args=(alpha_start,), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_sph = res.x.copy()
        except Exception:
            continue

    if best_sph is None:
        best_sph = fib_sphere(n)
        best_score = -soft_metric_neg(best_sph, alpha_start)

    # 2. Basin Hopping to find better global minima
    # We define a local minimizer for basin hopping
    def local_minimizer(x):
        # Use L-BFGS-B with a slightly higher alpha for more precise local min
        res = minimize(soft_metric_neg, x, args=(100.0,), method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12})
        return {'fun': res.fun, 'x': res.x, 'nit': res.nit}

    try:
        if time.time() < deadline - 120:
            bh = basinhopping(soft_metric_neg, best_sph, 
                              minimizer=local_minimizer,
                              niter=100, stepsize=0.5, 
                              interval=20, disp=False,
                              args=(100.0,)) # Note: args are passed to the objective function in basinhopping? 
                              # Actually, basinhopping passes x to the objective. The local_minimizer uses the objective internally.
                              # Wait, basinhopping's objective function signature is f(x, *args).
                              # local_minimizer calls minimize(soft_metric_neg, x, args=(100.0,)).
                              # So bh needs to call soft_metric_neg(x, 100.0).
                              # Let's re-structure slightly to be safe.
            
            # Redefine local minimizer to match basinhopping expectations if needed, 
            # but standard basinhopping expects minimizer(x) returning dict.
            # The objective function passed to basinhopping is used for acceptance criteria.
            
            # Let's run basin hopping
            # We need to pass alpha to the objective function used by basin hopping for comparison.
            # Let's create a wrapper for the objective with fixed alpha for BH.
            def obj_for_bh(x):
                return soft_metric_neg(x, 100.0)
            
            bh = basinhopping(obj_for_bh, best_sph, 
                              minimizer=local_minimizer,
                              niter=100, stepsize=0.5, 
                              interval=20, disp=False)
            
            if bh.fun < -best_score: # Since we minimize negative score
                # bh.fun is negative score. So -bh.fun is positive score.
                current_score = -bh.fun
                if current_score > best_score:
                    best_score = current_score
                    best_sph = bh.x.copy()
    except Exception:
        pass

    # 3. Final Polish with Nelder-Mead on the EXACT metric
    # The soft metric is an approximation. Nelder-Mead can handle nonsmooth functions.
    def neg_exact_metric(sph):
        return -exact_metric(sph)

    try:
        if time.time() < deadline - 10:
            res_final = minimize(neg_exact_metric, best_sph, method='Nelder-Mead',
                                 options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000, 'adaptive': True})
            final_score = -res_final.fun
            if np.isfinite(final_score) and final_score > best_score:
                best_score = final_score
                best_sph = res_final.x.copy()
    except Exception:
        pass

    # Convert to Cartesian
    result = sph_to_cart(best_sph)
    
    # Ensure points are on unit sphere (numerical cleanup)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    mask = norms < 1e-12
    if np.any(mask):
        norms[mask] = 1.0
        result[mask, 0] = 1.0
        result[mask, 1:] = 0.0
    result = result / norms
    
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
