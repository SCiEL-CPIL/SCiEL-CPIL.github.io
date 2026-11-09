"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

def construct():
    t0 = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # --- Helper Functions ---
    
    def exact_radius(centers):
        """Compute the exact radius for a given configuration."""
        if len(centers) != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Boundary clearance
        boundary = 0.5 - np.max(np.abs(centers), axis=1)
        min_boundary = np.min(boundary)
        if min_boundary <= 0:
            return 0.0
        # Pairwise distances
        # For N=50, O(N^2) is fast enough
        d = pdist(centers, metric='euclidean')
        min_pair = np.min(d) / 2.0
        return min(min_pair, min_boundary)

    def soft_min_score(x, beta):
        """
        Smooth surrogate: maximize soft-min of (pairwise dists/2, wall dists).
        We minimize negative soft-min.
        """
        centers = x.reshape(N, 3)
        # Boundary distances
        boundary = 0.5 - np.max(np.abs(centers), axis=1)
        # Pairwise distances / 2
        d = pdist(centers, metric='euclidean') / 2.0
        
        all_vals = np.concatenate([boundary, d])
        
        # Soft-min approximation: -min(x) ≈ (1/beta) log(sum(exp(-beta x)))
        # We want to minimize this value (which is negative of the soft min).
        # Wait, the standard soft-min is min(x) ≈ -(1/beta) log(sum(exp(-beta x))).
        # So we want to MAXIMIZE min(x), which is MINIMIZING (1/beta) log(sum(exp(-beta x))).
        
        # Numerical stability
        x_neg = -beta * all_vals
        x_max = np.max(x_neg)
        log_sum_exp = x_max + np.log(np.sum(np.exp(x_neg - x_max)))
        f = (1.0 / beta) * log_sum_exp
        return f

    def soft_min_grad(x, beta):
        """
        Gradient of the soft-min objective w.r.t. centers.
        Objective: f = (1/beta) log(sum(exp(-beta * all_vals)))
        """
        centers = x.reshape(N, 3)
        boundary = 0.5 - np.max(np.abs(centers), axis=1)
        d = pdist(centers, metric='euclidean') / 2.0
        all_vals = np.concatenate([boundary, d])
        
        # Gradient of f w.r.t. all_vals
        x_neg = -beta * all_vals
        x_max = np.max(x_neg)
        exp_vals = np.exp(x_neg - x_max)
        sum_exp = np.sum(exp_vals)
        
        # df/d(all_vals_k) = (1/beta) * (1/sum_exp) * (-beta) * exp(-beta*all_vals_k)
        #                  = -exp_vals_k / sum_exp
        grad_all = -exp_vals / sum_exp
        
        grad_centers = np.zeros_like(centers)
        
        # 1. Boundary gradient
        # boundary_i = 0.5 - max_k |c_ik|
        # d(boundary_i)/d(c_ik) = -sign(c_ik) if k = argmax_j |c_ij|
        for i in range(N):
            k = np.argmax(np.abs(centers[i]))
            # grad_all[i] is the weight for this boundary term
            grad_centers[i, k] += grad_all[i] * (-np.sign(centers[i, k]))
        
        # 2. Pairwise gradient
        # d_k = |c_a - c_b|/2
        # d(d_k)/d(c_a) = (c_a - c_b) / |c_a - c_b| * 0.5
        # d(d_k)/d(c_b) = (c_b - c_a) / |c_a - c_b| * 0.5
        
        # pdist order: (0,1), (0,2), ..., (0,N-1), (1,2), ...
        pair_idx = 0
        for a in range(N - 1):
            for b in range(a + 1, N):
                w = grad_all[N + pair_idx]
                if w == 0.0:
                    pair_idx += 1
                    continue
                diff = centers[a] - centers[b]
                dist = np.linalg.norm(diff)
                if dist > 1e-15:
                    unit = diff / dist
                    grad_centers[a] += w * unit * 0.5
                    grad_centers[b] += w * (-unit) * 0.5
                pair_idx += 1
        
        return grad_centers.flatten()

    # --- Optimization Loop ---
    
    best_x = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Standard Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 2. Jittered Grid
    rng = np.random.default_rng(42)
    inits.append(grid[:N].copy() + rng.normal(0, 0.02, size=(N, 3)))
    
    # 3. Different Grid (4x4x4)
    m2 = 4
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
    inits.append(grid2[:N].copy())
    
    # 4. Random Uniform (safe inside)
    rng = np.random.default_rng(123)
    inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 5. 3x4x5 Grid
    m3, m4, m5 = 3, 4, 5
    tx = (np.arange(m3) + 0.5) / m3 - 0.5
    ty = (np.arange(m4) + 0.5) / m4 - 0.5
    tz = (np.arange(m5) + 0.5) / m5 - 0.5
    grid3 = np.array([(x, y, z) for z in tz for y in ty for x in tx])
    inits.append(grid3[:N].copy())

    # Betas for annealing
    betas = [20, 50, 100, 200, 500]
    
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        x_curr = x_init.copy()
        score_curr = exact_radius(x_curr)
        
        if score_curr > best_score:
            best_score = score_curr
            best_x = x_curr.copy()
            
        # Annealing loop
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            def obj(x):
                return soft_min_score(x, beta)
            
            def grad(x):
                return soft_min_grad(x, beta)
                
            try:
                # L-BFGS-B
                result = scipy.optimize.minimize(
                    obj, 
                    x_curr.flatten(), 
                    jac=grad, 
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                
                x_new = result.x.reshape(N, 3)
                x_new = np.clip(x_new, -0.5, 0.5)
                score_new = exact_radius(x_new)
                
                # Accept if better or if it's a valid improvement in the smooth landscape
                # We strictly require exact radius improvement to avoid drifting away
                if score_new > score_curr:
                    x_curr = x_new
                    score_curr = score_new
                    if score_curr > best_score:
                        best_score = score_curr
                        best_x = x_curr.copy()
                else:
                    # If no improvement, maybe try a small perturbation and restart this beta stage?
                    # Or just break to next beta?
                    # Let's try a small random jump to escape local min in the smooth landscape
                    if beta > 50: # Only do this for higher betas where we might be stuck
                        rng_jump = np.random.default_rng(int(time.time()*1000) + init_idx + beta)
                        x_perturbed = x_curr + rng_jump.normal(0, 0.01, size=(N, 3))
                        x_perturbed = np.clip(x_perturbed, -0.49, 0.49)
                        # Re-optimize from perturbed state with same beta
                        result2 = scipy.optimize.minimize(
                            obj, 
                            x_perturbed.flatten(), 
                            jac=grad, 
                            method='L-BFGS-B',
                            bounds=[(-0.5, 0.5)] * (N * 3),
                            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        x_new2 = result2.x.reshape(N, 3)
                        x_new2 = np.clip(x_new2, -0.5, 0.5)
                        score_new2 = exact_radius(x_new2)
                        if score_new2 > score_curr:
                            x_curr = x_new2
                            score_curr = score_new2
                            if score_curr > best_score:
                                best_score = score_curr
                                best_x = x_curr.copy()
            except Exception:
                pass

    # Final High-Precision Polish on the best solution found
    if best_x is not None and time.time() < deadline - 2.0:
        x_curr = best_x.copy()
        score_curr = best_score
        
        # Use very high beta for final polish
        beta_final = 2000
        
        def obj_final(x):
            return soft_min_score(x, beta_final)
            
        def grad_final(x):
            return soft_min_grad(x, beta_final)
            
        try:
            result = scipy.optimize.minimize(
                obj_final, 
                x_curr.flatten(), 
                jac=grad_final, 
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            x_new = result.x.reshape(N, 3)
            x_new = np.clip(x_new, -0.5, 0.5)
            score_new = exact_radius(x_new)
            
            if score_new > score_curr:
                x_curr = x_new
                score_curr = score_new
                best_x = x_curr
                best_score = score_curr
        except Exception:
            pass

    # Validity check and cleanup
    if best_x is None:
        # Fallback to grid
        best_x = inits[0]
        
    best_x = np.clip(best_x, -0.5, 0.5)
    
    # Ensure no coincident centers (just in case)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-10:
                best_x[j] += 1e-5 * np.array([1, 0, 0])
                best_x[j] = np.clip(best_x[j], -0.5, 0.5)
                
    return best_x

# EVOLVE-BLOCK-END
