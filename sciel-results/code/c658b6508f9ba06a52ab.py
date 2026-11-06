"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n = 8
    num_pairs = n * (n - 1) // 2  # 28

    # Fix P0=(0,0) and P4=(2,0) to remove translation, rotation, and scale invariance.
    # This assumes the diameter is determined by the pair (0,4).
    fixed_idx = [0, 4]
    free_idx = [i for i in range(n) if i not in fixed_idx]
    n_free = len(free_idx)  # 6
    n_dims = n_free * 2

    def unpack(x):
        p = np.zeros((n, 2))
        p[0] = [0.0, 0.0]
        p[4] = [2.0, 0.0]
        for i, idx in enumerate(free_idx):
            p[idx] = [x[2 * i], x[2 * i + 1]]
        return p

    def objective(x, beta=100.0):
        """
        Minimize: -sum(log d_ij) + penalty for violating diameter <= 2
        
        Uses a smooth quadratic penalty for max(d_ij) > 2, plus a log-sum-exp
        soft-max to handle the nonsmooth max smoothly.
        """
        p = unpack(x)
        # Compute all pairwise distances
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        # Upper triangle only
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(dists_sq[iu])
        d = np.maximum(d, 1e-15)

        sum_log_d = np.sum(np.log(d))

        # Soft-max via log-sum-exp for smooth max approximation
        max_d = np.max(d)
        scaled = beta * (d - max_d)
        scaled = np.clip(scaled, -500, 500)
        log_sum = np.log(np.sum(np.exp(scaled)))
        soft_max = max_d + log_sum / beta

        # Quadratic penalty for exceeding diameter 2
        excess = np.maximum(soft_max - 2.0, 0.0)
        penalty = 100.0 * excess * excess

        val = -sum_log_d + penalty
        return val

    # Initial guess: Regular octagon
    # Vertices of regular octagon with diameter 2.
    # Center at (1, 0), radius 1.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_unit = np.column_stack((np.cos(angles), np.sin(angles)))
    # Shift so center is at (1,0)
    init_pts = pts_unit + np.array([1.0, 0.0])
    
    # Ensure P0 is (0,0) and P4 is (2,0) for the fixed constraint.
    # In the regular octagon generated above:
    # k=0: (1+1, 0) = (2,0)
    # k=4: (1-1, 0) = (0,0)
    # So we need to swap index 0 and 4 in our assignment, or just permute the init points.
    # Let's create a specific initial configuration that matches the fixed points.
    # We want p[0] = (0,0) and p[4] = (2,0).
    # The regular octagon has points at angles 0, pi/4, ..., 7pi/4.
    # Point at angle pi is (-1, 0) + (1,0) = (0,0). This is k=4.
    # Point at angle 0 is (1, 0) + (1,0) = (2,0). This is k=0.
    # So in the standard ordering, P0 is (2,0) and P4 is (0,0).
    # We fix P0=(0,0) and P4=(2,0).
    # So we swap the roles of the initial points 0 and 4.
    init_pts_swapped = init_pts.copy()
    init_pts_swapped[0], init_pts_swapped[4] = init_pts_swapped[4], init_pts_swapped[0]
    
    # Now init_pts_swapped[0] is (0,0) and init_pts_swapped[4] is (2,0).
    x0 = np.array([init_pts_swapped[i] for i in free_idx]).flatten()
    
    best_x = x0.copy()
    best_obj = objective(x0)

    # Phase 1: Local optimization from regular octagon with multiple methods
    for method in ['BFGS', 'L-BFGS-B']:
        try:
            res = minimize(objective, x0, method=method,
                           options={'maxiter': 2000, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Basin Hopping to escape local minima
    np.random.seed(42)
    
    def step(x):
        return x + np.random.normal(0, 0.3, size=x.shape)

    if time.time() < deadline - 40:
        try:
            # Use the current best as the starting point for basin hopping
            x_start_bh = best_x.copy()
            
            res_bh = basinhopping(objective, x_start_bh, 
                                  minimizer_kwargs={'method': 'BFGS', 'options': {'maxiter': 300, 'gtol': 1e-8, 'disp': False}},
                                  stepfunc=step, T=500.0, stepsize=0.3, niter=80, interval=5, disp=False)
            
            if res_bh.fun < best_obj:
                best_obj = res_bh.fun
                best_x = res_bh.x.copy()
                
        except Exception:
            pass

    # Phase 3: Multi-restart with structured perturbations
    if time.time() < deadline - 40:
        restarts = []
        
        # Perturb the regular octagon with various scales
        for scale in [0.05, 0.1, 0.2, 0.4, 0.8]:
            for trial in range(8):
                noise = np.random.normal(0, scale, size=(n, 2))
                noise[0] = 0
                noise[4] = 0
                perturbed = init_pts_swapped + noise
                x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
                restarts.append(x0_pert)
        
        # Symmetry-breaking perturbations: shift some points radially
        for shift in [0.1, 0.2, 0.3]:
            for pattern in [1, 2]:  # alternating or every other
                perturbed = init_pts_swapped.copy()
                for i in range(1, n):
                    if i != 4:
                        if (i % pattern) == 0:
                            # Radial shift
                            center = np.array([1.0, 0.0])
                            vec = perturbed[i] - center
                            norm = np.linalg.norm(vec)
                            if norm > 1e-10:
                                vec_unit = vec / norm
                                perturbed[i] = perturbed[i] + shift * vec_unit
                x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
                restarts.append(x0_pert)
        
        # Random starts in a reasonable box
        for trial in range(15):
            x0_rand = np.random.uniform(-0.5, 2.5, size=n_dims)
            restarts.append(x0_rand)

        for x0_r in restarts:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(objective, x0_r, method='BFGS',
                               options={'maxiter': 800, 'gtol': 1e-9, 'disp': False})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x.copy()
            except Exception:
                continue

    # Phase 4: High-precision polish pass with multiple methods
    if time.time() < deadline - 15:
        for method in ['L-BFGS-B', 'BFGS']:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, best_x, method=method,
                               options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16, 'disp': False})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

    # Final ultra-precise polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='BFGS',
                           options={'maxiter': 10000, 'gtol': 1e-16, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    p = unpack(best_x)

    # Ensure distinct points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-8:
                p[j] += np.array([1e-6, 0])

    # Scale to ensure diameter is exactly 2 (in case soft-max allowed slight violations or shrinkage)
    max_d = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(p[i] - p[j])
            if d > max_d:
                max_d = d

    if max_d > 0 and abs(max_d - 2.0) > 1e-10:
        scale_factor = 2.0 / max_d
        p = p * scale_factor

    # Return as list of lists of floats
    return [[float(p[i][0]), float(p[i][1])] for i in range(n)]

# EVOLVE-BLOCK-END
