import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2

def pairwise_distances_sq(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective_full(x):
    """Objective for full 32D parameter space."""
    points = x.reshape(-1, D)
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals <= 0):
        return 1e10
    
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    return d_max_sq / d_min_sq

def objective_concentric(params):
    """
    Objective for concentric circle parameterization.
    params: [r0, r1, r2, r3, theta0, theta1, theta2, theta3]
    4 rings of 4 points each, with rotational symmetry.
    """
    n_rings = 4
    pts_per_ring = 4
    
    radii = params[:n_rings]
    phases = params[n_rings:]
    
    points = []
    for i in range(n_rings):
        r = radii[i]
        theta0 = phases[i]
        for j in range(pts_per_ring):
            theta = theta0 + 2 * np.pi * j / pts_per_ring
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
    
    points = np.array(points)
    return objective_full(points.flatten())

def construct():
    # --- Strategy 1: Concentric circles with rotational symmetry (8 params) ---
    # 4 rings of 4 points each
    # params: [r0, r1, r2, r3, theta0, theta1, theta2, theta3]
    
    bounds_concentric = [
        (0.1, 3.0),  # r0
        (0.1, 3.0),  # r1
        (0.1, 3.0),  # r2
        (0.1, 3.0),  # r3
        (0.0, 2*np.pi),  # theta0
        (0.0, 2*np.pi),  # theta1
        (0.0, 2*np.pi),  # theta2
        (0.0, 2*np.pi),  # theta3
    ]
    
    best_score = np.inf
    best_points = None
    
    # Initial seeds for concentric parameterization
    np.random.seed(42)
    seed_params_list = []
    
    # Seed A: equally spaced radii, zero phases
    seed_params_list.append(np.array([0.5, 1.0, 1.5, 2.0, 0, 0, 0, 0]))
    
    # Seed B: equally spaced radii, staggered phases
    seed_params_list.append(np.array([0.5, 1.0, 1.5, 2.0, 0, np.pi/4, np.pi/2, 3*np.pi/4]))
    
    # Seed C: random
    for _ in range(10):
        r = np.sort(np.random.uniform(0.2, 2.5, 4))
        th = np.random.uniform(0, 2*np.pi, 4)
        seed_params_list.append(np.concatenate([r, th]))
    
    # Evaluate initial seeds
    for sp in seed_params_list:
        s = objective_concentric(sp)
        if s < best_score:
            best_score = s
            # Reconstruct points
            n_rings = 4
            pts_per_ring = 4
            radii = sp[:n_rings]
            phases = sp[n_rings:]
            points = []
            for i in range(n_rings):
                r = radii[i]
                theta0 = phases[i]
                for j in range(pts_per_ring):
                    theta = theta0 + 2 * np.pi * j / pts_per_ring
                    points.append([r * np.cos(theta), r * np.sin(theta)])
            best_points = np.array(points)
    
    # Differential evolution on 8D concentric parameter space
    try:
        de_result = differential_evolution(
            objective_concentric,
            bounds_concentric,
            seed=42,
            maxiter=200,
            popsize=25,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init='latinhypercube'
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            n_rings = 4
            pts_per_ring = 4
            radii = de_result.x[:n_rings]
            phases = de_result.x[n_rings:]
            points = []
            for i in range(n_rings):
                r = radii[i]
                theta0 = phases[i]
                for j in range(pts_per_ring):
                    theta = theta0 + 2 * np.pi * j / pts_per_ring
                    points.append([r * np.cos(theta), r * np.sin(theta)])
            best_points = np.array(points)
    except Exception:
        pass
    
    # --- Strategy 2: Full 32D optimization from the best concentric result ---
    if best_points is not None:
        bounds_full = [(-3.0, 3.0)] * (N * D)
        
        try:
            de_full = differential_evolution(
                objective_full,
                bounds_full,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            
            if de_full.fun < best_score:
                best_score = de_full.fun
                best_points = de_full.x.reshape(-1, D)
        except Exception:
            pass
        
        # Local polish with Powell
        try:
            res = minimize(
                objective_full,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(-1, D)
        except Exception:
            pass
        
        # Final Nelder-Mead polish
        try:
            res = minimize(
                objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(-1, D)
        except Exception:
            pass
    
    # Ensure distinct points
    if best_points is None:
        # Fallback: simple grid
        best_points = np.array([[float(i % 4), float(i // 4)] for i in range(N)])
    
    d_sq = pairwise_distances_sq(best_points)
    mask = ~np.eye(N, dtype=bool)
    if np.any(d_sq[mask] < 1e-10):
        # Nudge overlapping points
        for i in range(N):
            for j in range(i+1, N):
                if d_sq[i, j] < 1e-10:
                    best_points[j, 0] += 1e-5
                    best_points[j, 1] += 1e-5
                    d_sq = pairwise_distances_sq(best_points)
    
    return best_points
