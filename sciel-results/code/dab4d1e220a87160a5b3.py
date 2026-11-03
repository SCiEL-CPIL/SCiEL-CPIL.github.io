import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Phase 1: Generate a good initial configuration using hexagonal lattice
    def gen_hex_lattice(spacing):
        points = []
        range_val = int(1.5 / spacing) + 2
        for j in range(-range_val, range_val + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > 1.0:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = spacing / 2.0
            for i in range(-range_val, range_val + 1):
                x = i * spacing + offset
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        if len(points) < N:
            return None
        points = np.array(points)
        # Sort by distance from origin, keep closest N
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        return points[indices[:N]]

    # Try multiple spacings to get a good initial configuration
    x = None
    for sp in [0.11, 0.10, 0.09, 0.08]:
        x = gen_hex_lattice(sp)
        if x is not None:
            break
    if x is None:
        x = np.random.randn(N, 2) * 0.3

    # Helper to compute the radius (score)
    def calc_radius(pts):
        tree = cKDTree(pts)
        # Get nearest neighbor distances
        d, _ = tree.query(pts, k=2)
        min_pair_dist = np.min(d[:, 1]) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair_dist, min_boundary)

    best_x = x.copy()
    best_r = calc_radius(x)

    # Phase 2: Gradient-based optimization using a soft-min approach
    # We maximize r = min(min_pair_dist/2, min_boundary)
    # Using a soft approximation: r_soft = -1/beta * log(sum(exp(-beta * d_i)))
    # where d_i are the "distances" (pair distances / 2 and boundary distances)

    def soft_min_value(pts, beta):
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        pair_dists = d[:, 1] / 2.0
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        all_dists = np.concatenate([pair_dists, boundary_dists])
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # For numerical stability
        min_d = np.min(all_dists)
        exps = np.exp(-beta * (all_dists - min_d))
        return min_d - np.log(np.sum(exps)) / beta

    def soft_min_gradient(pts, beta):
        tree = cKDTree(pts)
        d, idx = tree.query(pts, k=2)
        pair_dists = d[:, 1] / 2.0
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        # Weights for soft min
        all_dists = np.concatenate([pair_dists, boundary_dists])
        min_d = np.min(all_dists)
        exps = np.exp(-beta * (all_dists - min_d))
        weights = exps / np.sum(exps)
        
        pair_weights = weights[:N]
        bnd_weights = weights[N:]
        
        grads = np.zeros_like(pts)
        
        # Pair contributions
        for i in range(N):
            j = idx[i, 1]
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                unit = diff / dist
                # Gradient of -d_i/2 with respect to pts[i] is -unit/2
                # But we have weight * (-d_i/2), so grad contribution is -weight * unit / 2
                grads[i] += -pair_weights[i] * unit / 2.0 * beta
                # Symmetric contribution to pts[j]
                grads[j] += pair_weights[i] * unit / 2.0 * beta
        
        # Boundary contributions
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 1e-10:
                unit = pts[i] / norms[i]
                # Gradient of -(1 - |p_i|) = |p_i| - 1, so grad is unit
                grads[i] += bnd_weights[i] * unit * beta
        
        return grads

    # Simulated annealing with gradient steps
    x_curr = x.copy()
    step_size = 0.02
    beta = 5.0
    T = 0.01  # Temperature for annealing

    while time.time() < deadline - 30.0:
        grad = soft_min_gradient(x_curr, beta)
        
        if not np.all(np.isfinite(grad)):
            beta *= 0.5
            continue
        
        # Gradient ascent step (we want to maximize the soft min)
        x_new = x_curr + step_size * grad
        
        # Project back into unit circle
        norms = np.linalg.norm(x_new, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            x_new[mask] /= norms[mask, np.newaxis]
        
        # Accept with probability based on improvement
        r_new = calc_radius(x_new)
        r_curr = calc_radius(x_curr)
        
        if r_new >= r_curr or np.random.random() < np.exp((r_new - r_curr) / T):
            x_curr = x_new
            if r_new > best_r:
                best_r = r_new
                best_x = x_curr.copy()
            # Slowly increase beta (sharpen the soft min)
            beta *= 1.02
            step_size = min(step_size * 1.01, 0.05)
        else:
            step_size *= 0.95
        
        # Anneal temperature
        T *= 0.995
        if T < 1e-4:
            T = 1e-4

    # Phase 3: Local refinement using Nelder-Mead on critical points
    def identify_critical(pts, r_target):
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        pair_dists = d[:, 1] / 2.0
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        crit = set()
        # Points with tight pair constraints
        tight_pairs = np.where(pair_dists < r_target * 1.1)[0]
        for i in tight_pairs:
            j = tree.query(pts[i], k=2)[1][0]
            crit.add(int(i))
            crit.add(int(j))
        
        # Points with tight boundary constraints
        tight_bnd = np.where(boundary_dists < r_target * 1.1)[0]
        for i in tight_bnd:
            crit.add(int(i))
        
        return list(crit)

    current_x = best_x.copy()
    current_r = best_r

    for round_num in range(8):
        if time.time() > deadline - 5.0:
            break
        
        crit_indices = identify_critical(current_x, current_r)
        if not crit_indices:
            break
        
        # Limit to at most 30 points for efficiency
        if len(crit_indices) > 30:
            # Sort by tightness
            tree = cKDTree(current_x)
            d, _ = tree.query(current_x, k=2)
            pair_dists = d[:, 1] / 2.0
            boundary_dists = 1.0 - np.linalg.norm(current_x, axis=1)
            tightness = np.array([min(pair_dists[i], boundary_dists[i]) for i in crit_indices])
            order = np.argsort(tightness)
            crit_indices = [crit_indices[i] for i in order[:30]]
        
        active_indices = np.array(crit_indices)
        
        def obj_local(x_flat):
            pts = current_x.copy()
            pts[active_indices] = x_flat.reshape(-1, 2)
            
            # Penalty for going outside
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms > 1.0):
                return 1e6 + np.sum(norms[norms > 1.0] - 1.0)
            
            tree = cKDTree(pts)
            d, _ = tree.query(pts, k=2)
            min_pair = np.min(d[:, 1]) / 2.0
            min_bnd = np.min(1.0 - norms)
            return -min(min_pair, min_bnd)
        
        x0 = current_x[active_indices].flatten()
        
        try:
            res = minimize(
                obj_local,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res.fun < obj_local(x0):
                new_x = current_x.copy()
                new_x[active_indices] = res.x.reshape(-1, 2)
                
                if np.all(np.linalg.norm(new_x, axis=1) <= 1.0 + 1e-9):
                    # Project any slightly outside points back
                    norms = np.linalg.norm(new_x, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        new_x[mask] /= norms[mask, np.newaxis]
                    
                    r_new = calc_radius(new_x)
                    if r_new > current_r:
                        current_r = r_new
                        current_x = new_x
                        if current_r > best_r:
                            best_r = current_r
                            best_x = current_x.copy()
        except Exception:
            continue

    # Phase 4: Final polish with small random perturbations
    if time.time() < deadline - 2.0:
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            noise = np.random.randn(N, 2) * 0.003
            x_pert = best_x + noise
            norms = np.linalg.norm(x_pert, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                x_pert[mask] /= norms[mask, np.newaxis]
            
            r_pert = calc_radius(x_pert)
            if r_pert > best_r:
                best_r = r_pert
                best_x = x_pert

    # Final validation
    norms = np.linalg.norm(best_x, axis=1)
    if np.any(norms > 1.0):
        best_x[norms > 1.0] /= norms[norms > 1.0, np.newaxis]
    
    return best_x
