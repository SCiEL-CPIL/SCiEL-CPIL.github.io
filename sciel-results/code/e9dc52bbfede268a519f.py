import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None
        
        # Vectorized circumcenter computation
        a = p[facets[:, 0]]  # (F, 3)
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]
        
        v1 = a - b  # (F, 3)
        v2 = b - c
        n = np.cross(v1, v2)  # (F, 3)
        norm_n = np.linalg.norm(n, axis=1)  # (F,)
        
        # Handle degenerate facets
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        
        return facets, circumcenters, radii

    def compute_forces(p, beta=50.0):
        """Compute smooth forces from soft-max energy + repulsion."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None
        
        # Soft-max energy: E = (1/beta) * log(sum(exp(beta * r_i^2)))
        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        weights = radii * exp_terms / sum_exp  # dE/dr_i

        forces = np.zeros_like(p)
        
        # Attractive forces from facet circumcenters (vectorized via scatter-add)
        # For each facet k, add weights[k] * (cc[k] - p[idx]) for idx in facets[k]
        # Equivalent: for each point i, sum over facets containing i
        # Use np.add.at for efficient scatter
        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                forces[idx] += w * (cc - p[idx])
        
        # Repulsive forces (vectorized)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += 0.1 * forces_rep
        
        return forces, facets, circumcenters, radii

    def optimize_from_start(pts_init, time_budget):
        """Run the full optimization pipeline from a starting configuration."""
        pts = pts_init.copy()
        best_pts_local = pts.copy()
        best_radius_local = float('inf')
        
        # Phase 1: Coarse optimization
        eta = 0.02
        beta = 30.0
        
        for step in range(1500):
            if time.time() > time_budget:
                break
            
            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break
            
            pts += eta * forces
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
            if radii is not None:
                current_radius = np.max(radii)
                if current_radius < best_radius_local:
                    best_radius_local = current_radius
                    best_pts_local = pts.copy()
            
            eta *= 0.998
            beta += 0.5
        
        # Phase 2: L-BFGS-B Polish
        if time.time() < time_budget - 5.0:
            def objective_sum_sq(params):
                X = params.reshape(N, 3)
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                X = X / norms
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                return np.sum(radii**2)
            
            try:
                res = minimize(objective_sum_sq, best_pts_local.flatten(), method='L-BFGS-B',
                               options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
                if res.fun < objective_sum_sq(best_pts_local.flatten()):
                    new_pts = res.x.reshape(N, 3)
                    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                    norms[norms < 1e-12] = 1.0
                    new_pts = new_pts / norms
                    facets, circumcenters, radii = compute_hull_data(new_pts)
                    if radii is not None:
                        current_radius = np.max(radii)
                        if current_radius < best_radius_local:
                            best_radius_local = current_radius
                            best_pts_local = new_pts.copy()
            except Exception:
                pass
        
        # Phase 3: Fine Polish with Gradient Descent
        if time.time() < time_budget - 2.0:
            pts = best_pts_local.copy()
            eta = 0.005
            beta = 100.0
            
            for step in range(500):
                if time.time() > time_budget - 1.0:
                    break
                forces, facets, circumcenters, radii = compute_forces(pts, beta)
                if forces is None:
                    break
                pts += eta * forces
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                pts = pts / norms
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius_local:
                        best_radius_local = current_radius
                        best_pts_local = pts.copy()
                eta *= 0.995
        
        return best_pts_local, best_radius_local

    # Generate multiple initial configurations
    # Config 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Config 2: Fibonacci with offset
    i2 = np.arange(N) + 0.5
    z2 = 1.0 - 2.0 * i2 / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + 0.37)
    r2_ = np.sqrt(1.0 - z2 * z2)
    fib2_pts = np.column_stack([r2_ * np.cos(phi2), r2_ * np.sin(phi2), z2])
    
    # Config 3: Random with good spread (use golden ratio spiral with different phase)
    i3 = np.arange(N) + 0.5
    z3 = 1.0 - 2.0 * i3 / N
    phi3 = np.pi * (1.0 + np.sqrt(5.0)) * (i3 + 0.71)
    r3 = np.sqrt(1.0 - z3 * z3)
    fib3_pts = np.column_stack([r3 * np.cos(phi3), r3 * np.sin(phi3), z3])
    
    initial_configs = [fib_pts, fib2_pts, fib3_pts]
    
    best_pts = fib_pts.copy()
    best_radius = float('inf')
    
    # Allocate time budget across restarts
    # First restart gets more time, subsequent ones get less
    if len(initial_configs) >= 3:
        budgets = [deadline - 30.0, deadline - 15.0, deadline - 2.0]
    else:
        budgets = [deadline - 2.0]
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 2.0:
            break
        
        # Give each restart a portion of remaining time
        if idx == 0:
            tb = deadline - 25.0  # First restart: most time
        elif idx == 1:
            tb = deadline - 10.0  # Second restart
        else:
            tb = deadline - 2.0   # Final restart
        
        if time.time() > tb - 1.0:
            continue
        
        pts_res, rad_res = optimize_from_start(init_pts, tb)
        
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
        
        # If first restart did well, use its result as seed for subsequent
        if idx == 0 and rad_res < best_radius:
            # Don't modify initial_configs, just note the best
            pass
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final L-BFGS-B polish on the best result
    if time.time() < deadline - 2.0:
        def objective_final(params):
            X = params.reshape(N, 3)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            X = X / norms
            facets, circumcenters, radii = compute_hull_data(X)
            if facets is None:
                return 1e10
            return np.sum(radii**2)
        
        try:
            res = minimize(objective_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
            if res.fun < objective_final(best_pts.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
