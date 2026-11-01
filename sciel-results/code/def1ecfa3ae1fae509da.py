import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin

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

    def get_max_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def soft_max_objective(params, beta=200.0):
        """
        Soft-max surrogate: approximates max(radii) via log-sum-exp.
        As beta -> inf, this approaches the true max.
        """
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        
        # Stable log-sum-exp
        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def repulsion_relax(pts, n_iters=800, alpha_start=0.08):
        """
        Repulsion relaxation with adaptive step size (backtracking line search).
        Minimize sum 1/||p_i - p_j||^2 via gradient descent on sphere.
        """
        pts = pts.copy()
        alpha = alpha_start
        
        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)
        
        current_energy = repulsion_energy(pts)
        
        for it in range(n_iters):
            # Calculate pairwise differences
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            
            # Avoid division by zero
            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
            
            # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            # Project onto tangent plane: F_tan = F - (F . p) p
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts
            
            # Adaptive step size with backtracking
            step = alpha
            improved = False
            for _ in range(10):
                new_pts = pts + step * forces_tan
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                # If no improvement, try smaller base alpha
                alpha *= 0.8
            else:
                alpha *= 1.01  # Slightly increase if successful
            
            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-6)
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        return pts

    def facet_local_search(pts, n_iters=100):
        """
        Facet-based local search: directly minimize the maximum circumradius
        by moving vertices of the largest facets toward their circumcenters.
        This is more effective than Lloyd's for covering radius optimization.
        """
        pts = pts.copy()
        
        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            # Find facets that are close to the maximum radius (top 20%)
            threshold = max_rad * 0.8
            large_facets = np.where(radii > threshold)[0]
            
            if len(large_facets) == 0:
                break
            
            # Compute weighted updates: move vertices toward circumcenters
            # of large facets, weighted by how much larger they are
            updates = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for f in large_facets:
                facet_pts = facets[f]
                cc = circumcenters[f]
                rad = radii[f]
                if rad < 1e-12:
                    continue
                # Weight: emphasize facets closer to the max
                weight = ((rad - threshold) / (max_rad - threshold + 1e-15)) ** 2
                weight = max(weight, 0.1)  # Minimum weight
                for idx in facet_pts:
                    updates[idx] += weight * (cc - pts[idx])
                    counts[idx] += weight
            
            # Apply updates where counts > 0
            mask = counts > 1e-10
            if np.any(mask):
                # Adaptive step size based on the gap between max and mean
                mean_rad = np.mean(radii)
                gap = max_rad - mean_rad
                step_size = 0.08 * (gap / (mean_rad + 1e-15))
                step_size = min(step_size, 0.15)  # Cap step size
                step_size = max(step_size, 0.01)  # Minimum step
                
                # Backtracking line search
                best_pts_trial = pts.copy()
                best_rad_trial = max_rad
                for trial_step in [step_size, step_size * 0.5, step_size * 0.25, step_size * 0.125]:
                    trial_pts = pts.copy()
                    trial_pts[mask] += trial_step * updates[mask] / counts[mask, np.newaxis]
                    # Normalize
                    norms = np.linalg.norm(trial_pts, axis=1, keepdims=True)
                    norms[norms < 1e-12] = 1.0
                    trial_pts = trial_pts / norms
                    
                    trial_rad = get_max_radius(trial_pts)
                    if trial_rad < best_rad_trial:
                        best_rad_trial = trial_rad
                        best_pts_trial = trial_pts.copy()
                
                if best_rad_trial < max_rad:
                    pts = best_pts_trial
                else:
                    # No improvement, reduce aggressiveness
                    break
            
            # Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
            if time.time() > deadline - 5.0:
                break
        
        return pts

    def pressure_relax(pts, n_iters=300, alpha_start=0.05):
        """
        Pressure-minimization dynamic: move vertices along the negative gradient
        of the convex hull volume. For a triangulated polyhedron, the gradient of
        volume w.r.t. vertex i is (1/3) * sum over adjacent facets of (area_f * normal_f)
        which equals (1/3) * sum of cross products. Moving against this gradient
        (i.e., toward the circumcenters of adjacent facets) reduces the hull volume,
        making the configuration more "compact" and reducing the covering radius.
        """
        pts = pts.copy()
        alpha = alpha_start
        
        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            # Compute the volume gradient: for each vertex, sum of facet normals weighted by area
            # For a triangle with vertices a, b, c, the contribution to the volume gradient
            # at each vertex is (1/6) * (a + b + c) * |cross(b-a, c-a)| / 3 ... 
            # Actually, dV/dp_i for vertex i in facet (i, j, k) = (1/6) * cross(p_j - p_i, p_k - p_i)
            # But we want to MINIMIZE volume, so we move in the direction -dV/dp_i
            # However, for covering radius, we actually want to move toward circumcenters
            # which is equivalent to the "pressure" force.
            
            # Compute pressure forces: for each vertex, sum of (circumcenter - vertex) weighted by facet area
            forces = np.zeros_like(pts)
            facet_areas = np.zeros(len(facets))
            
            for f in range(len(facets)):
                a = pts[facets[f, 0]]
                b = pts[facets[f, 1]]
                c = pts[facets[f, 2]]
                v1 = b - a
                v2 = c - a
                normal = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(normal)
                facet_areas[f] = area
                
                cc = circumcenters[f]
                # Force on each vertex: pull toward circumcenter, weighted by area
                for idx in facets[f]:
                    forces[idx] += area * (cc - pts[idx])
            
            # Project onto tangent plane
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts
            
            # Adaptive step size with backtracking
            step = alpha
            improved = False
            for _ in range(8):
                new_pts = pts + step * forces_tan
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                
                new_rad = get_max_radius(new_pts)
                current_rad = get_max_radius(pts)
                
                if new_rad < current_rad:
                    pts = new_pts
                    improved = True
                    break
                step *= 0.5
            
            if improved:
                alpha *= 1.02
            else:
                alpha *= 0.85
            
            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-7)
            
            if time.time() > deadline - 5.0:
                break
        
        return pts

    def epigraph_objective(params, beta=500.0):
        """
        Epigraph-style smooth objective: minimize t + (1/beta) * sum(exp(beta*(r_i - t)))
        This is equivalent to a soft-max with an explicit auxiliary variable t.
        """
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        
        # The epigraph formulation with t = max(radii) is equivalent to soft-max
        # But we can also use it as a constraint: minimize t s.t. r_i <= t
        # For smoothness, use: t + (1/beta) * sum(exp(beta*(r_i - t)))
        t = np.max(radii)
        if t < 1e-15:
            return t
        
        shifted = beta * (radii - t)
        shifted = np.clip(shifted, -50, 50)
        penalty = np.sum(np.exp(shifted)) / beta
        return t + penalty

    def icosahedral_init():
        """
        Generate an icosahedral geodesic dome with 50 vertices.
        Start with icosahedron (12 vertices), subdivide faces, and project to sphere.
        """
        # Icosahedron vertices
        t = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        
        # Icosahedron faces (20 triangles)
        ico_faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        
        # Subdivide each face into 4 smaller triangles (midpoint subdivision)
        # This gives 12 + 20*3 = 72 vertices, then we need to reduce to 50
        # Alternative: use 1 subdivision level and select 50 best points
        
        # For simplicity, use the icosahedron vertices plus midpoints of edges
        # and select 50 points with good spread
        
        # Start with icosahedron (12 points)
        points = [ico[i] for i in range(12)]
        
        # Add midpoints of all edges
        edge_mids = set()
        for face in ico_faces:
            for i in range(3):
                a, b = face[i], face[(i+1) % 3]
                if a > b:
                    a, b = b, a
                if (a, b) not in edge_mids:
                    edge_mids.add((a, b))
                    mid = (ico[a] + ico[b]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    points.append(mid)
        
        # We now have 12 + 30 = 42 points. Add face centers to get 62, then select 50
        face_centers = []
        for face in ico_faces:
            center = np.mean(ico[face], axis=0)
            center = center / np.linalg.norm(center)
            face_centers.append(center)
        
        # Combine all candidates
        all_candidates = np.array(points + face_centers)
        
        # Select 50 points with maximum minimum distance (greedy)
        selected = [all_candidates[0]]
        for _ in range(N - 1):
            min_dists = []
            for cand in all_candidates:
                min_dist = min(np.linalg.norm(cand - s) for s in selected)
                min_dists.append(min_dist)
            min_dists = np.array(min_dists)
            # Exclude already selected
            for s in selected:
                idx = np.where(np.all(all_candidates == s, axis=1))[0]
                if len(idx) > 0:
                    min_dists[idx] = -1
            best_idx = np.argmax(min_dists)
            selected.append(all_candidates[best_idx])
        
        return np.array(selected)

    def optimize_from_start(pts_init, obj_func, maxiter=80):
        """Run L-BFGS-B optimization from a starting configuration."""
        best_pts_local = pts_init.copy()
        best_radius_local = get_max_radius(pts_init)
        
        # Normalize initial points
        norms = np.linalg.norm(pts_init, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts_init = pts_init / norms
        
        try:
            res = minimize(obj_func, pts_init.flatten(), method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200})
            
            if res.fun < obj_func(pts_init.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius_local:
                    best_radius_local = current_radius
                    best_pts_local = new_pts.copy()
        except Exception:
            pass
            
        return best_pts_local, best_radius_local

    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initial_configs.append(fib_pts)
    
    # 2. Fibonacci with different phase offsets
    for phase in [0.37, 0.618, 0.25]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + phase)
        r2 = np.sqrt(1.0 - z2 * z2)
        fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initial_configs.append(fib2_pts)
    
    # 3. Icosahedral geodesic dome initialization
    try:
        ico_pts = icosahedral_init()
        initial_configs.append(ico_pts)
    except Exception:
        pass
    
    # 4. Random points (seeded for reproducibility)
    for seed_offset in range(3):
        np.random.seed(42 + seed_offset * 100)
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initial_configs.append(pts)
        
    # 5. Greedy initialization - place points to maximize minimum distance
    def greedy_init(seed_val):
        np.random.seed(seed_val)
        pts = [np.random.randn(3)]
        norms = np.linalg.norm(pts[0])
        pts[0] = pts[0] / norms
        for _ in range(N - 1):
            candidates = np.random.randn(2000, 3)
            norms = np.linalg.norm(candidates, axis=1, keepdims=True)
            candidates = candidates / norms
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)
    
    initial_configs.append(greedy_init(101))
    initial_configs.append(greedy_init(202))
    initial_configs.append(greedy_init(303))

    best_pts = initial_configs[0].copy()
    best_radius = get_max_radius(best_pts)
    
    # Phase 1: Repulsion relaxation to get good spread
    for idx, init_pts in enumerate(initial_configs[:5]):  # First 5 configs
        if time.time() > deadline - 50.0:
            break
        
        relaxed = repulsion_relax(init_pts, n_iters=200, alpha_start=0.03)
        rad_relaxed = get_max_radius(relaxed)
        if rad_relaxed < best_radius:
            best_radius = rad_relaxed
            best_pts = relaxed.copy()
    
    # Phase 2: Annealing soft-max optimization with multiple betas
    beta_schedule = [50, 100, 200, 400, 800, 1500]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30.0:
            break
        
        def obj_sm(params, _beta=beta):
            return soft_max_objective(params, beta=_beta)
        
        # Try from current best and a couple of other starts
        starts_to_try = [best_pts]
        if len(initial_configs) > 1:
            starts_to_try.append(initial_configs[1])
        if len(initial_configs) > 2:
            starts_to_try.append(initial_configs[2])
        
        for start_idx, init_pts in enumerate(starts_to_try):
            if time.time() > deadline - 30.0:
                break
            
            maxiter = 40 if start_idx == 0 else 30
            pts_res, rad_res = optimize_from_start(init_pts, obj_sm, maxiter=maxiter)
            if rad_res < best_radius:
                best_radius = rad_res
                best_pts = pts_res.copy()
    
    # Phase 3: Facet-based local search and pressure relaxation
    if time.time() < deadline - 25.0:
        facet_pts = facet_local_search(best_pts, n_iters=60)
        rad_facet = get_max_radius(facet_pts)
        if rad_facet < best_radius:
            best_radius = rad_facet
            best_pts = facet_pts.copy()
        
        # Pressure relaxation: move vertices against hull volume gradient
        if time.time() < deadline - 20.0:
            pressure_pts = pressure_relax(best_pts, n_iters=200, alpha_start=0.04)
            rad_pressure = get_max_radius(pressure_pts)
            if rad_pressure < best_radius:
                best_radius = rad_pressure
                best_pts = pressure_pts.copy()
        
        # Another round of soft-max after pressure relaxation
        if time.time() < deadline - 15.0:
            def obj_sm_mid(params, _beta=300.0):
                return soft_max_objective(params, beta=_beta)
            
            pts_res, rad_res = optimize_from_start(best_pts, obj_sm_mid, maxiter=40)
            if rad_res < best_radius:
                best_radius = rad_res
                best_pts = pts_res.copy()
        
        # Another facet search round
        if time.time() < deadline - 10.0:
            facet_pts2 = facet_local_search(best_pts, n_iters=40)
            rad_facet2 = get_max_radius(facet_pts2)
            if rad_facet2 < best_radius:
                best_radius = rad_facet2
                best_pts = facet_pts2.copy()
        
        # Another pressure relaxation round
        if time.time() < deadline - 8.0:
            pressure_pts2 = pressure_relax(best_pts, n_iters=150, alpha_start=0.03)
            rad_pressure2 = get_max_radius(pressure_pts2)
            if rad_pressure2 < best_radius:
                best_radius = rad_pressure2
                best_pts = pressure_pts2.copy()
    
    # Phase 4: Final high-precision polish with epigraph objective
    if time.time() < deadline - 5.0:
        def obj_final(params, _beta=2000.0):
            return epigraph_objective(params, beta=_beta)
        
        try:
            res = minimize(obj_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 60, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 400})
            if res.fun < obj_final(best_pts.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass
        
        # Final facet-based polish
        if time.time() < deadline - 3.0:
            facet_pts3 = facet_local_search(best_pts, n_iters=25)
            rad_facet3 = get_max_radius(facet_pts3)
            if rad_facet3 < best_radius:
                best_radius = rad_facet3
                best_pts = facet_pts3.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
