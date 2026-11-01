"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def _time_left():
        return deadline - time.time()

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    def compute_facet_data(pts):
        """
        Compute all facet circumcenters and their angular radii (vectorized).
        Returns: (radii_array, centers_array, simplices_array)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.array([]), np.array([]), np.array([])
        
        simplices = hull.simplices  # shape (F, 3)
        F = len(simplices)
        
        v1 = pts[simplices[:, 0]]
        v2 = pts[simplices[:, 1]]
        v3 = pts[simplices[:, 2]]
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        valid = (norms[:, 0] > 1e-12)
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([])
        
        normals = normals[valid]
        norms = norms[valid]
        v1 = v1[valid]
        simplices = simplices[valid]
        
        centers = normals / norms
        dots = np.sum(centers * v1, axis=1)
        centers[dots < 0] *= -1
        
        diffs = centers - v1
        d = np.linalg.norm(diffs, axis=1)
        radii = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
        
        return radii, centers, simplices

    def compute_covering_radius(pts):
        """Compute the covering radius in radians and the worst-case circumcenter."""
        radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            return np.pi, np.array([0.0, 0.0, 1.0])
        
        max_idx = np.argmax(radii)
        return radii[max_idx], centers[max_idx]

    def soft_max_radius_and_grad(pts, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points using analytical gradients.
        Returns: (soft_radius, gradient_array)
        """
        radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            return np.pi, np.zeros_like(pts)
        
        max_r = np.max(radii)
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)
        
        grads = np.zeros_like(pts)
        F = len(radii)
        
        for i in range(F):
            w = weights[i]
            if w < 1e-15:
                continue
            r = radii[i]
            if r < 1e-12:
                continue
            
            C = centers[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            
            simplex = simplices[i]
            p_pts = pts[simplex]
            
            grad_contrib = -C / sin_r
            
            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps
            
            for k in range(3):
                grads[simplex[k]] += w * tangential_grads[k]
        
        return soft_radius, grads

    def lloyd_iteration(pts, iterations=30):
        """
        Lloyd's algorithm for k-means on the sphere:
        1. Compute Voronoi cells (via convex hull facets)
        2. Move each point to the centroid of its Voronoi cell
        This tends to produce more uniform distributions.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if _time_left() < 10:
                break
            
            radii, centers, simplices = compute_facet_data(pts)
            if len(radii) == 0:
                break
            
            # For each point, find the centroid of its Voronoi cell
            # The Voronoi cell of point i is the intersection of half-spaces
            # defined by the facets containing i.
            # Approximation: move each point toward the average of its adjacent facet centers
            new_pts = pts.copy()
            for i in range(N):
                # Find all facets containing point i
                adjacent_centers = []
                for f in range(len(simplices)):
                    if i in simplices[f]:
                        adjacent_centers.append(centers[f])
                
                if len(adjacent_centers) > 0:
                    centroid = np.mean(adjacent_centers, axis=0)
                    # Move toward centroid but stay on sphere
                    new_pts[i] = pts[i] + 0.5 * (centroid - pts[i])
                    new_pts[i] /= np.linalg.norm(new_pts[i])
            
            # Check improvement
            old_radius = compute_covering_radius(pts)[0]
            new_radius = compute_covering_radius(new_pts)[0]
            
            if new_radius < old_radius:
                pts = new_pts
            else:
                # Try smaller step
                new_pts = pts.copy()
                for i in range(N):
                    adjacent_centers = []
                    for f in range(len(simplices)):
                        if i in simplices[f]:
                            adjacent_centers.append(centers[f])
                    
                    if len(adjacent_centers) > 0:
                        centroid = np.mean(adjacent_centers, axis=0)
                        new_pts[i] = pts[i] + 0.2 * (centroid - pts[i])
                        new_pts[i] /= np.linalg.norm(new_pts[i])
                
                new_radius = compute_covering_radius(new_pts)[0]
                if new_radius < old_radius:
                    pts = new_pts
                else:
                    break
        
        return pts

    def repulsion_relax(pts, iterations=100):
        """
        Repulsion-based relaxation: each point is pushed away from its neighbors.
        This helps escape local minima by spreading points more evenly.
        """
        pts = pts.copy()
        for it in range(iterations):
            if _time_left() < 10:
                break
            
            # Compute pairwise distances
            dots = np.dot(pts, pts.T)
            np.fill_diagonal(dots, 1.0)
            
            # Repulsion force: each point is pushed away from close neighbors
            forces = np.zeros_like(pts)
            for i in range(N):
                # Only consider points within some angular distance
                mask = dots[i] < 0.95  # roughly < 18 degrees
                if np.any(mask):
                    for j in np.where(mask)[0]:
                        # Force direction: away from j
                        force_dir = pts[i] - dots[i, j] * pts[j]
                        force_norm = np.linalg.norm(force_dir)
                        if force_norm > 1e-10:
                            # Strength decreases with distance (stronger when closer)
                            strength = (0.95 - dots[i, j]) ** 2
                            forces[i] += strength * force_dir / force_norm
            
            # Apply forces with adaptive step size
            step = 0.01 * (0.95 ** (it // 20))
            for i in range(N):
                if np.linalg.norm(forces[i]) > 1e-10:
                    pts[i] += step * forces[i]
                    pts[i] /= np.linalg.norm(pts[i])
            
            # Periodically check and keep best
            if it % 20 == 0:
                radius = compute_covering_radius(pts)[0]
                if radius > np.pi * 0.6:  # got worse
                    break
        
        return pts

    def simulated_annealing(pts, initial_temp=0.3, final_temp=0.01, iterations=2000):
        """
        Simulated annealing for global exploration.
        Accepts worse solutions with probability exp(-delta/T) to escape local minima.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_radius = compute_covering_radius(pts)[0]
        
        n_accepted = 0
        for it in range(iterations):
            if _time_left() < 15:
                break
            
            # Temperature schedule (geometric cooling)
            T = initial_temp * (final_temp / initial_temp) ** (it / iterations)
            
            # Random perturbation: move 1-3 random points
            n_move = np.random.randint(1, 4)
            indices = np.random.choice(N, size=n_move, replace=False)
            
            new_pts = pts.copy()
            for idx in indices:
                # Random direction on tangent space
                rand_vec = np.random.randn(3)
                radial_comp = np.dot(rand_vec, pts[idx])
                rand_vec = rand_vec - radial_comp * pts[idx]
                rand_norm = np.linalg.norm(rand_vec)
                if rand_norm > 1e-10:
                    rand_vec /= rand_norm
                    # Step size proportional to temperature
                    step = T * 0.5
                    new_pts[idx] = pts[idx] + step * rand_vec
                    new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            new_radius = compute_covering_radius(new_pts)[0]
            delta = new_radius - compute_covering_radius(pts)[0]
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                pts = new_pts
                n_accepted += 1
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
            
            # Occasionally try a larger perturbation
            if it % 50 == 0 and T > 0.05:
                n_move = np.random.randint(3, 8)
                indices = np.random.choice(N, size=n_move, replace=False)
                new_pts = pts.copy()
                for idx in indices:
                    rand_vec = np.random.randn(3)
                    radial_comp = np.dot(rand_vec, pts[idx])
                    rand_vec = rand_vec - radial_comp * pts[idx]
                    rand_norm = np.linalg.norm(rand_vec)
                    if rand_norm > 1e-10:
                        rand_vec /= rand_norm
                        step = T * 1.0
                        new_pts[idx] = pts[idx] + step * rand_vec
                        new_pts[idx] /= np.linalg.norm(new_pts[idx])
                
                new_radius = compute_covering_radius(new_pts)[0]
                delta = new_radius - compute_covering_radius(pts)[0]
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    pts = new_pts
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_pts = new_pts.copy()
        
        return best_pts, best_radius

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        """
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if _time_left() < 10:
                break

            radius, worst_center = compute_covering_radius(pts)

            if radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = radius

            try:
                hull = ConvexHull(pts)
            except:
                break

            facet_data = []
            for simplex in hull.simplices:
                v1 = pts[simplex[0]]
                v2 = pts[simplex[1]]
                v3 = pts[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                facet_data.append((c, simplex))

            if not facet_data:
                break

            target_indices_set = set()
            for c, simplex in facet_data:
                if np.dot(c, worst_center) > 0.999:
                    for idx in simplex:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = pts[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    def lbfgs_stage(pts, beta, max_iter=300):
        """
        Run L-BFGS-B with soft-max covering radius at given beta.
        """
        if _time_left() < 20:
            return pts
        
        x0 = pts.flatten()
        
        def objective_and_grad(x, b=beta):
            if _time_left() < 15:
                return 1e10, np.zeros_like(x)
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            pts = pts / norms
            soft_r, grads = soft_max_radius_and_grad(pts, beta=b)
            return soft_r, grads.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'maxfun': max_iter*3, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 200}
            )
            
            if result.success or result.fun < objective_and_grad(x0)[0]:
                lbfgs_pts = project_to_sphere(result.x.reshape(N, 3))
                return lbfgs_pts
        except Exception:
            pass
        
        return pts

    # Phase 1: Generate multiple diverse initial configurations
    initial_configs = []
    
    # Fibonacci spiral
    initial_configs.append(fibonacci_sphere(N))
    
    # Fibonacci with different offset
    initial_configs.append(fibonacci_sphere(N, offset=0.3))
    
    # Random configuration (seeded for reproducibility)
    np.random.seed(123)
    rand_pts = np.random.randn(N, 3)
    rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
    initial_configs.append(rand_pts)
    
    # Another random with different seed
    np.random.seed(456)
    rand_pts2 = np.random.randn(N, 3)
    rand_pts2 /= np.linalg.norm(rand_pts2, axis=1, keepdims=True)
    initial_configs.append(rand_pts2)
    
    # Evaluate all initial configs
    best = None
    best_radius = np.pi
    for config in initial_configs:
        radius, _ = compute_covering_radius(config)
        if radius < best_radius:
            best_radius = radius
            best = config.copy()
    
    current = best.copy()

    # Phase 2: Lloyd's algorithm for better initial distribution
    if _time_left() > 60:
        lloyd_pts = lloyd_iteration(current, iterations=30)
        lloyd_radius, _ = compute_covering_radius(lloyd_pts)
        if lloyd_radius < best_radius:
            best_radius = lloyd_radius
            best = lloyd_pts.copy()
            current = lloyd_pts.copy()

    # Phase 3: Repulsion relaxation
    if _time_left() > 50:
        repuls_pts = repulsion_relax(current, iterations=80)
        repuls_radius, _ = compute_covering_radius(repuls_pts)
        if repuls_radius < best_radius:
            best_radius = repuls_radius
            best = repuls_pts.copy()
            current = repuls_pts.copy()

    # Phase 4: Simulated annealing for global exploration
    if _time_left() > 80:
        sa_pts, sa_radius = simulated_annealing(current, initial_temp=0.3, final_temp=0.01, iterations=1500)
        if sa_radius < best_radius:
            best_radius = sa_radius
            best = sa_pts.copy()
            current = sa_pts.copy()

    # Phase 5: L-BFGS-B with soft-max covering radius (progressive beta)
    if _time_left() > 40:
        beta_schedule = [30.0, 80.0, 150.0, 300.0]
        for beta in beta_schedule:
            if _time_left() < 30:
                break
            current = lbfgs_stage(current, beta, max_iter=200)
            radius, _ = compute_covering_radius(current)
            if radius < best_radius:
                best_radius = radius
                best = current.copy()

    # Phase 6: Lloyd's algorithm again after optimization
    if _time_left() > 30:
        lloyd_pts = lloyd_iteration(best, iterations=20)
        lloyd_radius, _ = compute_covering_radius(lloyd_pts)
        if lloyd_radius < best_radius:
            best_radius = lloyd_radius
            best = lloyd_pts.copy()
            current = lloyd_pts.copy()

    # Phase 7: Worst-facet optimization with adaptive step size
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    for step in step_sizes:
        if _time_left() < 20:
            break
        current = optimize_worst_facet(current, step, iterations=50)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 8: Final L-BFGS-B polish with high beta
    if _time_left() > 15:
        final_pts = lbfgs_stage(best, beta=500.0, max_iter=400)
        final_radius, _ = compute_covering_radius(final_pts)
        if final_radius < best_radius:
            best_radius = final_radius
            best = final_pts.copy()

    # Phase 9: Additional random restarts with SA + refinement
    if _time_left() > 40:
        np.random.seed(42)
        for restart in range(30):
            if _time_left() < 30:
                break
            noise_scale = 0.08 * (0.8 ** (restart // 5))
            perturbed = best.copy()
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            
            # Quick SA refinement
            sa_pts, sa_radius = simulated_annealing(perturbed, initial_temp=0.2, final_temp=0.02, iterations=500)
            
            # L-BFGS polish
            polished = lbfgs_stage(sa_pts, beta=200.0, max_iter=150)
            pol_radius, _ = compute_covering_radius(polished)
            
            if pol_radius < best_radius:
                best_radius = pol_radius
                best = polished.copy()

    # Final normalization and validation
    best = project_to_sphere(best)
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
