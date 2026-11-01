"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def _normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return pts / norms

    def _fib_sample(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    def _compute_hull_data(pts):
        """
        Compute convex hull, facet circumcenters, and circumradii.
        Returns (circumcenters, radii, simplices) or (None, None, None) on failure.
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None

        simplices = hull.simplices
        M = len(simplices)
        circumcenters = np.zeros((M, 3))
        radii = np.zeros(M)

        for k, simplex in enumerate(simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]

            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                c = (v1 + v2 + v3) / 3.0
                c = _normalize(c.reshape(1, 3))[0]
            else:
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c

            circumcenters[k] = c
            d = np.linalg.norm(c - v1)
            radii[k] = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))

        return circumcenters, radii, simplices

    def _covering_radius_deg(pts):
        cc, radii, _ = _compute_hull_data(pts)
        if radii is None:
            return np.pi  # Invalid
        return np.degrees(np.max(radii))

    def _soft_max_objective(params, beta):
        pts = _normalize(params.reshape(N, 3))
        cc, radii, simplices = _compute_hull_data(pts)
        if radii is None:
            return 1e6, np.zeros_like(params)

        max_r = np.max(radii)
        if max_r < 1e-15:
            return 0.0, np.zeros_like(params)

        scaled = beta * (radii - max_r)
        scaled = np.clip(scaled, -100, 100)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        weights = exp_scaled / sum_exp
        value = max_r + np.log(sum_exp) / beta

        # Gradient computation
        grad = np.zeros((N, 3))
        sin_r = np.sin(radii)
        sin_r = np.maximum(sin_r, 1e-12)

        for k in range(len(radii)):
            w = weights[k]
            if w < 1e-10:
                continue
            c = cc[k]
            indices = simplices[k]
            for idx in indices:
                v = pts[idx]
                c_dot_v = np.dot(c, v)
                grad_vec = -(c - c_dot_v * v) / sin_r[k]
                grad[idx] += w * grad_vec

        return value, grad.ravel()

    def _repulsion_relax(pts, num_iters=200, step_size=0.02):
        """
        Repulsion-based relaxation: push points away from each other to spread them out.
        """
        for it in range(num_iters):
            if time.time() > deadline - 15:
                break
            # Compute pairwise repulsion
            dists = 1.0 - pts @ pts.T
            np.fill_diagonal(dists, 0.0)
            # Force proportional to inverse distance (capped)
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = dists[i, j]
                    if d > 1e-8:
                        f = 1.0 / (d + 0.1)
                        forces[i] += f * (pts[i] - pts[j])
            # Normalize and apply
            forces = _normalize(forces)
            new_pts = pts + step_size * forces
            new_pts = _normalize(new_pts)
            # Check if we improved
            _, radii, _ = _compute_hull_data(new_pts)
            if radii is not None and np.max(radii) < np.max(_compute_hull_data(pts)[1]):
                pts = new_pts
            else:
                step_size *= 0.9
                if step_size < 0.001:
                    break
        return pts

    def _lloyd_step_vectorized(pts, num_iters=50, step_size=0.1):
        """
        Vectorized Lloyd iteration: move each point toward the circumcenter of its
        nearest facet.
        """
        for it in range(num_iters):
            if time.time() > deadline - 15:
                break
            cc, radii, _ = _compute_hull_data(pts)
            if cc is None:
                break
            max_r = np.max(radii)
            if max_r < 1e-10:
                break

            # For each point, find nearest circumcenter
            # Compute distances from each point to all circumcenters
            # pts: (N, 3), cc: (M, 3)
            # dist[i, k] = ||pts[i] - cc[k]||^2
            pts_sq = np.sum(pts * pts, axis=1, keepdims=True)  # (N, 1)
            cc_sq = np.sum(cc * cc, axis=1, keepdims=True).T  # (1, M)
            dists_sq = pts_sq + cc_sq - 2.0 * (pts @ cc.T)  # (N, M)
            nearest_k = np.argmin(dists_sq, axis=1)  # (N,)

            # Move each point toward its nearest circumcenter
            targets = cc[nearest_k]  # (N, 3)
            dirs = targets - pts
            dir_norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            dir_norms = np.maximum(dir_norms, 1e-12)
            dirs = dirs / dir_norms

            new_pts = pts + step_size * dirs
            new_pts = _normalize(new_pts)

            # Check improvement
            _, new_radii, _ = _compute_hull_data(new_pts)
            if new_radii is not None and np.max(new_radii) < max_r:
                pts = new_pts
            else:
                step_size *= 0.5
                if step_size < 0.001:
                    break

        return pts

    def _projected_gradient_polish(pts, max_iters=200, lr=0.01):
        """
        Projected gradient descent on the sphere, targeting the max-radius facet.
        """
        current_pts = pts.copy()
        _, current_radii, _ = _compute_hull_data(current_pts)
        if current_radii is None:
            return current_pts
        current_max = np.max(current_radii)

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            cc, radii, simplices = _compute_hull_data(current_pts)
            if cc is None:
                break

            # Focus on the largest few facets
            max_r = np.max(radii)
            if max_r < 1e-10:
                break

            # Subgradient: for each point in the max-radius facets, move away from circumcenter
            grad = np.zeros((N, 3))
            top_k = max(1, len(radii) // 4)  # Consider top 25% of facets
            top_indices = np.argsort(radii)[-top_k:]

            for k in top_indices:
                c = cc[k]
                sin_r = np.sin(radii[k])
                sin_r = max(sin_r, 1e-12)
                for idx in simplices[k]:
                    v = current_pts[idx]
                    c_dot_v = np.dot(c, v)
                    grad_vec = -(c - c_dot_v * v) / sin_r
                    grad[idx] += grad_vec

            # Project gradient onto tangent space and normalize
            grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norms = np.maximum(grad_norms, 1e-12)
            grad = grad / grad_norms

            # Update
            new_pts = current_pts - lr * grad
            new_pts = _normalize(new_pts)

            # Check improvement
            _, new_radii, _ = _compute_hull_data(new_pts)
            if new_radii is not None and np.max(new_radii) < current_max:
                current_pts = new_pts
                current_max = np.max(new_radii)
            else:
                lr *= 0.8
                if lr < 0.0001:
                    break

        return current_pts

    def _lbfgs_stage(pts, beta=50.0, maxiter=100):
        """
        L-BFGS optimization using soft-max surrogate.
        """
        x0 = pts.ravel()
        result = minimize(
            _soft_max_objective,
            x0,
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        candidate = _normalize(result.x.reshape(N, 3))
        return candidate

    def _local_polish(pts, max_iters=100):
        """
        Local polishing using coordinate descent with adaptive step sizes.
        """
        current_pts = pts.copy()
        _, current_radii, _ = _compute_hull_data(current_pts)
        if current_radii is None:
            return current_pts
        current_max = np.max(current_radii)

        step_sizes = np.ones(N) * 0.05

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break

                best_new = current_pts[i].copy()
                best_new_max = current_max

                for _ in range(8):
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)

                    new_p = current_pts[i] + step_sizes[i] * direction
                    new_p = _normalize(new_p.reshape(1, 3))[0]

                    trial_pts = current_pts.copy()
                    trial_pts[i] = new_p

                    _, trial_radii, _ = _compute_hull_data(trial_pts)
                    if trial_radii is not None:
                        trial_max = np.max(trial_radii)
                        if trial_max < best_new_max:
                            best_new_max = trial_max
                            best_new = new_p

                if best_new_max < current_max:
                    current_pts[i] = best_new
                    current_max = best_new_max
                    step_sizes[i] = min(step_sizes[i] * 1.2, 0.2)
                    improved = True
                else:
                    step_sizes[i] *= 0.7

            if not improved:
                break

        return current_pts

    # Main optimization
    best_points = None
    best_radius = np.inf

    # Generate initial configurations
    initial_configs = []
    initial_configs.append(_fib_sample(N))

    # Add some random initializations
    for _ in range(5):
        random_pts = _normalize(np.random.randn(N, 3))
        initial_configs.append(random_pts)

    # Process each initial configuration
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break

        current_pts = init_pts.copy()

        # Phase 1: Repulsion relaxation
        current_pts = _repulsion_relax(current_pts, num_iters=50, step_size=0.02)

        # Phase 2: Lloyd iteration
        current_pts = _lloyd_step_vectorized(current_pts, num_iters=30, step_size=0.1)

        # Phase 3: L-BFGS optimization
        if time.time() < deadline - 15:
            candidate = _lbfgs_stage(current_pts, beta=50.0, maxiter=100)
            r = _covering_radius_deg(candidate)
            if r < best_radius:
                best_radius = r
                best_points = candidate.copy()

        # Phase 4: Projected gradient polish
        if time.time() < deadline - 10:
            current_pts = _projected_gradient_polish(
                best_points if best_points is not None else current_pts,
                max_iters=100, lr=0.01
            )
            r = _covering_radius_deg(current_pts)
            if r < best_radius:
                best_radius = r
                best_points = current_pts.copy()

        # Phase 5: Local polish
        if time.time() < deadline - 5:
            polished = _local_polish(
                best_points if best_points is not None else current_pts,
                max_iters=50
            )
            r = _covering_radius_deg(polished)
            if r < best_radius:
                best_radius = r
                best_points = polished.copy()

    # Final verification and normalization
    if best_points is None:
        best_points = _fib_sample(N)

    best_points = _normalize(best_points)

    return best_points
# EVOLVE-BLOCK-END
