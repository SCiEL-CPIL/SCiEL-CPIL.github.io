import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Vectorized circumcenter calculation
            # Circumcenter direction for triangle on sphere:
            # p ∝ sin(a_side)*a + sin(b_side)*b + sin(c_side)*c
            # where a_side is the side opposite to vertex a, etc.
            
            # Side lengths (opposite to vertices a, b, c respectively)
            # a_side is opposite a, so it's the distance between b and c
            # b_side is opposite b, so it's the distance between a and c
            # c_side is opposite c, so it's the distance between a and b
            
            ab = np.sum(a * b, axis=1)
            bc = np.sum(b * c, axis=1)
            ca = np.sum(c * a, axis=1)
            
            a_side = np.arccos(np.clip(bc, -1.0, 1.0))
            b_side = np.arccos(np.clip(ca, -1.0, 1.0))
            c_side = np.arccos(np.clip(ab, -1.0, 1.0))
            
            sa = np.sin(a_side)
            sb = np.sin(b_side)
            sc = np.sin(c_side)
            
            p = sa[:, np.newaxis] * a + sb[:, np.newaxis] * b + sc[:, np.newaxis] * c
            p_norm = np.linalg.norm(p, axis=1)
            
            radii = np.zeros(len(faces))
            valid = p_norm > 1e-10
            
            if np.any(valid):
                p_valid = p[valid] / p_norm[valid, np.newaxis]
                # Max angular distance from circumcenter to any vertex
                cos_ra = np.clip(np.sum(p_valid * a[valid], axis=1), -1.0, 1.0)
                cos_rb = np.clip(np.sum(p_valid * b[valid], axis=1), -1.0, 1.0)
                cos_rc = np.clip(np.sum(p_valid * c[valid], axis=1), -1.0, 1.0)
                ra = np.arccos(cos_ra)
                rb = np.arccos(cos_rb)
                rc = np.arccos(cos_rc)
                radii[valid] = np.maximum(np.maximum(ra, rb), rc)
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                ab_d = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                bc_d = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                ca_d = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(ab_d, bc_d), ca_d)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            grid = fibonacci_spiral(5000)
            dot = np.dot(grid, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def repulsion_energy_grad(pts, exponent=2.0):
        """Compute repulsion energy and gradient for uniform distribution."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        
        w_grad = exponent * (dist_sq ** (-exponent/2 - 1))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ pts
        
        return float(np.sum(w)), grad

    def gradient_descent_repulsion(pts, exponent, alpha_init, decay, max_iter):
        """Optimize points by gradient descent on repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            energy, grad = repulsion_energy_grad(pts, exponent)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    def lloyd_refine(pts, n_iter=30, n_grid=3000):
        """Lloyd/Voronoi centroid refinement for uniform distribution."""
        pts = pts.copy()
        grid = fibonacci_spiral(n_grid)
        
        for it in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            
            dots = np.dot(grid, pts.T)  # (M, N)
            nearest = np.argmax(dots, axis=1)  # (M,)
            
            new_pts = np.zeros_like(pts)
            counts = np.zeros(N)
            for j in range(N):
                mask = nearest == j
                if np.sum(mask) > 0:
                    new_pts[j] = np.sum(grid[mask], axis=0)
                    counts[j] = np.sum(mask)
            
            for j in range(N):
                if counts[j] > 0:
                    new_pts[j] /= np.linalg.norm(new_pts[j])
            
            pts = new_pts
        return pts

    def gradient_descent_coverage(pts, beta_min, beta_max, lr, decay, max_iter, n_grid=2000):
        """Gradient descent on soft-min/soft-max coverage objective."""
        grid = fibonacci_spiral(n_grid)
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Compute dots: (M, N)
            dots = np.dot(grid, pts.T)
            dots = np.clip(dots, -1.0, 1.0)
            ang_dists = np.arccos(dots)  # (M, N)
            
            # Soft-min over points for each grid point
            T_min = beta_min
            max_d = np.max(ang_dists, axis=1, keepdims=True)
            exp_neg = np.exp(-T_min * (ang_dists - max_d))
            sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
            soft_min_dists = -1.0 / T_min * np.log(sum_exp) + max_d.squeeze()
            
            # Weights for soft-min
            w_jk = exp_neg / sum_exp  # (M, N)
            
            # Soft-max over grid points
            T_max = beta_max
            max_sm = np.max(soft_min_dists, keepdims=True)
            exp_sm = np.exp(T_max * (soft_min_dists - max_sm))
            sum_exp_sm = np.sum(exp_sm)
            softmax_max = exp_sm / sum_exp_sm  # (M,)
            
            # Gradient computation
            sin_dists = np.sin(ang_dists)
            sin_dists = np.maximum(sin_dists, 1e-10)
            
            weights = softmax_max[:, np.newaxis] * w_jk / sin_dists  # (M, N)
            grad_pts = grid.T @ weights  # (N, 3)
            
            # Update
            pts = pts + lr * grad_pts
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts = pts / norms
            
            lr *= decay
        
        return pts

    # Stage 1: Start with Fibonacci spiral
    points = fibonacci_spiral(N)

    # Stage 2: Lloyd refinement for good global distribution
    points = lloyd_refine(points, n_iter=20, n_grid=2000)

    # Stage 3: Repulsion-based optimization for better uniformity
    points = gradient_descent_repulsion(points, exponent=2.0, alpha_init=0.05, decay=0.999, max_iter=300)

    # Stage 4: Gradient descent on coverage field
    points = gradient_descent_coverage(points, beta_min=30.0, beta_max=5.0, lr=0.05, decay=0.999, max_iter=400, n_grid=2000)

    # Stage 5: Fine-tune with higher beta
    points = gradient_descent_coverage(points, beta_min=60.0, beta_max=10.0, lr=0.02, decay=0.999, max_iter=300, n_grid=2000)

    # Stage 6: Target worst facets
    if time.time() < deadline - 20.0:
        for refine_iter in range(30):
            if time.time() > deadline - 10.0:
                break
            
            # Compute facet circumradii
            try:
                hull = ConvexHull(points)
                faces = hull.simplices
                a = points[faces[:, 0]]
                b = points[faces[:, 1]]
                c = points[faces[:, 2]]
                
                ab = np.sum(a * b, axis=1)
                bc = np.sum(b * c, axis=1)
                ca = np.sum(c * a, axis=1)
                
                a_side = np.arccos(np.clip(bc, -1.0, 1.0))
                b_side = np.arccos(np.clip(ca, -1.0, 1.0))
                c_side = np.arccos(np.clip(ab, -1.0, 1.0))
                
                sa = np.sin(a_side)
                sb = np.sin(b_side)
                sc = np.sin(c_side)
                
                p = sa[:, np.newaxis] * a + sb[:, np.newaxis] * b + sc[:, np.newaxis] * c
                p_norm = np.linalg.norm(p, axis=1)
                
                radii = np.zeros(len(faces))
                valid = p_norm > 1e-10
                
                if np.any(valid):
                    p_valid = p[valid] / p_norm[valid, np.newaxis]
                    cos_ra = np.clip(np.sum(p_valid * a[valid], axis=1), -1.0, 1.0)
                    cos_rb = np.clip(np.sum(p_valid * b[valid], axis=1), -1.0, 1.0)
                    cos_rc = np.clip(np.sum(p_valid * c[valid], axis=1), -1.0, 1.0)
                    ra = np.arccos(cos_ra)
                    rb = np.arccos(cos_rb)
                    rc = np.arccos(cos_rc)
                    radii[valid] = np.maximum(np.maximum(ra, rb), rc)
                
                if np.any(~valid):
                    a_deg = a[~valid]
                    b_deg = b[~valid]
                    c_deg = c[~valid]
                    ab_d = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                    bc_d = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                    ca_d = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                    radii[~valid] = np.maximum(np.maximum(ab_d, bc_d), ca_d)
                
                max_r = np.max(radii)
                worst_face = np.argmax(radii)
                
                # Nudge the three points of the worst facet toward the circumcenter
                wf = faces[worst_face]
                ai, bi, ci = points[wf[0]], points[wf[1]], points[wf[2]]
                a_side_i = np.arccos(np.clip(np.dot(bi, ci), -1, 1))
                b_side_i = np.arccos(np.clip(np.dot(ai, ci), -1, 1))
                c_side_i = np.arccos(np.clip(np.dot(ai, bi), -1, 1))
                sa_i = np.sin(a_side_i)
                sb_i = np.sin(b_side_i)
                sc_i = np.sin(c_side_i)
                p_i = sa_i * ai + sb_i * bi + sc_i * ci
                p_norm_i = np.linalg.norm(p_i)
                if p_norm_i > 1e-10:
                    p_i = p_i / p_norm_i
                    step = 0.01
                    points[wf[0]] = points[wf[0]] + step * (p_i - points[wf[0]])
                    points[wf[1]] = points[wf[1]] + step * (p_i - points[wf[1]])
                    points[wf[2]] = points[wf[2]] + step * (p_i - points[wf[2]])
                    for idx in wf:
                        points[idx] /= np.linalg.norm(points[idx])
            except Exception:
                break

    # Stage 7: Final gradient descent polish
    if time.time() < deadline - 5.0:
        points = gradient_descent_coverage(points, beta_min=80.0, beta_max=15.0, lr=0.01, decay=0.999, max_iter=150, n_grid=2000)

    # Final projection
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points = points / norms

    return points
# EVOLVE-BLOCK-END
