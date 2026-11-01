"""Program for covering the sphere with 130 equal caps using facet-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Leave 8s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            # Compute circumcenter direction for each triangle
            # For triangle abc on unit sphere, circumcenter direction is proportional to
            # cross(b-a, c-a) + ... Actually use the formula:
            # The circumcenter of triangle with vertices a,b,c on unit sphere
            # lies along the direction of the normal to the plane, at distance 1/|n| from origin
            # where n is the unit normal. The circumradius R satisfies cos(R) = |n| * (1/|n|) = 1?
            # No. Let me use: circumradius = arccos(|n_unit|) where n_unit is the normalized
            # circumcenter direction... 
            # Actually: for three points a,b,c on unit sphere, the circumcircle radius R
            # satisfies: the circumcenter C is at distance d from origin where d = cos(R).
            # The direction of C is (a × b + b × c + c × a) / |a × b + b × c + c × a|
            # and cos(R) = |a × b + b × c + c × a| / |a × b + b × c + c × a| ... no.
            
            # Correct formula: circumcenter direction n = (a×b + b×c + c×a)
            # |n| = 2 * area * something... 
            # Actually the circumradius R: cos(R) = |n| / |n| where n is unnormalized...
            # Let me use: the circumcenter is at C = n / |n|^2 * something
            # 
            # Standard result: for points a, b, c on unit sphere, the circumradius R is given by
            # cos(R) = |a×b + b×c + c×a| / (2 * |a×b| * |b×c| * |c×a| / (2*V)) ... too complex
            #
            # Simpler: circumcenter direction = (a×b + b×c + c×a) normalized
            # The circumradius R satisfies: the circumcenter C has |C| = 1/cos(R)
            # and C direction = n_hat where n = a×b + b×c + c×a
            # |n| = 2 * (sin(A) + sin(B) + sin(C)) * ... 
            #
            # Let me just compute it numerically:
            # The circumcenter C satisfies |C - a| = |C - b| = |C - c| = sin(R)
            # and C is in the plane of a,b,c. C = alpha*a + beta*b + gamma*c with alpha+beta+gamma=1
            # |C-a|^2 = |C-b|^2 => 2*C.(b-a) = |b|^2 - |a|^2 = 0 (unit sphere)
            # So C.(b-a) = 0, C.(c-a) = 0, meaning C is perpendicular to (b-a) and (c-a)
            # So C is along (b-a) × (c-a) direction.
            # C = t * ((b-a) × (c-a)) / |((b-a) × (c-a))|^2 * ... 
            # Actually C = lambda * n where n = (b-a)×(c-a)
            # C.(b-a) = 0 is automatic since n ⊥ (b-a)
            # |C - a|^2 = |C|^2 - 2 C.a + |a|^2 = |C|^2 - 2C.a + 1 = sin^2(R)
            # |C|^2 = lambda^2 |n|^2
            # C.a = lambda * n.a
            # lambda^2 |n|^2 - 2 lambda n.a + 1 = sin^2(R)
            # Also |C - b|^2 = |C|^2 - 2C.b + 1 = sin^2(R)
            # So C.a = C.b => lambda n.a = lambda n.b => n.(a-b) = 0 ✓
            # From |C-a|^2 = sin^2(R): lambda^2|n|^2 - 2lambda n.a + 1 - sin^2(R) = 0
            # lambda^2|n|^2 - 2lambda n.a + cos^2(R) = 0
            # The circumcenter is the point on the line through origin in direction n
            # that is equidistant from a, b, c.
            # lambda = n.a / |n|^2 (from C.a = C.b and the geometry)
            # Wait, let me just use: cos(R) = |n| / (2 * area_factor)
            # 
            # Simplest robust approach: compute circumcenter explicitly
            n_vec = np.cross(b - a, c - a)  # normal to plane
            n_norm = np.linalg.norm(n_vec, axis=1)
            
            radii = np.zeros(len(faces))
            valid = n_norm > 1e-10
            
            if np.any(valid):
                # C = lambda * n_vec / n_norm (unit direction)
                # C.a = C.b => lambda * (n_vec/n_norm).a = lambda * (n_vec/n_norm).b
                # This is automatically satisfied
                # |C - a|^2 = sin^2(R)
                # Let u = n_vec / n_norm (unit normal)
                # C = d * u where d is distance from origin
                # |d*u - a|^2 = d^2 - 2d*(u.a) + 1 = sin^2(R)
                # d^2 - 2d*(u.a) + cos^2(R) = 0
                # Also the circumcenter must be equidistant: d*(u.a) = d*(u.b) => u.(a-b)=0 ✓
                # Solve: d = u.a ± sqrt((u.a)^2 - cos^2(R))... circular
                # 
                # Better: the circumradius R is the angle such that the great circle
                # through a,b,c has radius R. 
                # cos(R) = |n_vec| / (|a-b| * |b-c| * |c-a| / (4 * area))... 
                #
                # Let me use the formula: for triangle with sides (chord lengths) 
                # a_s = |b-c|, b_s = |c-a|, c_s = |a-b|
                # Circumradius (on sphere) R: sin(R) = a_s / (2 * sin(A)) where A is angle
                # Or: sin(R) = abc / (4 * Area) where Area is spherical area
                #
                # Simplest: compute circumcenter numerically
                # u = n_vec / n_norm
                # C = d * u
                # Minimize |C - a|^2 subject to C in direction u:
                # f(d) = d^2 - 2d*(u.a) + 1
                # f'(d) = 2d - 2(u.a) = 0 => d = u.a
                # But this gives C = (u.a)*u, and |C - a|^2 = (u.a)^2 - 2(u.a)^2 + 1 = 1 - (u.a)^2
                # sin^2(R) = 1 - (u.a)^2 => cos(R) = |u.a|
                # But we need C to also satisfy |C-b| = |C-c|, which requires u.a = u.b = u.c
                # This is NOT generally true! The circumcenter is NOT simply the projection.
                #
                # Correct: C = alpha*a + beta*b + gamma*c, alpha+beta+gamma=1
                # |C-a|^2 = |C-b|^2 => 2C.(b-a) = 0 => C ⊥ (b-a)
                # |C-a|^2 = |C-c|^2 => C ⊥ (c-a)
                # So C is parallel to (b-a)×(c-a) = n_vec
                # C = d * n_vec / n_norm
                # C.(b-a) = 0 is automatic
                # |C-a|^2 = |C|^2 - 2C.a + 1 = d^2 - 2d*(n_vec.a/n_norm) + 1
                # |C-b|^2 = d^2 - 2d*(n_vec.b/n_norm) + 1
                # Setting equal: n_vec.a = n_vec.b (should hold for valid triangle)
                # sin^2(R) = d^2 - 2d*(n_vec.a/n_norm) + 1
                # To find d: we need another condition. The circumcenter is unique.
                # From |C-a|^2 = |C-b|^2: already used.
                # We need: C is in the affine plane of a,b,c.
                # C = d * n_vec/n_norm must satisfy: C = a + s*(b-a) + t*(c-a)
                # This gives: d * n_vec/n_norm - a = s*(b-a) + t*(c-a)
                # Dot with n_vec: d * n_norm - n_vec.a = 0 (since n_vec ⊥ (b-a) and (c-a))
                # So d = n_vec.a / n_norm
                # Then sin^2(R) = (n_vec.a/n_norm)^2 - 2*(n_vec.a/n_norm)^2 + 1 = 1 - (n_vec.a/n_norm)^2
                # cos(R) = |n_vec.a / n_norm|
                # 
                # Let me verify: for equilateral triangle on equator, a=(1,0,0), b=(0,1,0), c=(0,0,1)
                # n_vec = (b-a)×(c-a) = (-1,1,0)×(-1,0,1) = (1,1,1)
                # n_norm = sqrt(3)
                # n_vec.a = -1
                # cos(R) = |-1/sqrt(3)| = 1/sqrt(3) => R = arccos(1/sqrt(3)) ≈ 54.7°
                # But the actual circumradius of this triangle is 90° (they're on a great circle? No)
                # a,b,c are at 90° from each other. The circumcircle through them: center at (1,1,1)/sqrt(3)
                # distance from center to a: |(1,1,1)/sqrt(3) - (1,0,0)| = |(-2,1,1)/sqrt(3)| = sqrt(6)/sqrt(3) = sqrt(2)
                # sin(R) = sqrt(2) => R = 45°? No, sin(R)=sqrt(2) is impossible.
                # Wait, the circumradius on the sphere: the points are at angular distance 90° apart.
                # The circumcenter is at (1,1,1)/sqrt(3), and the angular distance from center to each point:
                # cos(angle) = (1,1,1)/sqrt(3) . (1,0,0) = 1/sqrt(3) => angle = arccos(1/sqrt(3)) ≈ 54.74°
                # So R ≈ 54.74°. Let me check: is this correct?
                # Three points at 90° mutual distance on sphere: yes, circumradius is arccos(1/sqrt(3)).
                # OK so the formula cos(R) = |n_vec.a / n_norm| seems correct!
                
                u_a = np.sum(n_vec[valid] * a[valid], axis=1) / n_norm[valid]
                cos_R = np.abs(u_a)
                cos_R = np.clip(cos_R, 0.0, 1.0)
                radii[valid] = np.arccos(cos_R)
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            i = np.arange(3000) + 0.5
            z = 1.0 - 2.0 * i / 3000
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
            grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            dot = np.dot(grid, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def facet_logsumexp_objective_and_grad(pts, temperature=80.0):
        """
        Smooth surrogate for covering radius using log-sum-exp over facet circumradii.
        Returns (value, gradient) where value ≈ max circumradius (in radians).
        Vectorized gradient computation.
        """
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            n_faces = len(faces)
            
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            n_vec = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n_vec, axis=1)
            
            radii = np.zeros(n_faces)
            valid = n_norm > 1e-10
            
            if np.any(valid):
                u_a = np.sum(n_vec[valid] * a[valid], axis=1) / n_norm[valid]
                cos_R = np.abs(u_a)
                cos_R = np.clip(cos_R, 0.0, 1.0 - 1e-12)
                radii[valid] = np.arccos(cos_R)
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            # Log-sum-exp for smooth max
            max_r = np.max(radii)
            shifted = radii - max_r
            exp_shifted = np.exp(shifted * temperature)
            sum_exp = np.sum(exp_shifted)
            value = max_r + np.log(sum_exp) / temperature
            
            # Weights for gradient
            w = exp_shifted / sum_exp  # shape (n_faces,)
            
            grad = np.zeros_like(pts)
            
            if np.any(valid):
                n_vec_v = n_vec[valid]
                n_norm_v = n_norm[valid]
                u = n_vec_v / n_norm_v[:, np.newaxis]  # circumcenter directions, shape (n_valid, 3)
                
                sin_R = np.sin(radii[valid])
                sin_R = np.maximum(sin_R, 1e-10)
                
                weights = w[valid] / sin_R  # (n_valid,)
                
                # Vectorized: for each valid facet, compute tangent push for all 3 vertices
                valid_faces = faces[valid]  # shape (n_valid, 3)
                valid_pts = pts[valid_faces]  # shape (n_valid, 3, 3)
                
                # u for each facet: (n_valid, 1, 3)
                u_expanded = u[:, np.newaxis, :]  # (n_valid, 1, 3)
                
                # dot products u.v for each vertex: (n_valid, 3)
                u_dot_v = np.sum(u_expanded * valid_pts, axis=2)  # (n_valid, 3)
                
                # tangent = u - (u.v) * v, shape (n_valid, 3, 3)
                tangents = u_expanded - u_dot_v[:, :, np.newaxis] * valid_pts  # (n_valid, 3, 3)
                
                # Weighted contribution: (n_valid, 3, 3)
                weighted = weights[:, np.newaxis, np.newaxis] * tangents  # (n_valid, 3, 3)
                
                # Accumulate into grad using np.add.at
                for k in range(3):
                    np.add.at(grad, valid_faces[:, k], weighted[:, k, :])
            
            # Project gradient to tangent space
            for i in range(N):
                p = pts[i]
                grad[i] -= np.dot(grad[i], p) * p
            
            return float(value), grad
        
        except Exception:
            # Fallback: use grid-based gradient
            i = np.arange(2000) + 0.5
            z = 1.0 - 2.0 * i / 2000
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
            grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
            dot = np.dot(grid, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            nearest = np.argmin(ang_dists, axis=1)
            
            T = 30.0
            max_d = np.max(min_dists)
            shifted = min_dists - max_d
            exp_shifted = np.exp(shifted * T)
            sum_exp = np.sum(exp_shifted)
            value = max_d + np.log(sum_exp) / T
            w_j = exp_shifted / sum_exp
            
            grad = np.zeros_like(pts)
            for j in range(len(grid)):
                if w_j[j] > 1e-10:
                    k = nearest[j]
                    d = dot[j, k]
                    denom = np.sqrt(max(1.0 - d * d, 1e-10))
                    grad[k] += w_j[j] * (-grid[j] / denom)
            
            for i in range(N):
                p = pts[i]
                grad[i] -= np.dot(grad[i], p) * p
            
            return float(value), grad

    def optimize_facet_lbfgsb(points, maxiter=150, temperature=80.0, time_limit=None):
        """Optimize using L-BFGS-B on the facet log-sum-exp surrogate."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        def objective(x):
            pts = x.reshape(N, 3)
            norms_x = np.linalg.norm(pts, axis=1, keepdims=True)
            norms_x[norms_x < 1e-10] = 1e-10
            pts = pts / norms_x
            val, _ = facet_logsumexp_objective_and_grad(pts, temperature)
            return val
        
        def gradient(x):
            pts = x.reshape(N, 3)
            norms_x = np.linalg.norm(pts, axis=1, keepdims=True)
            norms_x[norms_x < 1e-10] = 1e-10
            pts = pts / norms_x
            _, g = facet_logsumexp_objective_and_grad(pts, temperature)
            return g.flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def repulsion_gradient_descent(pts, exponent=3.0, alpha=0.05, decay=0.99, max_iter=200):
        """Optimize points by gradient descent on repulsion energy for good initial spread."""
        for _ in range(max_iter):
            if time.time() > deadline - 20.0:
                break
            
            dot = np.dot(pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            
            w_grad = exponent * (dist_sq ** (-exponent / 2 - 1))
            np.fill_diagonal(w_grad, 0.0)
            
            grad = w_grad @ pts
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + facet L-BFGS-B optimization
    n_restarts = 6

    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 4:
            # Fibonacci with different offset
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(300 + restart)
            points = points + 0.15 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(500 + restart)
                points = best_points + 0.03 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion for good spread
        points = repulsion_gradient_descent(points, 3.0, 0.05, 0.99, 150)

        # Then optimize with facet L-BFGS-B using temperature schedule
        time_left = deadline - time.time() - 12.0
        if time_left > 8.0:
            # Start with higher temperature (smoother), then lower
            points, _ = optimize_facet_lbfgsb(points, maxiter=80, temperature=60.0)
            time_left2 = deadline - time.time() - 12.0
            if time_left2 > 8.0:
                points, _ = optimize_facet_lbfgsb(points, maxiter=60, temperature=100.0)

        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)

        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local search refinement with more directions and adaptive steps
    if best_points is not None and time.time() < deadline - 15.0:
        points = best_points.copy()
        improved = True
        max_rounds = 8
        round_count = 0

        while improved and round_count < max_rounds and time.time() < deadline - 6.0:
            improved = False
            round_count += 1

            for i in range(N):
                if time.time() > deadline - 6.0:
                    break

                old_p = points[i].copy()

                rng = np.random.RandomState(i * 1000 + round_count * 100)
                for trial in range(10):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm

                    # Try multiple step sizes including smaller ones
                    for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p

                        rad = compute_covering_radius_exact(points)
                        if rad < best_rad - 1e-12:
                            best_rad = rad
                            best_points = points.copy()
                            improved = True
                            old_p = new_p.copy()
                            break
                        else:
                            points[i] = old_p
                    else:
                        continue
                    break

    # Phase 3: Final polish with lower temperature L-BFGS-B
    if best_points is not None and time.time() < deadline - 8.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 4.0
        if time_left > 6.0:
            # Two-stage polish: moderate then high precision
            final, _ = optimize_facet_lbfgsb(final, maxiter=60, temperature=80.0)
            time_left2 = deadline - time.time() - 4.0
            if time_left2 > 6.0:
                final, _ = optimize_facet_lbfgsb(final, maxiter=40, temperature=150.0)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points
