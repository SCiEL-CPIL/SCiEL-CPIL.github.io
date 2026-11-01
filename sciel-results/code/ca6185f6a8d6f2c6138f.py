"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()

    def get_covering_radius_and_grads(x_flat):
        """
        Computes a smooth proxy for the covering radius and its gradient.
        We minimize the maximum squared distance from the origin to the circumcenters 
        of the hull facets. This is equivalent to minimizing the covering radius 
        (since dist = arccos(1/sqrt(1+|c|^2)) is monotonic with |c|).
        Actually, a better proxy is the maximum circumradius R_facet = |c_planar|.
        The covering radius is arccos(1/sqrt(1+R^2)) if the center is inside.
        Minimizing max R is equivalent to minimizing max covering radius.
        
        We return the max R and the gradient of R w.r.t x.
        For the active facet, the gradient is derived from the circumcenter position.
        """
        pts = x_flat.reshape(N, 3)
        
        # Ensure points are on the sphere (project if drifted)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            # If hull fails, return a large penalty and zero gradient
            return 10.0, np.zeros_like(x_flat)
        
        max_r2 = 0.0
        worst_simplex = None
        grad = np.zeros_like(x_flat)
        
        # We need to find the facet with the largest circumradius.
        # Circumradius squared of triangle p1, p2, p3:
        # |c|^2 = (|u|^2 |v|^2 - (u.v)^2) / (4 * Area_2D^2) ?? No.
        # Let's compute the circumcenter c_planar and its norm.
        
        # For efficiency, we can compute the circumradius directly.
        # R = (a*b*c) / (4*A) where a,b,c are side lengths and A is area of the planar triangle.
        # Or using vectors:
        # u = p2-p1, v = p3-p1
        # Area_2D = 0.5 * |u x v|
        # R^2 = (|u|^2 |v|^2 - (u.v)^2) / (4 * Area_2D^2) = (|u|^2 |v|^2 - (u.v)^2) / |u x v|^2
        # Wait, that's for the triangle in 3D. The circumcenter is in the plane of the triangle.
        # The distance from origin to the circumcenter is what we care about?
        # No, the covering radius is the distance from the circumcenter (on the sphere) to the vertices.
        # Let c_sph be the normalized circumcenter.
        # dist = arccos(dot(c_sph, p1)).
        # To make it smooth, we can minimize the squared distance in the tangent plane?
        # Or simply minimize the "radius" of the cap, which is determined by the angle.
        
        # Let's stick to the circumcenter norm approach or the direct angle.
        # The angle is arccos(cos_theta). cos_theta = dot(c_sph, p1).
        # Minimizing angle is equivalent to maximizing cos_theta (since angle is in [0, pi/2] typically for good coverings).
        # So we want to MAXIMIZE the minimum cos_theta over all facets?
        # No, we want to MINIMIZE the MAX angle.
        # Max angle corresponds to Min cos_theta.
        # So we want to MAXIMIZE the minimum cos_theta over all facets.
        
        # Let's define objective = -min_cos_theta. We want to minimize this.
        # Gradient of -min_cos_theta is -gradient of the active facet's cos_theta.
        
        min_cos = 1.0 # Start with max possible
        active_simplex = None
        
        for simplex in hull.simplices:
            i, j, k = simplex
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            u = p2 - p1
            v = p3 - p1
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-12:
                continue
                
            # Circumcenter in plane: c = p1 + a*u + b*v
            # a = (0.5 * uu * vv - 0.5 * uv * vv) / det
            # b = (0.5 * vv * uu - 0.5 * uu * uv) / det
            a = 0.5 * (vv - uv) / det
            b = 0.5 * (uu - uv) / det
            
            c_planar = p1 + a * u + b * v
            c_norm = np.linalg.norm(c_planar)
            if c_norm < 1e-12:
                continue
                
            c_sph = c_planar / c_norm
            
            # Check if c_sph is inside the spherical triangle
            P = np.column_stack([p1, p2, p3])
            try:
                coeffs = np.linalg.solve(P, c_sph)
                if np.all(coeffs > -1e-6):
                    cos_theta = np.dot(c_sph, p1)
                    # Clip for safety
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    if cos_theta < min_cos:
                        min_cos = cos_theta
                        active_simplex = simplex
                        # We will compute gradient later for this active simplex
            except np.linalg.LinAlgError:
                pass
                
        if active_simplex is None:
            # Fallback: use the first simplex or return 0
            return 0.0, np.zeros_like(x_flat)
            
        # Now compute gradient for the active simplex
        # We want to minimize -min_cos. So we want to increase min_cos.
        # Gradient of cos_theta = dot(c_sph, p1) w.r.t p1, p2, p3.
        # This is complex. A simpler approximation for L-BFGS is to use finite differences 
        # or a heuristic gradient.
        # However, L-BFGS can handle non-smoothness to some extent if we provide a consistent gradient.
        # Let's use a simpler proxy: The squared distance from the origin to the circumcenter is NOT the covering radius.
        # The covering radius is the angle.
        
        # Alternative Strategy:
        # Use the fact that for a facet, the covering radius is determined by the circumcenter.
        # If we move the vertices, the circumcenter moves.
        # Let's use a "soft" max of the angles.
        # But computing all angles is expensive.
        
        # Let's try a different smooth objective:
        # Minimize the sum of exp(beta * angle_i) for all facets.
        # This is differentiable everywhere.
        # As beta -> inf, it approaches the max.
        
        # We need to re-evaluate all facets for the sum.
        sum_exp = 0.0
        beta = 50.0 # Temperature for softmax
        
        for simplex in hull.simplices:
            i, j, k = simplex
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            u = p2 - p1
            v = p3 - p1
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-12:
                continue
                
            a = 0.5 * (vv - uv) / det
            b = 0.5 * (uu - uv) / det
            
            c_planar = p1 + a * u + b * v
            c_norm = np.linalg.norm(c_planar)
            if c_norm < 1e-12:
                continue
                
            c_sph = c_planar / c_norm
            
            P = np.column_stack([p1, p2, p3])
            try:
                coeffs = np.linalg.solve(P, c_sph)
                if np.all(coeffs > -1e-6):
                    cos_theta = np.dot(c_sph, p1)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle = np.arccos(cos_theta)
                    sum_exp += np.exp(beta * angle)
            except np.linalg.LinAlgError:
                pass
        
        # Objective: log(sum_exp) / beta approximates the max.
        # Gradient: We need d/dx sum_exp.
        # This requires the gradient of angle w.r.t x.
        # This is analytically difficult.
        
        # Let's go back to the "Active Facet" gradient approach but implement it carefully.
        # If we just move the points of the active facet to reduce the angle, we might break others.
        # L-BFGS with a non-smooth objective (min-max) often works well if we use the subgradient of the active set.
        
        # Let's compute the gradient of the active facet's angle.
        # angle = arccos(c_sph . p1)
        # This is getting too complex for a robust quick implementation.
        
        # SIMPLER ROBUST APPROACH:
        # Use Nelder-Mead or Powell? No, 150 vars.
        # Use L-BFGS-B with a finite difference gradient? Too slow (150 evals per grad step).
        
        # Let's use a "Repulsion" based force model which is smooth.
        # Force on each point is the sum of repulsive forces from other points?
        # No, that minimizes energy, not covering radius.
        
        # Let's try a different structural change:
        # **Spectral / Fourier Approach?** No.
        
        # Let's try **Global Minimax with Subgradient**.
        # Objective f(x) = max_k g_k(x) where g_k is the angle of facet k.
        # Subgradient is the gradient of the active g_k.
        # We can compute the gradient of g_k analytically.
        
        # Let's derive the gradient of the angle of the circumcenter of a triangle (p1, p2, p3) on the sphere.
        # This is a known problem.
        # Actually, there is a simpler geometric interpretation.
        # The farthest point from the set is the circumcenter of the "critical" facet.
        # If we move p1, p2, p3, the circumcenter moves.
        
        # Given the time constraints and complexity, let's use a **Heuristic Gradient** 
        # that moves the active points towards the circumcenter.
        # This is essentially what the seed program did, but now we use L-BFGS to manage the step size and history.
        
        # Wait, L-BFGS requires a smooth function.
        # If we provide a non-smooth function, L-BFGS might fail or oscillate.
        
        # **Alternative Strategy: Use `scipy.optimize.minimize` with method='L-BFGS-B' on a smoothed objective.**
        # Objective: Sum of exp(beta * angle_k).
        # We need the gradient of this sum.
        # We can approximate the gradient of angle_k using finite differences ONLY for the active facets?
        # No, that's inconsistent.
        
        # Let's try a **Coordinate Descent** with **Line Search**.
        # It is robust to non-smoothness.
        # For each point, we can optimize its position on the sphere to minimize the global covering radius.
        
        def optimize_point(idx, pts):
            # Optimize point idx while keeping others fixed
            # We can use a local search around the current point.
            # Or use the fact that the optimal position for a single point (given others) 
            # is such that it balances the Voronoi cell area?
            # No, that's for minimal energy. For covering, it's about the max distance.
            
            # Let's use a simple local minimization for this single point.
            # Variable: azimuth and elevation (2 DOF).
            
            best_val = np.inf
            best_pt = pts[idx].copy()
            
            # Compute current covering radius
            def eval_single(pt):
                trial_pts = pts.copy()
                trial_pts[idx] = pt / np.linalg.norm(pt)
                # Compute covering radius
                try:
                    hull = ConvexHull(trial_pts)
                    max_angle = 0.0
                    for simplex in hull.simplices:
                        i, j, k = simplex
                        p1, p2, p3 = trial_pts[i], trial_pts[j], trial_pts[k]
                        u = p2 - p1
                        v = p3 - p1
                        uu = np.dot(u, u)
                        vv = np.dot(v, v)
                        uv = np.dot(u, v)
                        det = uu * vv - uv * uv
                        if abs(det) < 1e-12: continue
                        a = 0.5 * (vv - uv) / det
                        b = 0.5 * (uu - uv) / det
                        c_planar = p1 + a * u + b * v
                        c_norm = np.linalg.norm(c_planar)
                        if c_norm < 1e-12: continue
                        c_sph = c_planar / c_norm
                        P = np.column_stack([p1, p2, p3])
                        try:
                            coeffs = np.linalg.solve(P, c_sph)
                            if np.all(coeffs > -1e-6):
                                cos_t = np.dot(c_sph, p1)
                                cos_t = np.clip(cos_t, -1.0, 1.0)
                                ang = np.arccos(cos_t)
                                if ang > max_angle: max_angle = ang
                        except: pass
                    return max_angle
                except: return np.pi

            current_val = eval_single(pts[idx])
            
            # Grid search in tangent plane
            # Project current point to tangent space
            p = pts[idx]
            # Find two orthogonal vectors in tangent plane
            v1 = np.cross(p, np.array([0, 0, 1.0]))
            if np.linalg.norm(v1) < 1e-6:
                v1 = np.cross(p, np.array([1, 0, 0.0]))
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(p, v1)
            
            step = 0.1
            for _ in range(10):
                improved = False
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0: continue
                        new_p = p + dx * v1 + dy * v2
                        new_p = new_p / np.linalg.norm(new_p)
                        val = eval_single(new_p)
                        if val < current_val - 1e-6:
                            current_val = val
                            best_pt = new_p
                            improved = True
                if improved:
                    step *= 0.7
                else:
                    break
            
            return best_pt, current_val

        # Main Loop: Coordinate Descent
        pts = x0.reshape(N, 3).copy()
        
        # Initial evaluation
        def compute_cr(pts):
            try:
                hull = ConvexHull(pts)
                max_angle = 0.0
                for simplex in hull.simplices:
                    i, j, k = simplex
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    u = p2 - p1
                    v = p3 - p1
                    uu = np.dot(u, u)
                    vv = np.dot(v, v)
                    uv = np.dot(u, v)
                    det = uu * vv - uv * uv
                    if abs(det) < 1e-12: continue
                    a = 0.5 * (vv - uv) / det
                    b = 0.5 * (uu - uv) / det
                    c_planar = p1 + a * u + b * v
                    c_norm = np.linalg.norm(c_planar)
                    if c_norm < 1e-12: continue
                    c_sph = c_planar / c_norm
                    P = np.column_stack([p1, p2, p3])
                    try:
                        coeffs = np.linalg.solve(P, c_sph)
                        if np.all(coeffs > -1e-6):
                            cos_t = np.dot(c_sph, p1)
                            cos_t = np.clip(cos_t, -1.0, 1.0)
                            ang = np.arccos(cos_t)
                            if ang > max_angle: max_angle = ang
                    except: pass
                return max_angle
            except: return np.pi

        current_val = compute_cr(pts)
        best_val = current_val
        best_pts = pts.copy()
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5.0:
            improved_global = False
            # Randomize order to avoid bias
            order = np.random.permutation(N)
            
            for idx in order:
                if time.time() > deadline - 5.0: break
                new_pt, new_val_local = optimize_point(idx, pts)
                # We need to re-evaluate the GLOBAL covering radius because moving one point affects the hull
                pts[idx] = new_pt
                global_val = compute_cr(pts)
                
                if global_val < current_val - 1e-8:
                    current_val = global_val
                    improved_global = True
                    if current_val < best_val:
                        best_val = current_val
                        best_pts = pts.copy()
                else:
                    # Revert
                    # We don't have the old point stored easily in optimize_point, 
                    # so we should have kept it.
                    # Let's refactor optimize_point to return the old point or we store it before.
                    pass # This is a flaw. We need to revert.
            
            # To fix the revert issue, we must store the old point.
            # Let's rewrite the loop properly.
            
            break # Break out of the flawed loop structure and implement a correct one below.

        # Correct Coordinate Descent Implementation
        pts = x0.reshape(N, 3).copy()
        current_val = compute_cr(pts)
        best_val = current_val
        best_pts = pts.copy()
        
        for epoch in range(100):
            if time.time() > deadline - 5.0: break
            
            improved_global = False
            order = np.random.permutation(N)
            
            for idx in order:
                if time.time() > deadline - 5.0: break
                
                old_pt = pts[idx].copy()
                
                # Optimize this point
                new_pt, _ = optimize_point(idx, pts)
                pts[idx] = new_pt
                
                global_val = compute_cr(pts)
                
                if global_val < current_val - 1e-8:
                    current_val = global_val
                    improved_global = True
                    if current_val < best_val:
                        best_val = current_val
                        best_pts = pts.copy()
                else:
                    # Revert
                    pts[idx] = old_pt
            
            if not improved_global:
                # Try random perturbations if stuck
                for _ in range(10):
                    if time.time() > deadline - 5.0: break
                    idx = np.random.randint(N)
                    old_pt = pts[idx].copy()
                    noise = np.random.normal(0, 0.1, 3)
                    new_pt = pts[idx] + noise
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    pts[idx] = new_pt
                    global_val = compute_cr(pts)
                    if global_val < current_val - 1e-8:
                        current_val = global_val
                        if current_val < best_val:
                            best_val = current_val
                            best_pts = pts.copy()
                    else:
                        pts[idx] = old_pt

        return best_pts.flatten()

# EVOLVE-BLOCK-END
