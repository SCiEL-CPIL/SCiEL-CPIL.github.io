import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        
        circumcenters = np.zeros((len(facets), 3))
        radii = np.zeros(len(facets))
        
        for k, facet in enumerate(facets):
            a = p[facet[0]]
            b = p[facet[1]]
            c = p[facet[2]]
            v1 = a - b
            v2 = b - c
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                circumcenters[k] = a
                radii[k] = 0.0
                continue
            cc = n / norm_n
            if np.dot(cc, a) < 0:
                cc = -cc
            circumcenters[k] = cc
            cos_r = np.clip(np.dot(cc, a), -1.0, 1.0)
            radii[k] = np.arccos(cos_r)
        
        return facets, circumcenters, radii

    def objective_sum_sq(params):
        """Objective for L-BFGS: Sum of squared radii."""
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
            
        return np.sum(radii**2)

    def polish_max_radius(pts, steps=200, eta=0.01):
        """
        Polish by minimizing the maximum radius directly using subgradients.
        The subgradient of max(r_i) is the average of gradients of active facets.
        Gradient of r_i (circumradius of facet i) w.r.t point p_k:
        If p_k is in facet i, grad_r_i = (cc_i - p_k) / sin(r_i) * (1/r_i?) 
        Actually, r = arccos(dot(cc, p)). 
        d r / d p = - (cc / sqrt(1 - dot(cc,p)^2)) = - cc / sin(r).
        So to minimize r, we want to move p in direction - grad_r = cc / sin(r).
        Wait, let's check signs.
        r = arccos(u), u = cc . p.
        dr/du = -1/sqrt(1-u^2) = -1/sin(r).
        du/dp = cc.
        dr/dp = -cc / sin(r).
        To decrease r, we move in direction -dr/dp = cc / sin(r).
        This direction is the circumcenter projected onto the tangent space?
        cc is unit vector. p is unit vector.
        Tangent projection of cc is cc - (cc.p)p.
        Note that cc . p = cos(r).
        So cc - cos(r)p.
        Direction = (cc - cos(r)p) / sin(r).
        This is exactly the unit tangent vector pointing towards the circumcenter.
        """
        current_pts = pts.copy()
        best_local_radius = float('inf')
        best_local_pts = current_pts.copy()
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
                
            facets, circumcenters, radii = compute_hull_data(current_pts)
            if facets is None:
                break
                
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local_pts = current_pts.copy()
                
            # Find active facets (those close to max radius)
            # Use a small tolerance to include all "maximal" facets
            tol = 1e-6
            active_mask = radii > (max_r - tol)
            
            if not np.any(active_mask):
                break
            
            # Compute subgradient: average of tangent directions towards circumcenters of active facets
            grad = np.zeros_like(current_pts)
            
            active_indices = np.where(active_mask)[0]
            for k in active_indices:
                facet = facets[k]
                cc = circumcenters[k]
                r_val = radii[k]
                
                # Avoid division by zero if r is 0 (should not happen for max radius > 0)
                if r_val < 1e-10:
                    continue
                    
                sin_r = np.sin(r_val)
                
                # Direction on sphere towards cc
                # proj_cc = cc - (cc.p)p
                # dir = proj_cc / ||proj_cc||
                # ||proj_cc|| = sin(r)
                # So dir = (cc - cos(r)p) / sin(r)
                cos_r = np.cos(r_val)
                # Note: cc . p[k] might differ slightly from cos(r) due to numerical errors
                # Using the formula derived: grad_r = -cc/sin(r) is not quite right on sphere.
                # Let's stick to the tangent projection.
                
                # We want to move p_k to reduce r.
                # The gradient of r w.r.t p_k in ambient space is -cc/sin(r).
                # Projected to tangent space:
                # grad_tan = grad_amb - (grad_amb . p_k) p_k
                # grad_amb = -cc / sin(r)
                # grad_amb . p_k = - (cc . p_k) / sin(r) = - cos(r) / sin(r) = -cot(r)
                # grad_tan = -cc/sin(r) + cot(r) p_k
                # = (-cc + cos(r)/sin(r) * sin(r) p_k) / sin(r)  <-- wait
                # = (-cc + cos(r) p_k) / sin(r)
                # This is the direction of INCREASING r?
                # Let's check: if we move p towards cc, r should decrease.
                # Vector from p to cc is cc - p.
                # Tangent part is cc - cos(r)p.
                # This vector points towards cc.
                # So to DECREASE r, we should move in direction (cc - cos(r)p).
                # The subgradient of max(r) is the average of gradients of active r's.
                # grad(r_k) = - (cc_k - cos(r_k)p_k) / sin(r_k) ??
                # Let's re-verify derivative.
                # f(p) = arccos(cc.p)
                # df/dp = -1/sqrt(1-(cc.p)^2) * cc = -cc/sin(r)
                # This is the gradient in R^3.
                # To do gradient descent on the sphere, we project the gradient to the tangent space.
                # grad_sphere = grad_R3 - (grad_R3 . p) p
                # grad_R3 . p = - (cc.p)/sin(r) = -cos(r)/sin(r)
                # grad_sphere = -cc/sin(r) - (-cos(r)/sin(r)) p
                # grad_sphere = (-cc + cos(r)p) / sin(r)
                # This is the direction of steepest ASCENT for r.
                # So we should move in the OPPOSITE direction to decrease r.
                # Direction to move: -grad_sphere = (cc - cos(r)p) / sin(r).
                # This is the unit vector in the tangent space pointing from p to cc.
                
                dir_vec = (cc - cos_r * current_pts[k]) / sin_r
                # Note: this is already unit length in tangent space.
                
                # Accumulate subgradient for point k
                # Since r is max, we want to minimize it.
                # The subgradient of max is in the convex hull of gradients of active functions.
                # We want to move in direction - (avg grad).
                # So we accumulate -grad_sphere for each active facet containing k.
                # Wait, if multiple facets share a point, and all are active, we sum their contributions?
                # Yes, the subgradient of max(f1, f2) at a point where f1=f2=max is the convex hull of {grad f1, grad f2}.
                # To minimize, we pick a direction in -convex_hull.
                # A common choice is the average: - (grad f1 + grad f2) / 2.
                # So we sum the "descent directions" (-grad_sphere) and divide by count?
                # Or just sum the gradients and take negative?
                # Let's sum the ascent gradients and then negate.
                
                grad[k] += -dir_vec # This is -grad_sphere, which is the descent direction for r_k
            
            # Normalize the gradient to avoid huge steps if many facets active
            # Actually, the magnitude of grad_sphere is 1.
            # If a point is in 3 active facets, the sum of 3 unit vectors has magnitude <= 3.
            # We should normalize the total gradient vector for each point to keep step size consistent?
            # Or just use eta * grad.
            
            # Project gradient to tangent space (it should already be, but numerical errors)
            # grad is sum of tangent vectors.
            # Move points
            current_pts += eta * grad
            
            # Normalize
            norms = np.linalg.norm(current_pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            current_pts = current_pts / norms
            
            # Optional: Check if we are improving
            # The loop already updates best_local if improved.
            
            # Anneal step size
            eta *= 0.995
            
        return best_local_pts, best_local_radius

    # Main Loop: L-BFGS-B on Sum of Squares, then Polish
    # We will do several iterations of (L-BFGS -> Polish -> Check Improvement)
    
    current_pts = pts.copy()
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < deadline - 20.0 and no_improve_count < max_no_improve:
        
        # Phase 1: L-BFGS-B on Sum of Squared Radii
        # This helps find a good local minimum for the "average" error
        try:
            # Use a tighter tolerance and more iterations if time allows
            res = minimize(objective_sum_sq, current_pts.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            if res.success or res.fun < objective_sum_sq(current_pts.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                current_pts = new_pts
        except Exception:
            pass
        
        # Phase 2: Polish Max Radius
        polished_pts, polished_radius = polish_max_radius(current_pts, steps=100, eta=0.05)
        
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_pts = polished_pts.copy()
            current_pts = polished_pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Perturb to escape local minimum
            # Add small random noise
            noise = np.random.normal(0, 0.05, size=current_pts.shape)
            current_pts += noise
            norms = np.linalg.norm(current_pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            current_pts = current_pts / norms
            
    # Final check
    if time.time() < deadline - 5.0:
        # One last polish on the global best
        final_pts, final_radius = polish_max_radius(best_pts, steps=200, eta=0.01)
        if final_radius < best_radius:
            best_radius = final_radius
            best_pts = final_pts.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
