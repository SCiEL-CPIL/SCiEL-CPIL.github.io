import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    def _normalize(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return points / norms

    def _covering_radius(points):
        """Compute the exact covering radius in radians."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return np.pi
        
        max_rad = 0.0
        for simplex in hull.simplices:
            a = points[simplex[0]]
            b = points[simplex[1]]
            c = points[simplex[2]]
            
            # Normal to the plane
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            # Ensure normal points outward (away from origin)
            # The centroid of the triangle is (a+b+c)/3. 
            # If dot(centroid, n) < 0, flip n.
            centroid = (a + b + c) / 3.0
            if np.dot(centroid, n) < 0:
                n = -n
            
            # The circumradius of the spherical triangle is the angle between
            # any vertex and the circumcenter (which is n for the plane).
            # Actually, the circumcenter of the spherical triangle is the normalized
            # vector sum of the vertices ONLY if the triangle is acute and small.
            # However, for the convex hull facet, the normal n is the center of the
            # circle passing through a,b,c on the sphere?
            # No. The intersection of the plane with the sphere is a circle.
            # The center of this circle is n (normalized).
            # The radius of this circle is arccos(dot(a, n)).
            # This is the circumradius of the spherical triangle.
            dot_a = np.dot(a, n)
            dot_a = np.clip(dot_a, -1.0, 1.0)
            rad = np.arccos(dot_a)
            
            if rad > max_rad:
                max_rad = rad
        
        return max_rad

    def _smooth_covering_radius_grad(points, beta):
        """
        Compute a smooth approximation of the covering radius and its gradient.
        Uses log-sum-exp on the cosines of the angles to approximate the maximum angle.
        f = -logsumexp(-beta * cos(theta_i))
        grad f = sum(exp(-beta * cos(theta_i)) * grad(-beta * cos(theta_i))) / sum(exp(-beta * cos(theta_i)))
        
        Here, theta_i is the angle between point p and circumcenter n_i of facet i incident to p?
        Actually, the covering radius is the max over ALL facets of the circumradius.
        It is not incident-specific in the global max, but for gradient descent, we can
        treat the objective as the max over all facets.
        
        Let R = max_i arccos(c_i) where c_i = dot(v_i, n_i).
        We want to minimize R.
        Smooth approx: S = -1/beta * log( sum_i exp(-beta * c_i) ) ... wait.
        arccos(c) is decreasing in c. Minimizing arccos(c) is maximizing c.
        So we want to maximize min_i c_i.
        Let g = min_i c_i.
        Smooth min: -1/beta * log( sum_i exp(-beta * c_i) ).
        We want to maximize this smooth min.
        So objective to minimize: J = 1/beta * log( sum_i exp(-beta * c_i) ).
        
        Gradient of J wrt p_k:
        J = 1/beta * log( sum_i w_i ) where w_i = exp(-beta * c_i).
        dJ/dp = 1/beta * 1/sum(w) * sum_i w_i * d(-beta * c_i)/dp
        dJ/dp = -1/sum(w) * sum_i w_i * d(c_i)/dp
        
        c_i = dot(v_i, n_i).
        n_i is the normal of facet i.
        If facet i does not contain point k, d(n_i)/dp_k = 0, d(v_i)/dp_k = 0.
        So only points in facet i contribute.
        
        For a facet with vertices a,b,c:
        n = normalize( (b-a) x (c-a) )
        This derivative is messy. 
        Alternative: Use the property that for a point on the sphere, the gradient of the distance
        to a fixed point n is simply -n (in the tangent space).
        But n changes as points move.
        
        Let's use a simpler surrogate: 
        The covering radius is determined by the "worst" facet.
        We can use the gradient of the distance from the points to the circumcenters of the facets they belong to.
        For a point p_i, let N_i be the set of facets containing p_i.
        Let r_i = max_{f in N_i} arccos(dot(p_i, n_f)).
        We want to minimize max_i r_i.
        
        This is getting complicated to derive analytically. 
        Let's use a simpler heuristic: 
        Move each point towards the circumcenter of its "worst" incident facet.
        This is the "relaxation" method.
        
        However, the directive suggests "stochastic gradient ascent on the Gram matrix".
        Let's try a different approach: 
        Maximize the minimum pairwise distance? No, that's packing, not covering.
        Covering is about the Voronoi cells.
        
        Let's stick to a robust local search:
        1. Compute convex hull.
        2. For each point, find the incident facet with the largest circumradius.
        3. Move the point towards the circumcenter of that facet.
        
        To make it faster and more robust, we can use scipy's minimize on a smooth surrogate.
        Surrogate: Sum of exp(beta * (angle_i - R)) ? No.
        
        Let's implement the "Move towards worst circumcenter" step with adaptive step size.
        """
        pass # Placeholder, will implement specific logic in main loop

    # 1. Initial Configuration: Fibonacci Spiral
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    best_pts = fibonacci_spiral(N)
    best_radius = _covering_radius(best_pts)

    # Optimization Loop
    step_size = 0.1
    min_step_size = 1e-4
    cooldown = 0
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute hull and facet data
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            # If hull fails, perturb
            best_pts = _normalize(best_pts + 0.01 * np.random.randn(N, 3))
            continue
        
        # Compute circumcenter and radius for each facet
        facets = hull.simplices
        n_facets = len(facets)
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        for i, simplex in enumerate(facets):
            a = best_pts[simplex[0]]
            b = best_pts[simplex[1]]
            c = best_pts[simplex[2]]
            
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            centroid = (a + b + c) / 3.0
            if np.dot(centroid, n) < 0:
                n = -n
            
            circumcenters[i] = n
            dot_a = np.clip(np.dot(a, n), -1.0, 1.0)
            radii[i] = np.arccos(dot_a)
        
        # Map point to worst incident facet
        # point_worst_facet[p] = index of facet with max radius among those containing p
        # point_worst_radius[p] = that max radius
        point_worst_facet = np.full(N, -1, dtype=int)
        point_worst_radius = np.zeros(N)
        
        # Efficient mapping: iterate facets, update points
        for i, simplex in enumerate(facets):
            if radii[i] == 0: continue
            for p_idx in simplex:
                if radii[i] > point_worst_radius[p_idx]:
                    point_worst_radius[p_idx] = radii[i]
                    point_worst_facet[p_idx] = i
        
        current_radius = np.max(point_worst_radius)
        
        if current_radius < best_radius:
            best_radius = current_radius
            # Step size can be maintained or slightly increased if improving
            # step_size = min(step_size * 1.05, 0.2)
        else:
            # Stagnation or worsening
            cooldown += 1
            if cooldown > 10:
                step_size = max(step_size * 0.9, min_step_size)
                # Random perturbation to escape local minima
                if cooldown % 50 == 0:
                    best_pts = _normalize(best_pts + 0.01 * np.random.randn(N, 3))
                    cooldown = 0
            # If we are stuck, maybe reduce step size more aggressively
            if current_radius > best_radius * 1.01:
                 step_size = max(step_size * 0.5, min_step_size)

        # Update points
        new_pts = best_pts.copy()
        for i in range(N):
            f_idx = point_worst_facet[i]
            if f_idx >= 0:
                target = circumcenters[f_idx]
                # Geodesic step towards target
                # new = normalize(p + alpha * (target - p))
                # To avoid overshooting, we can project
                new_pts[i] = best_pts[i] + step_size * (target - best_pts[i])
                new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
            else:
                # Point is inside hull? Shouldn't happen for convex hull vertices,
                # but if N=50 and hull has <50 vertices, some points are inside.
                # Move them towards the nearest facet circumcenter or randomly.
                # For simplicity, random small move
                new_pts[i] = best_pts[i] + 0.01 * np.random.randn(3)
                new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
        
        best_pts = _normalize(new_pts)

    # Final refinement: try a few random restarts if time permits
    if time.time() - start_time < time_limit - 20:
        for restart in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            # Perturb best solution
            perturbed = _normalize(best_pts + 0.1 * np.random.randn(N, 3))
            cur_pts = perturbed
            cur_rad = _covering_radius(cur_pts)
            if cur_rad < best_radius:
                best_pts = cur_pts
                best_radius = cur_rad
            # Short local search on perturbed
            for _ in range(200):
                if time.time() - start_time > time_limit - 5:
                    break
                try:
                    hull = ConvexHull(cur_pts)
                except:
                    break
                facets = hull.simplices
                n_facets = len(facets)
                circumcenters = np.zeros((n_facets, 3))
                radii = np.zeros(n_facets)
                for i, simplex in enumerate(facets):
                    a = cur_pts[simplex[0]]
                    b = cur_pts[simplex[1]]
                    c = cur_pts[simplex[2]]
                    v1 = b - a
                    v2 = c - a
                    n = np.cross(v1, v2)
                    norm_n = np.linalg.norm(n)
                    if norm_n < 1e-10: continue
                    n = n / norm_n
                    centroid = (a + b + c) / 3.0
                    if np.dot(centroid, n) < 0: n = -n
                    circumcenters[i] = n
                    dot_a = np.clip(np.dot(a, n), -1.0, 1.0)
                    radii[i] = np.arccos(dot_a)
                
                point_worst_facet = np.full(N, -1, dtype=int)
                point_worst_radius = np.zeros(N)
                for i, simplex in enumerate(facets):
                    if radii[i] == 0: continue
                    for p_idx in simplex:
                        if radii[i] > point_worst_radius[p_idx]:
                            point_worst_radius[p_idx] = radii[i]
                            point_worst_facet[p_idx] = i
                
                new_pts = cur_pts.copy()
                for i in range(N):
                    f_idx = point_worst_facet[i]
                    if f_idx >= 0:
                        target = circumcenters[f_idx]
                        new_pts[i] = cur_pts[i] + 0.05 * (target - cur_pts[i])
                        new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
                cur_pts = _normalize(new_pts)
                
                cur_rad = np.max(point_worst_radius)
                if cur_rad < best_radius:
                    best_radius = cur_rad
                    best_pts = cur_pts.copy()

    return best_pts
