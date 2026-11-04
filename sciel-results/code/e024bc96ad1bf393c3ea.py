import numpy as np
import time
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.linspace(0, 2*np.pi, 7)[:-1]
    
    def get_normals(angle_offset):
        angles = base_angles + angle_offset
        return np.array([[np.cos(a), np.sin(a)] for a in angles])

    def cost_func_cluster(params):
        points = params.reshape(-1, 2)
        # Objective: Minimize the radius of the smallest enclosing circle
        # Approximation: Max distance from centroid
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        radius = np.max(dists)
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2 * 100.0
        return radius + penalty

    best_radius = float('inf')
    best_points = None

    # Optimization loop
    for restart in range(50):
        if restart == 0:
            # Hexagonal lattice initialization
            pts = []
            for q in range(-3, 4):
                for r in range(-3, 4):
                    x = 1.5 * q
                    y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                    pts.append([x, y])
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists)[:N]
            init_pts = pts[indices]
            init_pts += np.random.normal(0, 0.1, init_pts.shape)
        else:
            init_pts = np.random.uniform(-3, 3, (N, 2))
            
        initial_params = init_pts.flatten()
        
        res = minimize(cost_func_cluster, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        final_pts = res.x.reshape(-1, 2)
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(final_pts[i] - final_pts[j])
                if d < MIN_DIST - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            centroid = np.mean(final_pts, axis=0)
            radius = np.max(np.linalg.norm(final_pts - centroid, axis=1))
            if radius < best_radius:
                best_radius = radius
                best_points = final_pts

    if best_points is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Now, find the optimal enclosing regular hexagon for best_points
    # We sweep the rotation angle of the outer hexagon
    best_L = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # The hexagon has 6-fold symmetry, so we only need to check 0 to 60 degrees
    # But the "width" function is periodic with 60 degrees.
    # Let's sample 100 angles
    for theta in np.linspace(0, np.pi/3, 100):
        normals = get_normals(theta)
        # Project points onto normals
        projections = best_points @ normals.T
        # For each direction k, find min and max
        mins = np.min(projections, axis=0)
        maxs = np.max(projections, axis=0)
        widths = maxs - mins
        # The side length L is determined by the max width
        # Width in direction n is L * sqrt(3)
        # So L = max(widths) / sqrt(3)
        L = np.max(widths) / np.sqrt(3)
        
        if L < best_L:
            best_L = L
            best_angle = theta
            
            # Find a valid center
            # Constraints: c . n_k <= mins[k] + h  and c . n_k >= maxs[k] - h
            # where h = L * sqrt(3) / 2
            h = best_L * np.sqrt(3) / 2
            
            # We want a point c that satisfies:
            # maxs[k] - h <= c . n_k <= mins[k] + h
            # Let's try the centroid of the points first
            c_candidate = np.mean(best_points, axis=0)
            
            # Check if c_candidate is valid
            valid_center = True
            projs_c = c_candidate @ normals.T
            for k in range(6):
                if projs_c[k] < maxs[k] - h - 1e-6 or projs_c[k] > mins[k] + h + 1e-6:
                    valid_center = False
                    break
            
            if not valid_center:
                # Find a valid center by intersecting the strips
                # This is a 2D linear programming problem.
                # We can solve it by finding the intersection of the half-planes.
                # Or simpler: try to adjust the centroid.
                # Let's just find the center of the feasible region.
                # The feasible region is a convex polygon.
                # We can find its vertices by intersecting the boundary lines.
                # Boundary lines: c . n_k = maxs[k] - h and c . n_k = mins[k] + h
                # There are 12 lines.
                # We can just pick the average of the midpoints of the intervals?
                # Midpoint m_k = (maxs[k] + mins[k]) / 2
                # Interval for c . n_k is [maxs[k] - h, mins[k] + h]
                # Note: maxs[k] - h <= mins[k] + h is guaranteed.
                # The midpoint of the interval is (maxs[k] + mins[k])/2.
                # If we set c such that c . n_k = (maxs[k] + mins[k])/2 for all k, 
                # this is generally overdetermined.
                # However, we can find a point by minimizing the distance to these midpoints?
                # Let's just use a simple iterative projection or pick a vertex.
                # For robustness, let's find the center of the bounding box of the feasible region?
                # Actually, the centroid of the points is usually very close.
                # Let's try to find the center by averaging the projections?
                # Let's just use the centroid of the 6 "midpoints" in the dual space? No.
                
                # Simple fallback: Use the centroid of the convex hull of the points?
                # Or just shift the centroid slightly?
                # Let's solve the small LP using scipy?
                # Or just pick the point that minimizes the max violation?
                
                # Let's try to find the center by intersecting the two most restrictive constraints.
                # Identify the directions with the smallest feasible interval length.
                # Interval length is (mins[k] + h) - (maxs[k] - h) = 2h - width[k].
                # The tightest constraints have the smallest length.
                # Let's pick the 2 tightest directions and intersect their midlines?
                # Midline for k: c . n_k = (maxs[k] + mins[k]) / 2
                # If we intersect two midlines, we get a point.
                # Let's pick the two directions with the smallest (2h - width[k]).
                lengths = 2*h - widths
                idx = np.argsort(lengths)[:2]
                n1 = normals[idx[0]]
                n2 = normals[idx[1]]
                m1 = (maxs[idx[0]] + mins[idx[0]]) / 2
                m2 = (maxs[idx[1]] + mins[idx[1]]) / 2
                
                # Solve:
                # x n1[0] + y n1[1] = m1
                # x n2[0] + y n2[1] = m2
                A = np.array([n1, n2])
                b = np.array([m1, m2])
                try:
                    c_candidate = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    c_candidate = np.mean(best_points, axis=0)
                    
            best_center = (float(c_candidate[0]), float(c_candidate[1]))

    # Construct output
    inner = [[float(x), float(y), 0.0] for x, y in best_points]
    outer_side = best_L
    return inner, best_center, outer_side, float(best_angle)
