import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE - elapsed()

    # 1. Define the 11 inner hexagon centers (Flower Packing)
    centers = np.array([
        [0.0, 0.0]
    ])
    # 6 ring hexagons
    r1 = np.sqrt(3)
    for k in range(6):
        angle = k * np.pi / 3
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    # 4 outer hexagons
    r2 = 3.0
    for k in range(4):
        angle = np.pi / 6 + k * np.pi / 2
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
    
    n_hex = len(centers)
    
    # Base vertices of a unit hexagon (centered at origin, rotation 0)
    # Vertex k at angle 60*k degrees
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)
    ])

    def get_all_vertices(alpha):
        """
        Returns array of shape (n_hex * 6, 2) with vertices of all inner hexagons.
        alpha: rotation angle in radians applied to all inner hexagons.
        """
        c = np.cos(alpha)
        s = np.sin(alpha)
        rot = np.array([[c, -s], [s, c]])
        
        all_verts = []
        for i in range(n_hex):
            verts = base_vertices @ rot.T + centers[i]
            all_verts.append(verts)
        return np.vstack(all_verts)

    def calc_S(theta, verts):
        """
        Calculate minimum side length S of outer hexagon (centered at 0,0, rotation theta)
        enclosing the given vertices.
        """
        # Normals of the outer hexagon
        # Side k is perpendicular to normal at angle theta + 30 + k*60
        normals = []
        for k in range(6):
            angle = theta + np.pi / 6 + k * np.pi / 3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals) # shape (6, 2)
        
        # Project all vertices onto all normals
        # dots: (n_points, 6)
        dots = verts @ normals.T
        
        # Max projection
        max_proj = np.max(dots)
        
        # Required S: apothem = S * sqrt(3) / 2
        # max_proj <= apothem  => S >= 2 * max_proj / sqrt(3)
        S = 2.0 * max_proj / np.sqrt(3.0)
        
        return S

    def overlap_penalty(alpha):
        """
        Calculate penalty for overlapping inner hexagons.
        Returns 0 if no overlap, positive value otherwise.
        """
        verts = get_all_vertices(alpha)
        # Reshape to (n_hex, 6, 2)
        verts = verts.reshape(n_hex, 6, 2)
        
        pen = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                # Check if hex i and j overlap
                # Simple check: min distance between polygons
                # For hexagons, if they are tangent, distance is 0.
                # We can check if any vertex of i is inside j or vice versa,
                # or if edges intersect.
                # Given the tight packing, a simpler metric is distance between centers vs threshold?
                # No, rotation matters.
                
                # Use a fast bounding box check first
                min_x_i = np.min(verts[i, :, 0])
                max_x_i = np.max(verts[i, :, 0])
                min_y_i = np.min(verts[i, :, 1])
                max_y_i = np.max(verts[i, :, 1])
                
                min_x_j = np.min(verts[j, :, 0])
                max_x_j = np.max(verts[j, :, 0])
                min_y_j = np.min(verts[j, :, 1])
                max_y_j = np.max(verts[j, :, 1])
                
                if max_x_i < min_x_j or max_x_j < min_x_i or max_y_i < min_y_j or max_y_j < min_y_i:
                    continue # No overlap possible
                
                # Detailed check: Point in polygon for vertices
                # If any vertex of i is inside j, they overlap (or touch)
                # We want to penalize *interior* overlap. Touching is allowed (distance 0).
                # However, for optimization, we want to avoid negative distance.
                # Let's use a simplified distance metric for speed if needed,
                # but for 11 hexes, exact check is fine.
                
                # Check vertex of i inside j
                for v in verts[i]:
                    if point_in_poly(v, verts[j]):
                        return 1000.0 # High penalty
                # Check vertex of j inside i
                for v in verts[j]:
                    if point_in_poly(v, verts[i]):
                        return 1000.0
                        
                # Check edge intersections
                # If edges intersect, they overlap.
                for e1_i in range(6):
                    p1 = verts[i, e1_i]
                    p2 = verts[i, (e1_i + 1) % 6]
                    for e2_j in range(6):
                        p3 = verts[j, e2_j]
                        p4 = verts[j, (e2_j + 1) % 6]
                        if seg_seg_intersect(p1, p2, p3, p4):
                            return 1000.0
                            
        return 0.0

    def point_in_poly(pt, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > pt[1]) != (yj > pt[1])) and \
               (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def seg_seg_intersect(p1, p2, p3, p4):
        # Check if segments p1-p2 and p3-p4 intersect
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def objective(params):
        if remaining() < 5.0:
            return 1e10
            
        theta = params[0]
        alpha = params[1]
        
        # Get vertices
        verts = get_all_vertices(alpha)
        
        # Calculate S
        S = calc_S(theta, verts)
        
        # Calculate overlap penalty
        pen = overlap_penalty(alpha)
        
        return S + pen

    # Initial guess
    # Theta = 0, Alpha = 0
    x0 = np.array([0.0, 0.0])
    
    # Bounds
    # Theta in [0, pi/3)
    # Alpha in [0, pi/3)
    bounds = [(0, np.pi/3), (0, np.pi/3)]
    
    best_score = float('inf')
    best_params = x0
    
    # Multi-start optimization
    n_starts = 10
    for i in range(n_starts):
        if remaining() < 10.0:
            break
            
        if i == 0:
            x_start = x0
        else:
            x_start = np.random.uniform(0, np.pi/3, 2)
            
        try:
            res = minimize(objective, x_start, method='Powell', bounds=bounds, 
                           options={'maxiter': 100, 'disp': False, 'ftol': 1e-9, 'xtol': 1e-9})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Extract best solution
    theta_best = best_params[0]
    alpha_best = best_params[1]
    
    # Final check: ensure no overlap
    if overlap_penalty(alpha_best) > 0:
        # Fallback to alpha=0 if rotation caused overlap
        alpha_best = 0.0
        # Re-optimize theta for alpha=0
        def obj_theta(theta_val):
            verts = get_all_vertices(0.0)
            return calc_S(theta_val, verts)
            
        res_t = minimize(obj_theta, 0.0, method='Nelder-Mead', options={'maxiter': 1000})
        theta_best = res_t.x[0]
        best_score = obj_theta(theta_best)
        
    # Construct return values
    inner = []
    for i in range(n_hex):
        x, y = centers[i]
        inner.append([x, y, np.rad2deg(alpha_best)])
        
    side = best_score
    # Add tiny margin
    side += 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_best)

if __name__ == "__main__":
    construct()
