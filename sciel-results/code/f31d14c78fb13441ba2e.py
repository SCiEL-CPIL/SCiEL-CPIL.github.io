import numpy as np
import time
from scipy.spatial import Voronoi, voronoi_plot_2d # voronoi_plot not needed
from scipy.optimize import minimize

N = 26

def construct():
    # Initialize on a grid
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            centers.append([x, y])
    centers = np.array(centers[:N])
    
    # We will iterate:
    # 1. Compute Voronoi diagram
    # 2. For each site, find the largest circle that fits in its Voronoi cell intersected with [0,1]^2
    # 3. Move site to the center of that circle
    # 4. Set radius to the circle's radius
    
    # To find the center of the largest inscribed circle in a convex polygon,
    # we can use a simple local search or solve a small optimization problem.
    # Given N=26, we can afford a small optimization per circle per iteration.
    
    def max_inscribed_circle_in_polygon(points):
        # points: (M, 2) array of vertices of a convex polygon
        # We want to find (cx, cy, r) such that the circle is inside the polygon.
        # This is equivalent to finding the point that maximizes the minimum distance to the edges.
        
        # Simple approach: Start from centroid, do gradient ascent on min distance to edges.
        # Or use scipy.optimize.minimize on -min_dist.
        
        # Centroid
        cx, cy = np.mean(points, axis=0)
        
        def neg_min_dist(params):
            cx, cy = params
            # Calculate distance to each edge
            min_d = np.inf
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i+1)%len(points)]
                # Distance from (cx,cy) to line segment p1-p2
                # Since it's convex and we start inside, distance to infinite line is sufficient if we ensure we stay inside.
                # But to be safe, let's use distance to segment.
                # Vector p1->p2
                v = p2 - p1
                w = [cx - p1[0], cy - p1[1]]
                c1 = np.dot(w, v)
                if c1 <= 0:
                    d = np.linalg.norm(w)
                else:
                    c2 = np.dot(v, v)
                    if c2 <= c1:
                        d = np.linalg.norm(w - v)
                    else:
                        t = c1 / c2
                        d = np.linalg.norm(w - t*v)
                if d < min_d:
                    min_d = d
            return -min_d
            
        # Optimize
        res = minimize(neg_min_dist, [cx, cy], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 50})
        cx, cy = res.x
        r = -res.fun
        return cx, cy, r

    def get_voronoi_cells(centers):
        # Returns list of polygons for each center
        # We need to handle the fact that Voronoi regions for points near boundary are unbounded.
        # We intersect them with the square [0,1]^2.
        
        try:
            vor = Voronoi(centers)
        except Exception:
            return [np.array([[0,0],[1,0],[1,1],[0,1]]) for _ in range(N)] # Fallback: whole square

        cells = []
        for i in range(N):
            # Get region index
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region:
                # Unbounded region, need to clip
                # This is complex. 
                # Alternative: Use a bounding box larger than [0,1]^2 for Voronoi, then clip.
                # Or, simpler: Just compute the convex hull of the Voronoi vertices + square corners? No.
                
                # Let's use a different strategy: 
                # Instead of scipy Voronoi which has issues with unbounded regions,
                # we can use the fact that N is small.
                # We can compute the Voronoi cell by intersecting half-planes.
                # Half-plane for center i: Points closer to i than to any j.
                # And points inside [0,1]^2.
                
                # Let's implement a simple half-plane intersection.
                # Start with the square [0,1]^2.
                # For each other center j, intersect with the half-plane closer to i.
                
                def intersect_half_plane(poly, a, b, c, d):
                    # a*x + b*y + c <= d
                    # poly is list of (x,y)
                    new_poly = []
                    n = len(poly)
                    if n == 0: return []
                    for i in range(n):
                        p1 = poly[i]
                        p2 = poly[(i+1)%n]
                        val1 = a*p1[0] + b*p1[1] + c - d
                        val2 = a*p2[0] + b*p2[1] + c - d
                        
                        if val1 <= 0:
                            new_poly.append(p1)
                        if val1 * val2 < 0:
                            # Intersection
                            t = val1 / (val1 - val2)
                            x = p1[0] + t * (p2[0] - p1[0])
                            y = p1[1] + t * (p2[1] - p1[1])
                            new_poly.append((x, y))
                    return new_poly

                # Initial polygon: [0,1]^2
                poly = [(0,0), (1,0), (1,1), (0,1)]
                
                # Intersect with half-planes from other centers
                for j in range(N):
                    if i == j: continue
                    xi, yi = centers[i]
                    xj, yj = centers[j]
                    # Midpoint line: (xi-xj)x + (yi-yj)y = 0.5*((xi^2+yi^2) - (xj^2+yj^2))
                    # We want points closer to i: (xi-xj)x + (yi-yj)y <= 0.5*((xi^2+yi^2) - (xj^2+yj^2))
                    a = xi - xj
                    b = yi - yj
                    c = 0
                    d = 0.5 * ((xi*xi + yi*yi) - (xj*xj + yj*yj))
                    
                    poly = intersect_half_plane(poly, a, b, c, d)
                    if len(poly) < 3:
                        # Degenerate, break early
                        break
                
                if len(poly) >= 3:
                    cells.append(np.array(poly))
                else:
                    # Fallback: small square around center
                    eps = 0.01
                    cells.append(np.array([[xi-eps, yi-eps], [xi+eps, yi-eps], [xi+eps, yi+eps], [xi-eps, yi+eps]]))
            else:
                # Bounded region
                verts = [vor.vertices[k] for k in region]
                # Clip to [0,1]^2 just in case (should be inside if centers are inside)
                # But Voronoi vertices might be outside if centers are near boundary?
                # Actually, if centers are inside [0,1]^2, the Voronoi vertices of bounded regions are inside.
                # However, we must ensure the cell is within [0,1]^2.
                # The half-plane method above handles the square boundary explicitly.
                # So let's just use the half-plane method for ALL regions to be consistent and robust.
                # Re-do for i using half-plane method to avoid scipy quirks.
                
                poly = [(0,0), (1,0), (1,1), (0,1)]
                for j in range(N):
                    if i == j: continue
                    xi, yi = centers[i]
                    xj, yj = centers[j]
                    a = xi - xj
                    b = yi - yj
                    c = 0
                    d = 0.5 * ((xi*xi + yi*yi) - (xj*xj + yj*yj))
                    poly = intersect_half_plane(poly, a, b, c, d)
                    if len(poly) < 3:
                        break
                
                if len(poly) >= 3:
                    cells.append(np.array(poly))
                else:
                    eps = 0.01
                    cells.append(np.array([[xi-eps, yi-eps], [xi+eps, yi-eps], [xi+eps, yi+eps], [xi-eps, yi+eps]]))
        return cells

    # Main Loop
    start_time = time.time()
    time_limit = 110
    
    for iter in range(50): # Max iterations
        if time.time() - start_time > time_limit:
            break
            
        cells = get_voronoi_cells(centers)
        
        new_centers = np.zeros_like(centers)
        new_radii = np.zeros(N)
        
        for i in range(N):
            poly = cells[i]
            if len(poly) < 3:
                # Keep old position and radius 0
                new_centers[i] = centers[i]
                new_radii[i] = 0.01
                continue
            
            cx, cy, r = max_inscribed_circle_in_polygon(poly)
            new_centers[i] = [cx, cy]
            new_radii[i] = r
            
        # Check convergence
        if np.linalg.norm(new_centers - centers) < 1e-5:
            centers = new_centers
            break
            
        centers = new_centers
        
    # Final radii calculation
    final_radii = np.zeros(N)
    for i in range(N):
        # Max radius is min distance to boundary and other circles
        r_i = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        for j in range(N):
            if i == j: continue
            dist = np.linalg.norm(centers[i] - centers[j])
            r_i = min(r_i, dist / 2.0) # Wait, if we just set radii to dist/2, they touch.
            # But we want to maximize sum. 
            # The Lloyd's step already set them to max inscribed in Voronoi cell.
            # The Voronoi cell boundary is the perpendicular bisector.
            # So the max radius in the cell is exactly half the distance to the nearest neighbor?
            # Yes, for the direction towards the nearest neighbor.
            # But the cell is a polygon. The max inscribed circle radius is the min distance to the cell edges.
            # The cell edges are bisectors. So min distance to bisector = min dist to neighbor / 2.
            # And min distance to square boundary.
            # So the radius calculated in max_inscribed_circle_in_polygon is correct.
            pass
            
    # Re-calculate radii robustly from final centers
    # The Lloyd's algorithm sets radius = min dist to cell boundary.
    # Let's just re-verify and adjust to ensure no overlap.
    
    # Actually, the `new_radii` from the last iteration are the ones corresponding to `centers`.
    # But `centers` was updated to `new_centers`.
    # Let's compute the radii one last time to be sure.
    
    # We can just use the radii computed in the last Lloyd step.
    # But `new_radii` was computed for `new_centers`.
    # So we return `new_centers` and `new_radii`.
    
    # Wait, the loop ends when `centers` is updated to `new_centers`.
    # So `new_radii` corresponds to the final `centers`.
    
    # One final check: Ensure validity
    # If any overlap, shrink.
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < new_radii[i] + new_radii[j]:
                # Shrink both
                scale = dist / (new_radii[i] + new_radii[j]) * 0.99
                new_radii[i] *= scale
                new_radii[j] *= scale
                
    result = np.column_stack((centers, new_radii))
    return result
