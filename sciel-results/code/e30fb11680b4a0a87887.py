import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Hexagonal lattice placement
    # For unit hexagons, center-to-center distance for edge-sharing is sqrt(3)
    # Axial coordinates: (q, r) -> (x, y) = (sqrt(3)*(q + r/2), 1.5*r)
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Generate hexagonal lattice points
    # Try different radii and find minimal containing hexagon
    max_r = 10
    for r in range(0, max_r + 1):
        if time.time() - start_time > deadline - 5:
            break
        
        # Generate all hex positions within hexagonal radius r
        centers = []
        for q in range(-r, r + 1):
            for rr in range(max(-r, -q - r), min(r, -q + r) + 1):
                x = math.sqrt(3) * (q + rr / 2.0)
                y = 1.5 * rr
                centers.append((x, y))
        
        if len(centers) < N:
            continue
        
        # Take first N centers (centered arrangement)
        # Sort by distance from origin to get compact arrangement
        centers.sort(key=lambda c: c[0]**2 + c[1]**2)
        selected = centers[:N]
        
        # Compute minimal containing hexagon side length
        # For a set of points, the minimal containing regular hexagon (with given orientation)
        # can be found by checking projections onto 6 directions
        
        # The 6 normal directions for a regular hexagon with rotation angle
        # For a hexagon with flat top (angle 0), normals are at 0, 60, 120, 180, 240, 300 degrees
        
        # Try several rotations of the containing hexagon
        best_for_this_set = float('inf')
        for angle_deg in range(0, 60, 5):
            angle_rad = math.radians(angle_deg)
            
            # 6 normal directions
            normals = []
            for i in range(6):
                theta = angle_rad + i * math.pi / 3
                normals.append((math.cos(theta), math.sin(theta)))
            
            # For each normal, find max and min projection
            valid = True
            total_width = 0
            for nx, ny in normals:
                projections = [x * nx + y * ny for x, y in selected]
                width = max(projections) - min(projections)
                total_width += width
            
            # The side length of containing hexagon: 
            # For a regular hexagon, the width in direction normal to a side is 2*apothem = 2*(s*sqrt(3)/2) = s*sqrt(3)
            # So s = width / sqrt(3) for each direction, take max
            # But we need the hexagon to contain all points, so for each of the 6 directions,
            # the required apothem is max_projection - min_projection / 2... 
            # Actually, the containing hexagon centered at origin: for each normal direction,
            # we need apothem >= max(|projection|)
            # But we can shift the center. Optimal center is midpoint of min and max for each pair.
            
            # For a given orientation, the minimal side length hexagon containing the points:
            # The hexagon is defined by 6 half-planes. We need to find the minimal s such that
            # there exists a center (cx, cy) where for all 6 normals:
            #   |dot(p - c, n_i)| <= s * sqrt(3) / 2 for all p in selected
            
            # This is a linear programming problem. For simplicity, use the center as centroid
            cx = sum(x for x, y in selected) / len(selected)
            cy = sum(y for x, y in selected) / len(selected)
            
            max_apothem_needed = 0
            for nx, ny in normals:
                max_proj = max((x - cx) * nx + (y - cy) * ny for x, y in selected)
                min_proj = min((x - cx) * nx + (y - cy) * ny for x, y in selected)
                apothem_needed = max(abs(max_proj), abs(min_proj))
                max_apothem_needed = max(max_apothem_needed, apothem_needed)
            
            # apothem = s * sqrt(3) / 2, so s = 2 * apothem / sqrt(3)
            side = 2 * max_apothem_needed / math.sqrt(3)
            
            if side < best_for_this_set:
                best_for_this_set = side
        
        if best_for_this_set < best_side:
            best_side = best_for_this_set
            best_inner = [[x, y, 0.0] for x, y in selected]
            best_center = (0.0, 0.0)
            best_angle = 0.0
    
    # If no valid solution found, fall back to a simple arrangement
    if best_inner is None:
        # Simple linear arrangement
        best_inner = [[math.sqrt(3) * i, 0.0, 0.0] for i in range(N)]
        best_side = N * math.sqrt(3) + 1  # rough estimate
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
