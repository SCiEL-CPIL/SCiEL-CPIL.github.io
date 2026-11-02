import math
import time
import numpy as np
import random

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Grid parameters
    # We search for inner hex centers in a region around the origin.
    # The optimal outer hexagon side for N=11 is likely around 4-5.
    # So inner centers should be within radius ~3.
    # Grid step: 0.05 is a good balance between resolution and search space.
    step = 0.05
    max_radius = 4.0
    
    # Pre-compute grid points
    # To save time, we only generate points within the max_radius circle
    grid_points = []
    r = int(max_radius / step)
    for i in range(-r, r + 1):
        x = i * step
        for j in range(-r, r + 1):
            y = j * step
            if x*x + y*y <= max_radius*max_radius:
                grid_points.append((x, y))
                
    # Shuffle grid points to randomize search order, but keep some structure?
    # Actually, for greedy packing, order matters.
    # We will try multiple random orderings or specific heuristics.
    
    def is_valid_packing(positions, angles):
        """
        Check if the given positions and angles form a valid packing
        inside a hypothetical outer hexagon.
        For the grid search, we assume all inner hexagons are aligned (angle=0)
        to simplify the collision check significantly.
        If a valid packing is found with aligned hexagons, we can then
        optimize the outer hexagon size.
        
        Note: Rotations might allow tighter packing, but for a "different strategy",
        restricting to aligned hexagons on a grid is a valid structural change.
        If this fails to beat the current score, it's acceptable as a first try.
        """
        n = len(positions)
        for i in range(n):
            x1, y1 = positions[i]
            for j in range(i+1, n):
                x2, y2 = positions[j]
                # Quick reject
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                # Minimum distance between centers of two disjoint unit hexagons (aligned)
                # is sqrt(3) * 1? No.
                # Distance between centers of two touching aligned hexagons is 2 * apothem = sqrt(3).
                # Wait, apothem is sqrt(3)/2. Distance between parallel sides is sqrt(3).
                # So if dist < sqrt(3) - epsilon, they overlap?
                # Actually, the minimum distance between centers of two non-overlapping
                # aligned unit hexagons is sqrt(3).
                # Let's verify:
                # Hexagon vertices at (1,0), (0.5, sqrt(3)/2)...
                # Two hexagons centered at (0,0) and (sqrt(3), 0).
                # Rightmost point of left hex: (1,0).
                # Leftmost point of right hex: (sqrt(3)-1, 0) ~ (0.732, 0).
                # 1 > 0.732, so they overlap?
                # No, the "width" of a hexagon along the x-axis (vertex to vertex) is 2.
                # The "width" along the y-axis (flat to flat) is sqrt(3).
                # If aligned, the minimum center distance to avoid overlap depends on relative angle.
                # For aligned hexagons, the Minkowski sum is a hexagon with side length 2.
                # The distance from center to boundary of Minkowski sum in direction theta is:
                # h(theta) = 1 + 1 = 2? No.
                # Support function of unit hex: h(t) = max(dot(v, t)) for v in hex.
                # If t is (1,0), h(t)=1.
                # So distance between centers must be >= h_H1(t) + h_H2(-t) for all t.
                # For aligned hexagons, the critical directions are the normal directions of the sides.
                # Normals at 30, 90, 150 deg.
                # In direction 90 deg (vertical), support is sqrt(3)/2.
                # So vertical distance must be >= sqrt(3)/2 + sqrt(3)/2 = sqrt(3).
                # In direction 30 deg, support is 1?
                # Vertex at 0 deg is (1,0). Dot with (cos30, sin30) = sqrt(3)/2.
                # Vertex at 60 deg is (0.5, sqrt(3)/2). Dot = 0.5*sqrt(3)/2 + sqrt(3)/2*0.5 = sqrt(3)/2.
                # So max support in direction 30 is sqrt(3)/2?
                # Let's check vertex (1,0). Angle 0.
                # Normal at 30 deg is (sqrt(3)/2, 1/2).
                # Dot((1,0), (sqrt(3)/2, 1/2)) = sqrt(3)/2.
                # So support is sqrt(3)/2 in all normal directions?
                # Yes, for a regular hexagon, the apothem is constant.
                # So the Minkowski sum of two aligned unit hexagons is a regular hexagon
                # with apothem 2 * (sqrt(3)/2) = sqrt(3).
                # The distance from center to any point on the boundary of the Minkowski sum
                # is at least the apothem sqrt(3).
                # So if center distance < sqrt(3), they overlap.
                # If center distance >= sqrt(3), they might still overlap if the direction is wrong?
                # No, the Minkowski sum is convex. If the vector between centers is inside
                # the Minkowski sum, they overlap.
                # The Minkowski sum contains the disk of radius sqrt(3) (apothem).
                # So if dist < sqrt(3), definitely overlap.
                # If dist > circumradius of Minkowski sum (which is 2*1=2? No, circumradius is 2*1=2?
                # Vertices of Minkowski sum are sums of vertices.
                # (1,0) + (1,0) = (2,0). Dist 2.
                # So if dist > 2, definitely no overlap.
                # Between sqrt(3) and 2, it depends on angle.
                # For simplicity and speed in grid search, we can use a conservative check.
                # Or use the exact check for aligned hexagons.
                
                # Exact check for aligned hexagons:
                # Transform vector (dx, dy) to local frame? No, they are aligned.
                # Check if (dx, dy) is inside the Minkowski sum hexagon.
                # The Minkowski sum is a hexagon with apothem sqrt(3) and circumradius 2.
                # Its vertices are at (2,0), (1, sqrt(3)), (-1, sqrt(3)), etc.
                # Let's just use the point-in-hexagon check for the Minkowski sum.
                # Or simpler:
                # The condition for non-overlap of two aligned unit hexagons centered at (0,0) and (dx,dy)
                # is that (dx,dy) is NOT inside the hexagon defined by:
                # |x| <= 1, |y| <= sqrt(3)/2, |x + y/sqrt(3)| <= 1? No.
                # The Minkowski sum of two unit hexagons (side 1) is a hexagon with side 2.
                # So we can just check if (dx, dy) is inside a hexagon of side 2 centered at origin.
                # If it is inside, they overlap. If outside, they are disjoint.
                
                # Check if (dx, dy) is inside hexagon of side 2.
                # Apothem of side-2 hex is 2 * sqrt(3)/2 = sqrt(3).
                # Circumradius is 2.
                # We can reuse the _point_in_hex_dist_sq logic but for side 2.
                # Or just:
                # max(|dx|, |dy|/sqrt(3), |dx/2 + dy/sqrt(3)|) <= 1 ? No, that's for side 1?
                # Let's stick to the distance check.
                # If dist_sq < 3.0: # sqrt(3)^2
                    # They definitely overlap.
                # If dist_sq > 4.0: # 2^2
                    # They definitely don't overlap.
                # Else:
                    # Need exact check.
                
                if dist_sq < 3.0:
                    return False
                if dist_sq > 4.0:
                    continue
                # Exact check: is (dx, dy) inside hexagon of side 2?
                # Hexagon of side 2 has vertices at (2,0), (1, sqrt(3)), etc.
                # Inequalities:
                # 1. |x| <= 2 * cos(30) = sqrt(3)? No.
                # Let's use the support function.
                # For a hexagon of side S, apothem is S*sqrt(3)/2.
                # Here S=2, apothem = sqrt(3).
                # Normals at 30, 90, 150.
                # proj1 = dx*cos(30) + dy*sin(30)
                # proj2 = dx*cos(90) + dy*sin(90) = dy
                # proj3 = dx*cos(150) + dy*sin(150)
                # If any |proj| > sqrt(3), it's outside.
                c30 = math.sqrt(3)/2
                s30 = 0.5
                c90 = 0.0
                s90 = 1.0
                c150 = -math.sqrt(3)/2
                s150 = 0.5
                
                p1 = dx*c30 + dy*s30
                p2 = dx*c90 + dy*s90
                p3 = dx*c150 + dy*s150
                
                if abs(p1) > math.sqrt(3) or abs(p2) > math.sqrt(3) or abs(p3) > math.sqrt(3):
                    continue # Outside, so no overlap
                else:
                    return False # Inside, so overlap
            
        return True

    # Greedy placement
    # We will try to place N hexagons.
    # To improve the chance of finding a good packing, we can try different "seeds" or random permutations.
    # But N=11 is small.
    # We can use a simple backtracking search on the grid.
    # However, the grid has ~2000 points. 11! is too big.
    # We need a heuristic.
    
    # Strategy:
    # 1. Start with center (0,0).
    # 2. Add neighbors in a spiral order, checking for collisions.
    # 3. If stuck, backtrack.
    
    # To make it "different", let's use a randomized greedy approach with multiple restarts.
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    # We assume aligned hexagons (angle 0) for the inner ones to simplify.
    # The outer hexagon can be rotated.
    
    def calculate_min_outer_side(positions):
        """
        Given a set of non-overlapping aligned unit hexagons,
        find the minimum side length of a regular hexagon that contains them all.
        The outer hexagon can be rotated and translated.
        
        For a set of points, the minimum enclosing hexagon is a complex problem.
        However, we can approximate by finding the bounding box and using that.
        Or, we can optimize the outer hexagon parameters (cx, cy, side, angle)
        using a fast local search.
        """
        if not positions:
            return 0.0, (0,0), 0.0, 0.0
            
        # Initial guess for outer center: centroid
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        
        # Initial side: large enough
        max_dist = max(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in positions)
        side = max_dist + 1.0
        
        # Optimize outer hexagon
        # We want to minimize 'side' such that all inner vertices are inside.
        # Since inner hexagons are aligned, their vertices are known.
        
        def outer_penalty(params):
            ocx, ocy, os, oang = params
            pen = 0.0
            for p in positions:
                # Check vertices of inner hex at p
                # Inner hex is aligned (angle 0)
                # Vertices: (1,0), (0.5, sqrt(3)/2), etc. relative to p
                # Actually, we can just check the 6 vertices.
                for i in range(6):
                    theta = i * math.pi / 3
                    vx = p[0] + math.cos(theta)
                    vy = p[1] + math.sin(theta)
                    # Check if (vx, vy) is inside outer hex
                    # Transform to local frame of outer hex
                    dx = vx - ocx
                    dy = vy - ocy
                    a = math.radians(oang)
                    rx = dx * math.cos(-a) - dy * math.sin(-a)
                    ry = dx * math.sin(-a) + dy * math.cos(-a)
                    
                    apothem = os * math.sqrt(3) / 2
                    # Check 3 strips
                    for k in range(3):
                        theta_n = math.radians(30 + k * 60)
                        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
                        if proj > apothem:
                            pen += (proj - apothem)**2
                        if proj < -apothem:
                            pen += (-apothem - proj)**2
            return pen

        # Simple local search for outer hex
        # Variables: cx, cy, side, angle
        # We will do a few iterations of coordinate descent or Nelder-Mead.
        # To save time, we'll use a coarse grid for angle and center, then refine side.
        
        # Angle candidates: 0, 30, 60 (symmetry)
        # Center candidates: around centroid
        # Side: binary search or gradient
        
        # For speed, let's just try a few angles and centers.
        min_side = float('inf')
        best_params = None
        
        for oang in [0, 30, 60]:
            # Try a small grid around centroid
            for dx in np.arange(-0.5, 0.51, 0.1):
                for dy in np.arange(-0.5, 0.51, 0.1):
                    ocx = cx + dx
                    ocy = cy + dy
                    
                    # Find min side for this center and angle
                    # Side must be large enough to contain all vertices.
                    # For each vertex, calculate required side.
                    # Required side for a point at distance d and angle phi relative to hex axis:
                    # The support function is h(t) = side * sqrt(3)/2 * max(...)?
                    # Actually, if a point is at local coords (rx, ry),
                    # the minimum apothem needed is max over normals of proj.
                    # apothem_min = max( |proj1|, |proj2|, |proj3| )
                    # side_min = apothem_min / (sqrt(3)/2)
                    
                    a_rad = math.radians(oang)
                    max_apothem_needed = 0.0
                    for p in positions:
                        for i in range(6):
                            theta = i * math.pi / 3
                            vx = p[0] + math.cos(theta)
                            vy = p[1] + math.sin(theta)
                            dxv = vx - ocx
                            dyv = vy - ocy
                            rx = dxv * math.cos(-a_rad) - dyv * math.sin(-a_rad)
                            ry = dxv * math.sin(-a_rad) + dyv * math.cos(-a_rad)
                            
                            # Projections
                            p1 = rx * math.cos(math.radians(30)) + ry * math.sin(math.radians(30))
                            p2 = rx * math.cos(math.radians(90)) + ry * math.sin(math.radians(90))
                            p3 = rx * math.cos(math.radians(150)) + ry * math.sin(math.radians(150))
                            
                            apothem_needed = max(abs(p1), abs(p2), abs(p3))
                            if apothem_needed > max_apothem_needed:
                                max_apothem_needed = apothem_needed
                                
                    side_needed = max_apothem_needed / (math.sqrt(3)/2)
                    if side_needed < min_side:
                        min_side = side_needed
                        best_params = (ocx, ocy, side_needed, oang)
                        
        return min_side, best_params[0], best_params[1], best_params[3]

    # Main search loop
    # We will try to build a packing greedily.
    # To avoid getting stuck in bad local optima, we will run multiple restarts with different random seeds for the grid point order.
    
    random.seed(42)
    
    for restart in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        # Shuffle grid points
        pts = grid_points[:]
        random.shuffle(pts)
        
        # Greedy placement
        placed = []
        # Start with center if available
        center_idx = None
        for i, p in enumerate(pts):
            if abs(p[0]) < step/2 and abs(p[1]) < step/2:
                center_idx = i
                break
                
        if center_idx is not None:
            placed.append(pts.pop(center_idx))
        else:
            # Pick a random point
            placed.append(pts.pop(0))
            
        while len(placed) < N:
            if time.time() > deadline:
                break
                
            added = False
            # Try to add the next point in the shuffled list
            # This is a simple greedy. It might fail to find a packing.
            # To improve, we could try to "fix" the placement by swapping.
            # But for now, let's just check if any remaining point fits.
            # This is O(N * M) per step. M is ~2000. N is 11.
            # 11 * 2000 * 11 (collision check) = 242k ops. Fast.
            
            for i in range(len(pts)):
                p = pts[i]
                # Check if p fits with all placed
                fits = True
                for q in placed:
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 3.0:
                        fits = False
                        break
                    if dist_sq <= 4.0:
                        # Exact check
                        c30 = math.sqrt(3)/2
                        s30 = 0.5
                        p1 = dx*c30 + dy*s30
                        p2 = dy
                        p3 = -dx*c30 + dy*s30
                        if abs(p1) <= math.sqrt(3) and abs(p2) <= math.sqrt(3) and abs(p3) <= math.sqrt(3):
                            fits = False
                            break
                if fits:
                    placed.append(p)
                    pts.pop(i)
                    added = True
                    break
                    
            if not added:
                break
                
        if len(placed) == N:
            # Found a valid packing
            side, ocx, ocy, oang = calculate_min_outer_side(placed)
            if side < best_side:
                best_side = side
                best_inner = placed
                best_outer = (ocx, ocy, oang)
                
    # Format result
    if best_inner is None:
        # Fallback to a simple linear arrangement if search failed
        # This will have a large side length, but is valid.
        best_inner = []
        for i in range(N):
            best_inner.append((i * math.sqrt(3), 0.0))
        # Calculate outer side for this
        side, ocx, ocy, oang = calculate_min_outer_side(best_inner)
        best_side = side
        best_outer = (ocx, ocy, oang)
    else:
        ocx, ocy, oang = best_outer
        
    inner_rows = []
    for p in best_inner:
        inner_rows.append([p[0], p[1], 0.0]) # Angle 0
        
    return inner_rows, (best_outer[0], best_outer[1]), best_side, best_outer[2]

if __name__ == "__main__":
    r = construct()
    print(r)
