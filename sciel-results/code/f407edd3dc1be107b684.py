import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _pairwise_clearance(p1, p2):
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.5: 
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    return penalty

def _min_enclosing_hexagon(inner_hexes):
    """
    Finds the minimum side length regular hexagon containing all inner_hexes.
    Uses LP for each candidate angle.
    """
    points = []
    for h in inner_hexes:
        verts = _hex_vertices(h[0], h[1], 1.0, h[2])
        points.extend(verts)
    
    points = np.array(points)
    if len(points) == 0:
        return 0, 0, 1.0, 0.0

    best_side = float('inf')
    best_params = (0, 0, 1.0, 0.0)
    
    # Scan angles. Due to symmetry, 0-60 is enough, but we scan 0-360 in coarse steps 
    # then refine. Actually, for a hexagon, rotating by 60 degrees gives the same shape 
    # but different vertex alignment. The "tightness" depends on how the inner cluster 
    # aligns with the outer hexagon's sides.
    
    # Coarse scan
    for angle_deg in np.arange(0, 60, 0.5):
        a = math.radians(angle_deg)
        normals = []
        for k in range(3):
            theta_n = a + math.pi/6 + k * math.pi/3
            normals.append((math.cos(theta_n), math.sin(theta_n)))
        
        A = []
        b = []
        
        for nx, ny in normals:
            for px, py in points:
                # n . (p - c) <= s * apothem_factor
                # n.x * px - n.x * cx + n.y * py - n.y * cy <= s * (sqrt(3)/2)
                # -n.x * cx - n.y * cy - s * (sqrt(3)/2) <= - (n.x * px + n.y * py)
                # Wait, standard form for linprog: A_ub x <= b_ub
                # We want to minimize s.
                # Constraints:
                # 1) n . (p - c) <= s * apothem  =>  -n . c - s * apothem <= -n . p
                #    A_row: [-nx, -ny, -apothem], b_val: -(nx*px + ny*py)
                # 2) -n . (p - c) <= s * apothem =>  n . c - s * apothem <= n . p
                #    A_row: [nx, ny, -apothem], b_val: (nx*px + ny*py)
                
                A.append([-nx, -ny, -math.sqrt(3)/2])
                b.append(-(nx*px + ny*py))
                
                A.append([nx, ny, -math.sqrt(3)/2])
                b.append(nx*px + ny*py)
        
        bounds = [(-100, 100), (-100, 100), (0, None)]
        
        try:
            res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_side:
                    best_side = s_val
                    best_params = (res.x[0], res.x[1], s_val, angle_deg)
        except Exception:
            pass
            
    # Refine around best angle
    if best_side < float('inf'):
        best_ang = best_params[3]
        # Search in a small window around the best coarse angle
        for angle_deg in np.arange(max(0, best_ang - 0.5), min(60, best_ang + 0.5), 0.05):
            a = math.radians(angle_deg)
            normals = []
            for k in range(3):
                theta_n = a + math.pi/6 + k * math.pi/3
                normals.append((math.cos(theta_n), math.sin(theta_n)))
                
            A = []
            b = []
            for nx, ny in normals:
                for px, py in points:
                    A.append([-nx, -ny, -math.sqrt(3)/2])
                    b.append(-(nx*px + ny*py))
                    A.append([nx, ny, -math.sqrt(3)/2])
                    b.append(nx*px + ny*py)
                    
            bounds = [(-100, 100), (-100, 100), (0, None)]
            try:
                res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if res.success:
                    s_val = res.x[2]
                    if s_val < best_side:
                        best_side = s_val
                        best_params = (res.x[0], res.x[1], s_val, angle_deg)
            except Exception:
                pass
                
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Strategy:
    # 1. Generate candidate positions on a hexagonal lattice.
    #    Hexagonal lattice vectors for unit hexagons (side 1):
    #    Distance between centers of touching hexes is 2 * apothem = sqrt(3).
    #    Let v1 = (sqrt(3), 0) and v2 = (sqrt(3)/2, 1.5).
    #    Any lattice point is i*v1 + j*v2.
    # 2. Use a greedy algorithm to select N=11 non-overlapping positions.
    #    Since all are on the lattice and aligned (angle 0), they don't overlap 
    #    if their lattice indices are distinct.
    #    However, we can also allow "off-lattice" shifts or rotations?
    #    For now, stick to lattice points with angle 0. This is a strong heuristic 
    #    because hexagons tile perfectly.
    # 3. The "search" is finding the compact set of 11 lattice points.
    #    This is equivalent to finding the minimum bounding hexagon for 11 points 
    #    on a triangular lattice.
    
    # Generate a grid of lattice points
    # Range: -10 to 10 in i, j coordinates
    lattice_points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * math.sqrt(3) + j * math.sqrt(3) / 2
            y = j * 1.5
            lattice_points.append((x, y, 0.0, i, j))
    
    # We want to find a set of 11 points that minimizes the enclosing hexagon size.
    # This is a combinatorial problem.
    # Heuristic: The most compact sets are those close to the center.
    # Let's score each lattice point by its "distance" from origin (in lattice metric or Euclidean).
    # Euclidean distance is a good proxy.
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Greedy selection:
    # Start with the closest point to origin.
    # Add the next closest point that doesn't overlap (all lattice points are disjoint 
    # if we just pick distinct indices, as they are centers of tiling hexes).
    # Wait, if we pick distinct lattice points, the hexagons are disjoint.
    # So the problem reduces to: Find the 11 lattice points with the smallest 
    # "bounding hexagon" size.
    
    # Since the number of points is small (11), and the lattice is regular, 
    # the optimal set is likely the "ball" of radius r in the lattice, plus some extras.
    # Let's just take the first 11 points from the sorted list (closest to origin).
    # This forms a compact cluster.
    
    selected = []
    used_indices = set()
    
    # The sorted list is by Euclidean distance from (0,0).
    # This is a good approximation for minimizing the bounding box.
    # Let's try a few different "centers" or shifts?
    # The lattice is fixed. The cluster shape is determined by which 11 points we pick.
    # Picking the 11 closest to origin is the standard "compact" choice.
    
    # Let's verify if picking the 11 closest is optimal.
    # Area argument: The area of the convex hull of the centers + padding.
    # For hexagons, the "bounding hexagon" size is related to the max projection 
    # onto the 3 normal directions.
    
    # Let's just take the top 11.
    for p in lattice_points:
        if len(selected) >= N:
            break
        # Check if this point is already "covered" or overlapping? 
        # No, distinct lattice points = disjoint hexagons.
        # So we just take the first N distinct points.
        selected.append(p[:3]) # (x, y, angle)
        
    # Now we have a candidate set of 11 hexagons.
    # Let's compute the min enclosing hexagon.
    
    inner = []
    for p in selected:
        inner.append([p[0], p[1], p[2]])
        
    ocx, ocy, os, oang = _min_enclosing_hexagon(inner)
    
    # Is this the best?
    # Maybe a slightly different set of 11 points yields a smaller bounding hexagon?
    # For example, if the 11th point is far in one direction, maybe swapping it 
    # for a point that is further in another direction but allows a more symmetric 
    # bounding box could help?
    # Actually, the bounding hexagon size is determined by the max extent in 3 directions.
    # A "balanced" set of points minimizes the max extent.
    # The 11 closest points to origin are:
    # 0: (0,0)
    # 1-6: neighbors at dist sqrt(3)
    # 7-12: next ring at dist 3 or sqrt(12)?
    # Let's check distances.
    # (1,0) -> sqrt(3) ~ 1.732
    # (0,1) -> 1.5
    # (1,1) -> sqrt(3+1.5^2+...) wait.
    # v1=(1.732, 0), v2=(0.866, 1.5).
    # (1,0): (1.732, 0), d=1.732
    # (0,1): (0.866, 1.5), d=sqrt(0.75+2.25)=sqrt(3)=1.732
    # (-1,0): d=1.732
    # (0,-1): d=1.732
    # (1,-1): v1-v2 = (0.866, -1.5), d=1.732
    # (-1,1): -v1+v2 = (-0.866, 1.5), d=1.732
    # So the 6 neighbors are all at dist 1.732.
    # Next ring:
    # (2,0): d=3.464
    # (0,2): d=3
    # (1,1): v1+v2 = (2.598, 1.5), d=sqrt(6.75+2.25)=sqrt(9)=3
    # (-1,1): already counted? No, (-1,1) is a neighbor.
    # Let's list points by distance:
    # 0: (0,0)
    # 1-6: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 7-12: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Distances for ring 2:
    # (2,0): 3.464
    # (1,1): 3.0
    # (0,2): 3.0
    # (-1,2): 3.0
    # (-2,1): 3.464
    # (-2,0): 3.464
    # (-1,-1): 3.0
    # (0,-2): 3.0
    # (1,-2): 3.0
    # (2,-1): 3.464
    
    # So the 11 closest points are:
    # 1 center + 6 neighbors + 4 from the ring of 10 at dist 3.0?
    # There are 6 points at dist 3.0: (1,1), (0,2), (-1,2), (-1,-1), (0,-2), (1,-2).
    # And 4 points at dist 3.464: (2,0), (-2,1), (-2,0), (2,-1).
    # Wait, (-2,1) is v*-2 + v*1 = (-3.464, 1.5), d=sqrt(12+2.25)=sqrt(14.25)=3.77?
    # Let's re-calc (-2,1). x = -2*1.732 + 1*0.866 = -3.464+0.866 = -2.598. y=1.5.
    # d^2 = 2.598^2 + 1.5^2 = 6.75 + 2.25 = 9. d=3.0.
    # So (-2,1) is also at dist 3.0.
    # Let's check all 12 points in the second ring (i,j with max(|i|,|j|,|i+j|)=2? No, hex distance).
    # Hex distance d(i,j) = (|i| + |j| + |i+j|) / 2.
    # Ring 1: d=1. 6 points.
    # Ring 2: d=2. 12 points.
    # Points with d=2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-3)? No.
    # The 12 points are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)? No.
    # Let's just trust the Euclidean sort in the code.
    # The code sorts by x^2+y^2.
    # It will pick the 11 closest.
    # This set is highly symmetric and compact.
    
    # To improve, we can try "shifting" the entire cluster?
    # No, the lattice is fixed. The cluster is defined by which lattice points are included.
    # We could try swapping one of the 4 chosen ring-2 points with a different one 
    # to see if it reduces the bounding hexagon.
    # But the bounding hexagon of a symmetric cluster is likely optimal.
    
    # Let's add a small local search: 
    # Try all 11-point subsets of the first 15 closest points?
    # C(15,11) is too large.
    # Instead, let's just stick to the 11 closest. It's a very strong baseline.
    
    # One potential improvement: Rotating the inner hexagons?
    # If we rotate them, they might pack tighter?
    # No, regular hexagons tile perfectly at angle 0. Rotating them creates gaps.
    # So angle 0 is optimal for lattice packing.
    
    # Final check: Is the outer hexagon orientation optimized?
    # Yes, _min_enclosing_hexagon scans angles.
    
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
