import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # The optimal configuration for 11 unit hexagons is the "flower" pattern:
    # 1 central hexagon + 6 surrounding + 4 in the "corners" of the ring.
    # In axial coordinates (q, r), these are:
    # Center: (0,0)
    # Ring: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Corners: (2,-1), (1,2), (-1,2), (-2,1) -- wait, let's verify the 3x3 block.
    # A 3x3 hexagonal block has 19 cells.
    # The "minimal" shape for 11 is often the hexagon of side 2 minus the 8 corners? No.
    # Side 2 hexagon has area 6. Area of 11 hexagons is 11.
    # Let's look at the specific candidate from the seed that likely works best.
    # The seed program had many candidates. One specific candidate is the "cross" or "flower".
    # Let's hardcode the best known configuration for N=11.
    # The configuration that minimizes the bounding hexagon side length S is the one where
    # the 11 hexagons form a shape that is as "hexagonal" as possible.
    
    # Axial coordinates for the 11 hexagons in the optimal "compact" shape:
    # This shape is a hexagon of side length 2 (which has 19 cells) with 8 corners removed?
    # No, 11 hexagons.
    # Let's use the candidate that is a "thickened" cross or similar.
    # Actually, the seed program's candidate 0 is:
    # [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    # Let's analyze this shape.
    # (0,0) center
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) are the 6 neighbors.
    # (1,1) is neighbor of (1,0) and (0,1).
    # (-1,2) is neighbor of (-1,1) and (0,1)? No, (-1,1)+(-1,1) = (-2,2). 
    # Axial neighbors of (0,1) are (1,1), (0,2), (-1,2), (-1,0), (0,0), (1,0).
    # So (-1,2) is a neighbor of (0,1).
    # (-2,1) is neighbor of (-1,1)? Neighbors of (-1,1): (0,1), (-1,2), (-2,2), (-2,1), (-2,0), (-1,0).
    # So (-2,1) is neighbor of (-1,1).
    # (2,-1) is neighbor of (1,-1)? Neighbors of (1,-1): (2,-1), (1,0), (0,0), (0,-1), (1,-2), (2,-2).
    # So (2,-1) is neighbor of (1,-1).
    
    # This shape is the "flower" (7 hexagons) plus 4 more attached to the "corners" of the ring.
    # This is a very compact shape.
    
    # Let's verify the side length of the enclosing hexagon for this shape.
    # If we place them at these axial positions, and assume the outer hexagon is aligned (rotation 0),
    # we can calculate the exact minimum S.
    
    # Axial to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Let's compute the bounding box in the "hexagonal" metric.
    # The distance from center to a vertex of a unit hexagon is 1.
    # The apothem is sqrt(3)/2.
    
    # For a regular hexagon of side S, the apothem is S * sqrt(3)/2.
    # The condition for a point (x,y) to be inside a hexagon of side S centered at (0,0) with rotation 0 is:
    # |x| <= S
    # |y| <= S * sqrt(3)
    # |x + y/sqrt(3)| <= S  (This is not quite right, let's use the standard form)
    
    # Standard form for hexagon with flat top (rot 0):
    # |x| <= S
    # |y| <= S * sqrt(3) / 2 * 2? No.
    # Vertices of unit hexagon (side 1) at rot 0:
    # (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # Apothem is sqrt(3)/2.
    # The inequality for a point (x,y) to be inside is:
    # |x| <= S
    # |sqrt(3)/2 * x + 1/2 * y| <= S * sqrt(3)/2
    # |sqrt(3)/2 * x - 1/2 * y| <= S * sqrt(3)/2
    
    # Let's check the vertices of our 11-hexagon shape.
    # We need to find the maximum projection of the vertices of the inner hexagons onto the normals of the outer hexagon.
    # The inner hexagons are at axial positions.
    # The "extremal" points of the inner hexagons are their vertices.
    # However, since the inner hexagons are aligned with the outer hexagon (rotation 0),
    # the extremal points in the direction of the outer hexagon's normals will be the vertices of the inner hexagons that point in that direction.
    
    # Let's just calculate the S required for the specific candidate.
    # Candidate: [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    
    # Let's convert to Cartesian.
    # (0,0) -> (0,0)
    # (1,0) -> (sqrt(3), 0)
    # (0,1) -> (sqrt(3)/2, 1.5)
    # (-1,1) -> (-sqrt(3)/2, 1.5)
    # (-1,0) -> (-sqrt(3), 0)
    # (0,-1) -> (-sqrt(3)/2, -1.5)
    # (1,-1) -> (sqrt(3)/2, -1.5)
    # (1,1) -> (sqrt(3)*1.5, 1.5) = (3*sqrt(3)/2, 1.5)
    # (-1,2) -> (sqrt(3)*(-1+1), 3) = (0, 3)
    # (-2,1) -> (sqrt(3)*(-2+0.5), 1.5) = (-1.5*sqrt(3), 1.5)
    # (2,-1) -> (sqrt(3)*(2-0.5), -1.5) = (1.5*sqrt(3), -1.5)
    
    # Let's check the projection on the x-axis (normal (1,0)).
    # Max x: 1.5*sqrt(3) from (1,1) and (2,-1).
    # Min x: -1.5*sqrt(3) from (-1,2) and (-2,1).
    # So the width in x is 3*sqrt(3).
    # The inner hexagons have side 1, so they extend 1 unit from their center in the x-direction?
    # No, the vertex of the inner hexagon at (1.5*sqrt(3), 1.5) is at x = 1.5*sqrt(3) + 1.
    # Wait, the inner hexagon centered at (1.5*sqrt(3), 1.5) has a vertex at x = 1.5*sqrt(3) + 1.
    # So the max x of the entire shape is 1.5*sqrt(3) + 1.
    # Similarly, min x is -1.5*sqrt(3) - 1.
    # So the condition |x| <= S * (something) needs to be met.
    # For the outer hexagon, the max x is S.
    # So S >= 1.5*sqrt(3) + 1.
    
    # Let's check the other directions.
    # Normal (sqrt(3)/2, 1/2).
    # Projection of (1.5*sqrt(3), 1.5): 1.5*sqrt(3)*sqrt(3)/2 + 1.5*1/2 = 2.25 + 0.75 = 3.
    # The inner hexagon vertex in this direction extends by 1.
    # So max projection is 3 + 1 = 4.
    # The outer hexagon's max projection in this direction is S * sqrt(3)/2.
    # So S * sqrt(3)/2 >= 4 => S >= 8/sqrt(3) = 8*sqrt(3)/3 = 2.666*sqrt(3) = 4.618.
    
    # Let's compare S from x-direction: 1.5*sqrt(3) + 1 = 2.598 + 1 = 3.598.
    # S from y-direction (normal (0,1)):
    # Max y of centers: 3 (from (-1,2)).
    # Inner hexagon extends 1 unit? No, the vertex is at y = 3 + sqrt(3)/2?
    # The vertex of the inner hexagon at (0,3) in the direction (0,1) is at y = 3 + 1? No.
    # The vertices of the inner hexagon at (0,3) are:
    # (1, 3), (0.5, 3+sqrt(3)/2), (-0.5, 3+sqrt(3)/2), (-1, 3), (-0.5, 3-sqrt(3)/2), (0.5, 3-sqrt(3)/2).
    # Max y is 3 + sqrt(3)/2.
    # So S * sqrt(3) >= 3 + sqrt(3)/2? No, the max y of the outer hexagon is S * sqrt(3).
    # So S * sqrt(3) >= 3 + sqrt(3)/2 => S >= 3/sqrt(3) + 0.5 = sqrt(3) + 0.5 = 1.732 + 0.5 = 2.232.
    
    # So the limiting factor is the direction (sqrt(3)/2, 1/2).
    # S = 8/sqrt(3) = 8*sqrt(3)/3.
    
    # Can we do better by rotating the outer hexagon?
    # The seed program tries to optimize the rotation.
    # Let's assume the optimal rotation is 0 for this shape, or close to it.
    # Actually, the shape is symmetric.
    
    # Let's just return this configuration. It is a known good packing.
    # We will refine the position slightly if needed, but this should be close to optimal.
    # The score is the side length S.
    
    # Let's calculate S precisely for this configuration with rotation 0.
    # The vertices of the inner hexagons are the points of interest.
    # We need to find the minimum S such that all vertices are inside the outer hexagon.
    
    # Let's use the LP approach from the seed program but just for this one candidate, to be safe and precise.
    # Or we can just return the hardcoded values if we are confident.
    # 8/sqrt(3) is approx 4.6188.
    
    # Let's check if there is a better shape.
    # The "3x3" block has 19 cells.
    # If we remove the 8 corners of the 3x3 block, we have 11 cells.
    # The 3x3 block is a hexagon of side 2? No, a hexagon of side 2 has 19 cells.
    # The corners of the 3x3 block are at axial coords:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-1,1), (0,1), (1,1), (2,1), (1,2), (0,2), (-1,2), (-2,1), (-2,2), (-1,2)...
    # The "corners" of the hexagon of side 2 are the 6 vertices.
    # The cells at the vertices are (2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2).
    # If we remove these 6, we have 13 cells.
    # If we remove 2 more, we have 11.
    
    # The shape I described earlier:
    # Center + 6 neighbors + 4 "corners" (1,1), (-1,2), (-2,1), (2,-1).
    # This shape has 11 cells.
    # Is it the same as the 3x3 block minus 8 corners?
    # The 3x3 block cells are:
    # r=2: (0,2), (1,2), (-1,2)
    # r=1: (0,1), (1,1), (-1,1), (2,1), (-2,1)
    # r=0: (0,0), (1,0), (-1,0), (2,0), (-2,0)
    # r=-1: (0,-1), (1,-1), (-1,-1), (2,-1), (-2,-1)
    # r=-2: (0,-2), (1,-2), (-1,-2)
    
    # My shape:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (1,1), (-1,2), (-2,1), (2,-1)
    
    # Let's check if these are in the 3x3 block.
    # (1,1) is in r=1.
    # (-1,2) is in r=2.
    # (-2,1) is in r=1.
    # (2,-1) is in r=-1.
    
    # This shape is not symmetric.
    # (1,1) is a "corner" of the ring.
    # (-1,2) is a "corner" of the ring.
    # (-2,1) is a "corner" of the ring.
    # (2,-1) is a "corner" of the ring.
    
    # Wait, the ring has 6 cells. The "corners" of the ring are the ones that are not adjacent to the center? No, all are adjacent.
    # The cells (1,1), (-1,2), (-2,1), (2,-1) are not adjacent to the center.
    # (1,1) is adjacent to (1,0) and (0,1).
    # (-1,2) is adjacent to (-1,1) and (0,1).
    # (-2,1) is adjacent to (-1,1) and (-1,0).
    # (2,-1) is adjacent to (1,-1) and (1,0).
    
    # So this shape is the "flower" plus 4 cells attached to the "edges" of the flower.
    # This shape is actually a hexagon of side 2 with the 8 "outer" corners removed?
    # The 3x3 block has 19 cells.
    # The 6 "corner" cells of the 3x3 block are:
    # (2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2).
    # If we remove these 6, we have 13 cells.
    # The remaining 13 cells are:
    # r=2: (0,2)
    # r=1: (0,1), (1,1), (-1,1), (2,1), (-2,1)
    # r=0: (0,0), (1,0), (-1,0)
    # r=-1: (0,-1), (1,-1), (-1,-1), (2,-1), (-2,-1)
    # r=-2: (0,-2)
    
    # My shape has:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (1,1), (-1,2), (-2,1), (2,-1)
    
    # (-1,2) is in the 3x3 block.
    # (-2,1) is in the 3x3 block.
    # (2,-1) is in the 3x3 block.
    # (1,1) is in the 3x3 block.
    
    # But my shape does NOT have (0,2), (2,1), (2,0), (1,-2), (0,-2), (-2,-1), (-1,-1).
    # It has (-1,2) which is a corner of the 3x3 block.
    # It has (-2,1) which is an edge of the 3x3 block.
    # It has (2,-1) which is an edge of the 3x3 block.
    # It has (1,1) which is an edge of the 3x3 block.
    
    # This shape is not the 3x3 block minus corners.
    # It is a different shape.
    
    # Let's stick with the seed program's optimization for this specific candidate.
    # It is robust and fast.
    
    # I will modify the seed program to ONLY evaluate this one candidate, and optimize it thoroughly.
    # This saves time and focuses on the best shape.
    
    axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
    
    pts = axial_to_cartesian(axial)
    
    # Optimize the outer hexagon for these fixed points.
    # We can use the LP method from the seed program.
    
    def calc_S_for_beta(pts, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D
        else:
            return float('inf'), None, None

    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # Grid search over beta
    n_grid = 100
    for i in range(n_grid):
        beta = i * (math.pi / 6) / (n_grid - 1)
        S, center, D = calc_S_for_beta(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center
            
    # Refine
    if best_center is not None:
        window = (math.pi / 6) / (n_grid - 1)
        lo = max(0.0, best_beta - 2 * window)
        hi = min(math.pi / 6, best_beta + 2 * window)
        
        def obj_1d(b):
            S, _, _ = calc_S_for_beta(pts, b)
            return S
            
        res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded', options={'maxiter': 200, 'xatol': 1e-12})
        if res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
            _, best_center, _ = calc_S_for_beta(pts, best_beta)
            
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hexagon_vertices(cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        verts = []
        for k in range(6):
            angle = rot + k * math.pi / 3.0
            x = cx + side * math.cos(angle)
            y = cy + side * math.sin(angle)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        dx = px - cx
        dy = py - cy
        dx_local = dx * math.cos(rot) + dy * math.sin(rot)
        dy_local = -dx * math.sin(rot) + dy * math.cos(rot)
        apothem = side * APOTHEM
        for k in range(6):
            angle = k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            dot = dx_local * nx + dy_local * ny
            if dot > apothem + 1e-10:
                return False
        return True

    def hexagons_disjoint(c1, c2, s1, s2, rot1_deg, rot2_deg):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return dist >= (s1 + s2) - 1e-9

    # Generate candidate configurations (axial coordinates for aligned hexagons)
    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(0,1),(0,-1),(1,1)])
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(1,1),(-1,1),(0,1)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(-1,1),(0,1),(1,1),(2,1),(3,1),(4,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3)])
    candidates.append([(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(2,-2)])

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation (vectorized)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        # Vectorized construction: for each face k, build rows for all points at once
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            # Negative side: -nx*x - ny*y - D <= -proj  =>  nx*x + ny*y + D >= proj
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            # Positive side: nx*x + ny*y - D <= proj
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def optimize_orientation_only(pts):
        """1D scalar minimization over outer hexagon orientation beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        n_grid = 97
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1)
                lo = max(0.0, best_beta - 2 * window)
                hi = min(math.pi / 6, best_beta + 2 * window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1) * 0.5
                lo = max(0.0, best_beta - window)
                hi = min(math.pi / 6, best_beta + window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def optimize_shape_and_inner(pts_init):
        """Optimize both inner positions and outer hexagon parameters."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None

        n_pts = len(pts_init)
        # Initial best with fixed pts
        S_best, beta_best, center_best = optimize_orientation_only(pts_init)
        if center_best is None:
            return float('inf'), 0.0, None, None

        best_S = S_best
        best_beta = beta_best
        best_center = center_best
        best_pts = pts_init.copy()

        # Smooth optimization of all variables
        # Variables: x1, y1, ..., xn, yn, cx, cy, beta
        # Total: 2*n_pts + 3
        
        def smooth_obj_all(params, temperature=0.05):
            pts = params[:2*n_pts].reshape(-1, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            return smooth_calc_S(pts, cx, cy, beta, temperature=temperature)

        initial_params = np.concatenate([pts_init.ravel(), [best_center[0], best_center[1], best_beta]])
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Try a few temperatures
                for temp in [0.1, 0.05, 0.02]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(smooth_obj_all, initial_params, args=(temp,), method='L-BFGS-B', 
                                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                    if res.fun < 1e10:
                        res_pts = res.x[:2*n_pts].reshape(-1, 2)
                        res_cx = res.x[2*n_pts]
                        res_cy = res.x[2*n_pts+1]
                        res_beta = ((res.x[2*n_pts+2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        
                        # Check validity of new pts (disjointness)
                        valid = True
                        min_dist = math.sqrt(3)
                        for i in range(n_pts):
                            for j in range(i+1, n_pts):
                                dx = res_pts[i,0] - res_pts[j,0]
                                dy = res_pts[i,1] - res_pts[j,1]
                                dist = math.sqrt(dx*dx + dy*dy)
                                if dist < min_dist - 1e-6:
                                    valid = False
                                    break
                            if not valid: break
                            
                        if valid:
                            S_exact = exact_S(res_pts, res_cx, res_cy, res_beta)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = res_beta
                                best_center = (res_cx, res_cy)
                                best_pts = res_pts
                                initial_params = res.x
        except Exception:
            pass

        # Polish with exact objective on the best found configuration
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def exact_obj_all(params):
                pts = params[:2*n_pts].reshape(-1, 2)
                cx = params[2*n_pts]
                cy = params[2*n_pts+1]
                beta = params[2*n_pts+2]
                beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                # Check disjointness penalty
                pen = 0
                min_dist = math.sqrt(3)
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        dx = pts[i,0] - pts[j,0]
                        dy = pts[i,1] - pts[j,1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < min_dist:
                            pen += (min_dist - dist)**2 * 10000
                
                return exact_S(pts, cx, cy, beta) + pen

            res = minimize(exact_obj_all, initial_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < 1e10:
                res_pts = res.x[:2*n_pts].reshape(-1, 2)
                res_cx = res.x[2*n_pts]
                res_cy = res.x[2*n_pts+1]
                res_beta = ((res.x[2*n_pts+2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                # Final validity check
                valid = True
                min_dist = math.sqrt(3)
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        dx = res_pts[i,0] - res_pts[j,0]
                        dy = res_pts[i,1] - res_pts[j,1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < min_dist - 1e-5:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    S_exact = exact_S(res_pts, res_cx, res_cy, res_beta)
                    if S_exact < best_S:
                        best_S = S_exact
                        best_beta = res_beta
                        best_center = (res_cx, res_cy)
                        best_pts = res_pts

        return best_S, best_beta, best_center, best_pts

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by a quick heuristic estimate (bounding box diagonal) to prioritize promising ones
    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        # Quick bounding hexagon estimate with beta=0
        angles = math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        # Center at centroid
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        pts_c = pts - np.array([cx, cy])
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    # Score and sort candidates
    scored_candidates = []
    for axial in candidates:
        est = quick_estimate(axial)
        scored_candidates.append((est, axial))
    scored_candidates.sort(key=lambda x: x[0])

    for idx, (est, axial) in enumerate(scored_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center, opt_pts = optimize_shape_and_inner(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = opt_pts if opt_pts is not None else pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
        # Fallback optimization for this single candidate
        S, beta, center, opt_pts = optimize_shape_and_inner(best_global_pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = opt_pts if opt_pts is not None else best_global_pts
            best_global_beta = beta
            best_global_center = center

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    SCALE = 1.0 + 1e-14
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
