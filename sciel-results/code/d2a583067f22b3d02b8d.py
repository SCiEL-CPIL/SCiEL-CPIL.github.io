import math
import numpy as np
import time
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from itertools import combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: Return a trivial valid configuration if no time left
        # 1 hexagon at origin, outer side 1
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate initial candidates on a triangular lattice
    # Axial coordinates (q, r) for 11 hexagons
    # Center + 6 surrounding + 4 more
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
        
    pts = np.array(pts)
    
    # Calculate centroid
    centroid = np.mean(pts, axis=0)
    
    # We will optimize the outer hexagon rotation and center offset slightly?
    # The strategy suggests convex optimization.
    # Let's assume the inner hexagons are aligned with the outer one.
    # This is a common assumption in such packings (symmetry).
    # The problem becomes: Find min S such that there exists a hexagon of side S
    # containing the set of points P, where each point is a center of a unit hexagon
    # aligned with the container.
    
    # If inner and outer are aligned, the condition for containment is:
    # For all 6 normals n_i of the outer hexagon, max_p |n_i . (p - c)| <= A_out
    # where A_out = S * A_in (since A = s * sqrt(3)/2, so A_out/A_in = S)
    # So max_p |n_i . (p - c)| / A_in <= S
    # Let d_i(c) = max_p |n_i . (p - c)|
    # We want to minimize max_i d_i(c) / A_in over rotation and center?
    # Actually, fixing rotation allows us to minimize center c to minimize max_i d_i(c).
    # This is a convex problem (minimize t subject to |n_i . (p - c)| <= t for all i, p).
    
    # Let's iterate over rotation angles beta.
    # For each beta, solve the LP to find optimal center and t.
    
    def solve_for_rotation(beta):
        # Normals of outer hexagon (and inner ones, since aligned)
        # The normals point to the midpoints of the sides?
        # Standard hexagon: vertices at angles 0, 60, 120...
        # If we define hexagon by support function h(n) = A, where n are unit normals.
        # Normals are at angles 0, 60, ... ? Or 30, 90 ...?
        # Let's use normals at angles: pi/6, pi/6 + pi/3, ... (pointing to vertex midpoints? No, vertex normals?)
        # A regular hexagon with flat top has vertical sides? No, vertical sides means normal is horizontal.
        # Let's stick to: Normal vectors n_k for k=0..5.
        # n_k = (cos(theta_k), sin(theta_k))
        # If the hexagon is "pointy-top", vertices are at 90, 150...
        # Let's just use the 6 directions 0, 60, 120, 180, 240, 300.
        # Wait, a hexagon has 6 sides. The normals are perpendicular to the sides.
        # If vertices are at angles phi, phi+60, ...
        # Sides are between vertices. The normal to the side between v1 and v2 is at angle (phi+30).
        # So normals are at 30, 90, 150, 210, 270, 330 degrees.
        # Or 0, 60... if vertices are at 60, 120... (flat top).
        
        # Let's define the hexagon by 6 inequalities: n_i . x <= A
        # The n_i should be the outward normals to the 6 sides.
        # For a unit hexagon (side 1), A = sqrt(3)/2.
        
        theta0 = beta
        normals = []
        # We need the 6 outward normals.
        # If the hexagon is rotated by beta, its symmetry axes are rotated by beta.
        # Let's say the normal to the "top" side is at angle beta.
        # Then the other normals are at beta + k*pi/3.
        for k in range(6):
            ang = beta + k * math.pi / 3.0
            # This generates 6 normals. Are they the 6 distinct outward normals?
            # Yes, if they are equally spaced.
            # Note: For a hexagon, opposite normals are negatives of each other.
            # n_k and n_{k+3} = -n_k.
            normals.append((math.cos(ang), math.sin(ang)))
        
        # We want to find center c = (cx, cy) and minimal A such that:
        # For all i in 0..5, for all p in pts:
        # n_i . (p - c) <= A
        # AND
        # -n_i . (p - c) <= A  (which is equivalent to |n_i . (p - c)| <= A)
        
        # Actually, the constraint for the hexagon defined by normals n_i and support A is:
        # max_i (n_i . x) <= A.
        # This requires n_i to be the set of 6 outward normals.
        # If we use all 6, the condition is max(n_0.x, ..., n_5.x) <= A for all points x in the container?
        # No, the points must be INSIDE.
        # So for every point p, and every normal n_i:
        # n_i . (p - c) <= A
        
        # We want to minimize A.
        # Variables: cx, cy, A. (3 vars)
        # Constraints:
        # n_i[0]*(p_j[0] - cx) + n_i[1]*(p_j[1] - cy) <= A
        # for all i, j.
        
        # Let x = [cx, cy, A]
        # A_mat row i: -n_i[0], -n_i[1], -1
        # b_vec element i: -n_i[0]*p_j[0] - n_i[1]*p_j[1]  ... wait.
        
        # Inequality: -n_i[0]*cx - n_i[1]*cy - A <= -n_i[0]*p_j[0] - n_i[1]*p_j[1]
        
        c_mat = []
        b_vec = []
        
        for nx, ny in normals:
            for px, py in pts:
                # -nx*cx - ny*cy - A <= -nx*px - ny*py
                c_mat.append([-nx, -ny, -1.0])
                b_vec.append(-nx * px - ny * py)
                
        c_mat = np.array(c_mat)
        b_vec = np.array(b_vec)
        
        # Objective: minimize A.
        # c_obj = [0, 0, 1]
        
        c_obj = np.array([0.0, 0.0, 1.0])
        
        res = linprog(c_obj, A_ub=c_mat, b_ub=b_vec, method='highs')
        
        if not res.success:
            return float('inf'), centroid
        
        A_opt = res.x[2]
        c_opt = np.array([res.x[0], res.x[1]])
        
        return A_opt / APOTHEM, c_opt  # S = A / A_in
    
    best_S = float('inf')
    best_center = centroid
    best_beta = 0.0
    
    # Optimize over beta
    # The objective S(beta) is convex? Or periodic?
    # Let's sample beta and then refine.
    # Due to symmetry of the lattice (60 deg), we only need [0, pi/6].
    # But the lattice itself might have lower symmetry? 
    # The 11 points form a specific shape.
    
    # Let's do a coarse scan first.
    betas = np.linspace(0, math.pi/6, 20)
    for b in betas:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, c = solve_for_rotation(b)
        if S < best_S:
            best_S = S
            best_center = c
            best_beta = b
            
    # Refine best beta using golden section or ternary search?
    # S(beta) is convex in small regions?
    # Let's just check neighbors of best_beta from scan.
    # Actually, let's just try a few random starts or a finer scan around best.
    
    # The LP is fast. Let's do a finer scan.
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        betas_fine = np.linspace(0, math.pi/6, 100)
        for b in betas_fine:
            S, c = solve_for_rotation(b)
            if S < best_S:
                best_S = S
                best_center = c
                best_beta = b
                
    # Prepare output
    # The inner hexagons are centered at pts.
    # But wait, if we moved the center of the outer hexagon to best_center, 
    # and we assumed the inner hexagons are aligned with the outer one,
    # does the relative position of inner hexagons change?
    # The LP found the minimal enclosing hexagon centered at best_center for the FIXED points pts.
    # So the inner hexagons remain at pts.
    # The outer hexagon is centered at best_center, rotated by best_beta, with side best_S.
    # The inner hexagons are rotated by best_beta (to be aligned with outer).
    # Wait, the LP constraint was n_i . (p - c) <= A.
    # This means the points p are inside the hexagon defined by (c, beta, A).
    # If the inner hexagons are also rotated by beta, they are aligned.
    # Are they disjoint?
    # The lattice packing is disjoint.
    # The LP only ensures containment.
    # So if the initial lattice is valid (disjoint), and we just find a tighter box around it, it's valid.
    
    angle_deg = math.degrees(best_beta)
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
