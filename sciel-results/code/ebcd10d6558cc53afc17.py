"""
Packing 11 unit regular hexagons inside a regular hexagon.
Strategy:
1. Generate candidate packings using known dense arrangements (hexagonal lattice clusters).
2. For each candidate, compute the minimum enclosing regular hexagon (MEH) of the set of hexagons.
   - The MEH is the smallest regular hexagon (of any orientation) that contains all the inner hexagons.
   - Since inner hexagons can rotate, but we fix them at 0 degrees for simplicity in the lattice,
     the container must contain the union of their convex hulls.
   - A regular hexagon containing a set of points is equivalent to the minimum enclosing hexagon.
   - However, the inner hexagons have area. We must ensure the container hexagon contains the *entire* inner hexagons, not just their centers.
   - This is equivalent to finding the minimum enclosing regular hexagon for the Minkowski sum of the set of centers with the unit hexagon shape.
   - Alternatively, we can model the constraint: for a container hexagon defined by center C, side length S, and rotation theta,
     every point of every inner hexagon must lie within the container.
   - This is a convex optimization problem if we fix the inner positions and rotations.
   - Here, we fix inner positions and rotations (0 degrees) and optimize the container's center, side length, and rotation.
   - The container hexagon H(C, S, theta) contains a point P iff for all 6 normals n_k(theta), dot(P - C, n_k) <= S * sqrt(3) / 2.
   - For an inner hexagon with center c_i and orientation 0, its vertices are c_i + v_j where v_j are the unit hexagon vertices.
   - The condition is: for all i, j, k: dot(c_i + v_j - C, n_k(theta)) <= S * sqrt(3) / 2.
   - This simplifies to: for all k, max_{i,j} dot(c_i + v_j - C, n_k(theta)) <= S * sqrt(3) / 2.
   - Let f_k(C, theta) = max_{i,j} dot(c_i + v_j - C, n_k(theta)).
   - We want to minimize S such that S >= 2/sqrt(3) * max_k f_k(C, theta).
   - For fixed theta, this is a convex problem in C and S. We can use scipy.optimize.minimize or solve a linear program.
   - Since the problem is small (11 hexagons, 6 vertices each = 66 points), we can brute-force theta or use optimization.

3. We will use a grid search over theta (0 to 60 degrees) and for each theta, solve for the optimal C and S using linear programming or simple optimization.
   - Actually, for fixed theta, the constraints are linear in C and S.
   - dot(c_i + v_j - C, n_k) <= S * sqrt(3)/2  =>  -dot(C, n_k) - S*sqrt(3)/2 <= -dot(c_i + v_j, n_k)
   - Let u = -C. Then dot(u, n_k) - S*sqrt(3)/2 >= dot(c_i + v_j, n_k) for all i,j.
   - Minimize S subject to these constraints. This is a linear program in variables u (2D) and S.
   - We can use scipy.optimize.linprog or scipy.optimize.minimize.

4. We will try several packings:
   - A 3x3 hexagonal cluster (7 hexagons) plus 4 more? No, 11 hexagons.
   - A common packing for 11 hexagons is a hexagon of side 3 (19 hexagons) minus 8? No.
   - Let's use a simple lattice arrangement: a hexagonal number pattern.
   - 11 hexagons can be arranged in a "flower" of 7 plus 4 on the sides?
   - Let's generate a few candidate center sets and evaluate them.

Candidate 1: A linear chain (bad).
Candidate 2: A hexagonal cluster of 7 centers, plus 4 centers extending one side.
Candidate 3: A 3x3 grid in hexagonal coordinates.

Let's implement the MEH solver and test a few packings.
"""

import numpy as np
import time
from scipy.optimize import linprog

# Unit hexagon vertices (pointy-top, side length 1)
# Vertices at angles 0, 60, ..., 300 degrees? Or flat-top?
# Standard regular hexagon with side 1 has circumradius 1.
# Let's use pointy-top: vertices at (0, 1), (sqrt(3)/2, 1/2), (sqrt(3)/2, -1/2), (0, -1), (-sqrt(3)/2, -1/2), (-sqrt(3)/2, 1/2)
# Wait, circumradius is 1. The distance from center to vertex is 1.
# Inradius (apothem) is sqrt(3)/2.
# Let's define the unit hexagon vertices.
hex_vertices = np.array([
    [0.0, 1.0],
    [np.sqrt(3)/2, 0.5],
    [np.sqrt(3)/2, -0.5],
    [0.0, -1.0],
    [-np.sqrt(3)/2, -0.5],
    [-np.sqrt(3)/2, 0.5]
])

def min_enclosing_hexagon_side(centers, theta):
    """
    Compute the minimum side length of a regular hexagon with rotation theta
    that contains all hexagons centered at `centers` (with orientation 0).
    """
    # Points to contain: all vertices of all inner hexagons
    # P = c_i + v_j
    # Constraint: dot(P - C, n_k) <= S * inradius
    # inradius = sqrt(3)/2
    # dot(P, n_k) - dot(C, n_k) <= S * sqrt(3)/2
    # Let u = C.
    # dot(C, n_k) + S * sqrt(3)/2 >= dot(P, n_k) for all P, k
    # We want to minimize S.
    # Variables: Cx, Cy, S
    # c = [0, 0, 1] (minimize S)
    
    # Normals of the container hexagon (outward pointing)
    # For a regular hexagon with rotation theta, the normals are at angles theta + 30, theta + 90, ... ?
    # Let's define the container hexagon by its 6 supporting lines.
    # The distance from center to each side is the inradius R = S * sqrt(3)/2.
    # The normals are orthogonal to the sides.
    # If the hexagon is rotated by theta, the normals are at angles theta + k*60 + 30?
    # Let's just use 6 normals spaced by 60 degrees, starting at theta.
    # Actually, the orientation of the hexagon is defined by the angle of its sides or vertices.
    # The problem says "rotation" of the outer hexagon.
    # Let's assume theta is the rotation of the outer hexagon.
    # The normals of the outer hexagon are at angles theta + 30, theta + 90, theta + 150, theta + 210, theta + 270, theta + 330.
    # Wait, a regular hexagon has 6-fold symmetry. The normals are separated by 60 degrees.
    # Let's define normals n_k(theta) = [cos(theta + k*pi/3), sin(theta + k*pi/3)] for k=0..5.
    # This corresponds to a hexagon with a vertex at angle theta? Or a side?
    # If n is a normal, the side is perpendicular to n.
    # Let's just iterate theta from 0 to 60 degrees and let the LP find the best C and S.
    
    normals = []
    for k in range(6):
        angle = theta + k * np.pi / 3
        normals.append([np.cos(angle), np.sin(angle)])
    normals = np.array(normals)
    
    # Points P
    all_points = []
    for c in centers:
        for v in hex_vertices:
            all_points.append(c + v)
    all_points = np.array(all_points)
    
    # Constraints:
    # For each k (0..5) and each point P:
    # dot(C, n_k) + S * sqrt(3)/2 >= dot(P, n_k)
    # -dot(C, n_k) - S * sqrt(3)/2 <= -dot(P, n_k)
    
    # Variables x = [Cx, Cy, S]
    A_ub = []
    b_ub = []
    
    inradius_factor = np.sqrt(3) / 2
    
    for k in range(6):
        n_k = normals[k]
        # dot(C, n_k) + S * inradius_factor >= dot(P, n_k)
        # -n_k[0]*Cx - n_k[1]*Cy - inradius_factor*S <= -dot(P, n_k)
        for P in all_points:
            row = [-n_k[0], -n_k[1], -inradius_factor]
            A_ub.append(row)
            b_ub.append(-np.dot(P, n_k))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    c = [0, 0, 1]
    
    # Bounds: S >= 0, Cx, Cy unbounded
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x[2]
    else:
        return float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidate packings
    # 1. Hexagonal cluster of 7 + 4
    # 2. 3x3 grid
    # 3. Other patterns
    
    candidates = []
    
    # Candidate 1: Hexagonal lattice
    # Centers in hexagonal coordinates (q, r)
    # Convert to Cartesian: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    # For unit hexagons, the distance between centers in a lattice is sqrt(3) (if sharing sides) or 2 (if touching vertices)?
    # If they are disjoint, they can touch. Distance between centers >= 2 * inradius = sqrt(3) if sharing a side?
    # Wait, two unit hexagons sharing a side have center distance sqrt(3).
    # Two unit hexagons touching at a vertex have center distance 2.
    # To be disjoint (interiors disjoint), center distance >= sqrt(3).
    
    # Let's use a hexagonal lattice with spacing sqrt(3).
    # This is the densest packing.
    
    # Pattern 1: 7 hexagons in a flower
    centers1 = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if abs(q + r) <= 1:
                x = 1.5 * np.sqrt(3) * q
                y = np.sqrt(3) * (r + q/2)
                centers1.append([x, y])
    # This is 7 centers. We need 11.
    # Add 4 more.
    # Extend the flower.
    
    # Let's just generate a 3x3 block in hexagonal coordinates.
    # q from -1 to 1, r from -1 to 1.
    # This is 9 centers.
    # Add 2 more.
    
    # Let's try a different approach: use a known packing.
    # For N=11, a good packing is a hexagon of side 2 (7 hexagons) plus 4 attached.
    
    # Let's define a few specific center sets.
    
    # Set 1: Linear
    c1 = [[i * np.sqrt(3), 0] for i in range(-5, 6)]
    candidates.append(c1)
    
    # Set 2: 3x3 Grid
    c2 = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            x = 1.5 * np.sqrt(3) * q
            y = np.sqrt(3) * (r + q/2)
            c2.append([x, y])
    # 9 centers. Add 2 more.
    c2.append([1.5 * np.sqrt(3) * 2, 0])
    c2.append([-1.5 * np.sqrt(3) * 2, 0])
    candidates.append(c2)
    
    # Set 3: Hexagonal number 11?
    # 1, 7, 19. 11 is not a hexagonal number.
    # 11 = 7 + 4.
    # 7 in a flower, 4 on the outside.
    c3 = []
    # Flower
    for q in range(-1, 2):
        for r in range(-1, 2):
            if abs(q + r) <= 1:
                x = 1.5 * np.sqrt(3) * q
                y = np.sqrt(3) * (r + q/2)
                c3.append([x, y])
    # Add 4
    c3.append([1.5 * np.sqrt(3) * 2, 0])
    c3.append([-1.5 * np.sqrt(3) * 2, 0])
    c3.append([0, np.sqrt(3) * 2])
    c3.append([0, -np.sqrt(3) * 2])
    candidates.append(c3)
    
    best_side = float('inf')
    best_config = None
    
    for centers in candidates:
        if time.time() > deadline:
            break
        # Optimize theta
        # The optimal theta is likely 0 or 30 degrees relative to the lattice.
        # Let's try theta = 0, 15, 30, 45, 60.
        for theta in [0, 15, 30, 45, 60]:
            if time.time() > deadline:
                break
            side = min_enclosing_hexagon_side(centers, np.radians(theta))
            if side < best_side:
                best_side = side
                best_config = (centers, theta)
        
        # Also try to optimize theta locally
        # We can do a simple grid search over theta
        for theta_deg in range(0, 60, 5):
            if time.time() > deadline:
                break
            theta = np.radians(theta_deg)
            side = min_enclosing_hexagon_side(centers, theta)
            if side < best_side:
                best_side = side
                best_config = (centers, theta_deg)
    
    if best_config is None:
        # Fallback
        centers = [[i * np.sqrt(3), 0] for i in range(-5, 6)]
        return [[c[0], c[1], 0.0] for c in centers], (0.0, 0.0), 40.0, 0.0
    
    centers, theta_deg = best_config
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # We need the center of the outer hexagon.
    # The LP gave us S, but we need C.
    # Let's re-run the LP for the best theta to get C.
    # Or, we can just center the cluster at (0,0) and adjust?
    # The LP minimizes S, but C is a free variable.
    # We should return the C that was optimal.
    
    # Let's modify min_enclosing_hexagon_side to return C as well.
    
    def get_best_c_and_s(centers, theta):
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        all_points = []
        for c in centers:
            for v in hex_vertices:
                all_points.append(c + v)
        all_points = np.array(all_points)
        
        A_ub = []
        b_ub = []
        inradius_factor = np.sqrt(3) / 2
        
        for k in range(6):
            n_k = normals[k]
            for P in all_points:
                row = [-n_k[0], -n_k[1], -inradius_factor]
                A_ub.append(row)
                b_ub.append(-np.dot(P, n_k))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        c = [0, 0, 1]
        bounds = [(None, None), (None, None), (0, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x[0], res.x[1], res.x[2]
        else:
            return 0.0, 0.0, float('inf')
    
    cx, cy, side = get_best_c_and_s(centers, np.radians(theta_deg))
    
    return inner, (cx, cy), side, theta_deg

