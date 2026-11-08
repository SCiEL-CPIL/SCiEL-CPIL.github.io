"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    # Hexagonal grid coordinates to Cartesian
    # For a regular hexagon with side s, vertices at angles 0,60,120,180,240,300
    # Unit hexagon (side=1): circumradius = 1, inradius = sqrt(3)/2
    # Adjacent hex centers are 2*inradius = sqrt(3) apart
    
    # Place 11 unit hexagons in a compact hexagonal arrangement
    # Use axial coordinates: (q, r) -> cartesian
    # For pointy-top hexagons: x = sqrt(3)*(q + r/2), y = 3*r/2
    # For flat-top hexagons: x = 3*q/2, y = sqrt(3)*(r + q/2)
    
    # Let's use a simple compact arrangement:
    # Central hex + 6 around + 4 more
    # Positions in hex grid (axial coords) for 11 hexagons:
    # A 2x2x2 hex block minus corners, or similar
    
    # Compact arrangement: rows of 3, 4, 4 (11 total)
    # In hex grid, this is like a parallelogram
    
    # Let me use explicit positions for a tight cluster
    # Hex grid positions (col, row) for a compact 11-hex cluster
    positions_axial = [
        (0, 0),    # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # 6 neighbors = 7 total
        (2, -1), (2, 0), (-2, 1), (-2, 0)  # 4 more = 11 total
    ]
    
    # Convert axial to cartesian for flat-top hexagons
    # x = 3/2 * q, y = sqrt(3) * (r + q/2)
    centers = []
    for q, r in positions_axial:
        x = 1.5 * q
        y = math.sqrt(3) * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # All hexagons have angle 0 (flat-top)
    angles = [0.0] * N
    
    # Find minimum outer hexagon side length that contains all inner hexagons
    # The outer hexagon must contain each inner hexagon entirely.
    # For a hexagon centered at c with side S and rotation theta,
    # a point p is inside if for each of 6 edges, the distance from c to p
    # projected onto the edge normal is <= S*sqrt(3)/2
    
    # Actually, for a regular hexagon with side S, the apothem (center to edge) is S*sqrt(3)/2
    # A point is inside if its distance from center along each of 6 directions (edge normals)
    # is <= apothem
    
    # The edge normals for a flat-top hexagon (angle 0) are at angles 30, 90, 150, 210, 270, 330 degrees
    # For a rotated hexagon with angle theta, normals are at 30+theta, 90+theta, etc.
    
    # To minimize S, we need to find the smallest S such that there exists a center (cx, cy)
    # and rotation theta where all inner hexagons fit.
    
    # Each inner hexagon has its own extent. The inner hexagon at position ci with rotation ai
    # occupies a region. For the inner hexagon to fit inside the outer hexagon,
    # the outer hexagon must contain all vertices of the inner hexagon.
    
    # Inner hexagon vertices: for hex at center (cx_i, cy_i) with angle a_i,
    # vertices are at (cx_i + cos(a_i + k*60°), cy_i + sin(a_i + k*60°)) for k=0..5
    
    # For the outer hexagon with center (ox, oy), side S, rotation theta:
    # A point p is inside if for all 6 edge normals n_j (unit vectors at angles theta + 30° + j*60°),
    # dot(p - outer_center, n_j) <= S * sqrt(3)/2
    
    # So we need: for all inner vertices p, for all j:
    # dot(p - outer_center, n_j) <= S * sqrt(3)/2
    
    # This means: max over all p and j of dot(p - outer_center, n_j) <= S * sqrt(3)/2
    
    # So S >= (2/sqrt(3)) * max_{p,j} dot(p - outer_center, n_j)
    
    # We want to minimize over outer_center and theta: max_{p,j} dot(p - outer_center, n_j(theta))
    
    # This is a convex optimization problem in outer_center for fixed theta.
    # The optimal outer_center for fixed theta is the center of the minimum enclosing circle
    # in the "hexagonal norm" defined by the 6 directions.
    
    # For simplicity, let's try theta=0 and find optimal center, then try a few theta values.
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(N):
        cx, cy = centers[i]
        a = math.radians(angles[i])
        for k in range(6):
            angle = a + k * math.pi / 3
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    best_S = float('inf')
    best_center = (0, 0)
    best_theta = 0
    
    # Try several rotations for the outer hexagon
    for theta_deg in range(0, 60, 5):  # 0 to 55 degrees in steps of 5
        if time.time() > deadline:
            break
        theta = math.radians(theta_deg)
        
        # Edge normals for outer hexagon
        normals = []
        for j in range(6):
            angle = theta + math.pi / 6 + j * math.pi / 3  # 30° + theta + j*60°
            normals.append((math.cos(angle), math.sin(angle)))
        
        # For fixed theta, find optimal outer center to minimize max projection
        # The optimal center minimizes max_{p, j} dot(p - c, n_j)
        # This is equivalent to finding the center of the smallest hexagon (in this norm)
        # containing all points.
        
        # For a given set of directions, the optimal center is where the max projection
        # is balanced. We can use a simple approach: the center should be the centroid
        # of the vertices, or we can do a few iterations.
        
        # Actually, for the hexagonal norm, the minimum enclosing hexagon center
        # can be found by considering that the constraint is:
        # For each normal n_j: max_p dot(p, n_j) - dot(c, n_j) <= S*sqrt(3)/2
        # And also: -max_p dot(p, n_j) + dot(c, n_j) <= S*sqrt(3)/2 (for the opposite edge)
        # Wait, no. The hexagon has 6 edges, each with a normal. The condition is:
        # dot(p - c, n_j) <= apothem for j=0..5
        # This means: dot(c, n_j) >= dot(p, n_j) - apothem for all p, j
        # And we want to minimize apothem = S*sqrt(3)/2
        
        # So apothem >= max_j (max_p dot(p, n_j) - dot(c, n_j))
        # We want to minimize over c: max_j (max_p dot(p, n_j) - dot(c, n_j))
        
        # Let M_j = max_p dot(p, n_j)
        # Minimize over c: max_j (M_j - dot(c, n_j))
        
        # This is a linear programming problem! Minimize t subject to:
        # t >= M_j - dot(c, n_j) for all j
        # i.e., dot(c, n_j) + t >= M_j for all j
        
        # Let's solve this with a simple approach: the optimal c is at the intersection
        # of some of the constraints. For 2D, we can try all pairs of constraints.
        
        # M_j for each normal
        M = np.zeros(6)
        for j in range(6):
            nx, ny = normals[j]
            M[j] = np.max(all_vertices[:, 0] * nx + all_vertices[:, 1] * ny)
        
        # Solve: minimize t s.t. dot(c, n_j) + t >= M_j for all j
        # This is: minimize t s.t. -dot(c, n_j) - t <= -M_j for all j
        # Variables: c_x, c_y, t
        # Constraints: -nx_j*c_x - ny_j*c_y - t <= -M_j for j=0..5
        
        # For 2D, the optimal solution has at least 2 active constraints (plus t constraint).
        # Let's try all pairs of normals and solve.
        
        # Alternative: use scipy if available, otherwise brute force over candidate centers
        
        # Candidate centers: centroid, and intersections of constraint boundaries
        # For now, use centroid as a good initial guess
        
        cx0 = np.mean(all_vertices[:, 0])
        cy0 = np.mean(all_vertices[:, 1])
        
        # Compute S for this center
        apothem_max = 0
        for j in range(6):
            nx, ny = normals[j]
            proj = np.max(all_vertices[:, 0] * nx + all_vertices[:, 1] * ny)
            proj_center = cx0 * nx + cy0 * ny
            apothem_max = max(apothem_max, proj - proj_center)
        
        S_candidate = 2 * apothem_max / math.sqrt(3)
        
        if S_candidate < best_S:
            best_S = S_candidate
            best_center = (cx0, cy0)
            best_theta = theta_deg
    
    # Try also with the center at (0,0) and other candidates
    for theta_deg in range(0, 60, 5):
        if time.time() > deadline:
            break
        theta = math.radians(theta_deg)
        normals = []
        for j in range(6):
            angle = theta + math.pi / 6 + j * math.pi / 3
            normals.append((math.cos(angle), math.sin(angle)))
        
        M = np.zeros(6)
        for j in range(6):
            nx, ny = normals[j]
            M[j] = np.max(all_vertices[:, 0] * nx + all_vertices[:, 1] * ny)
        
        # Try center at origin
        cx0, cy0 = 0.0, 0.0
        apothem_max = 0
        for j in range(6):
            nx, ny = normals[j]
            proj = np.max(all_vertices[:, 0] * nx + all_vertices[:, 1] * ny)
            proj_center = cx0 * nx + cy0 * ny
            apothem_max = max(apothem_max, proj - proj_center)
        
        S_candidate = 2 * apothem_max / math.sqrt(3)
        if S_candidate < best_S:
            best_S = S_candidate
            best_center = (cx0, cy0)
            best_theta = theta_deg
    
    # Format inner hexagons
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    
    return inner, best_center, best_S, best_theta
# EVOLVE-BLOCK-END
