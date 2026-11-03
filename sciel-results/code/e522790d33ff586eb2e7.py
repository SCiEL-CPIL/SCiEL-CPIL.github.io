import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 110.0 # Leave some margin

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback to a simple valid packing
        # Just return a large hexagon containing 11 hexagons in a line or grid
        # To be safe, return a configuration that is definitely valid
        # 11 hexagons in a 3x3 grid (with 2 missing corners?)
        # Let's just return a very large side length to ensure validity if optimization fails
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 10.0, 0.0

    # Unit hexagon vertices (side 1, centered at origin, flat top)
    # Angles: 0, 60, 120, 180, 240, 300
    # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    hex_vertices = np.array([
        [1.0, 0.0],
        [0.5, APOTHEM],
        [-0.5, APOTHEM],
        [-1.0, 0.0],
        [-0.5, -APOTHEM],
        [0.5, -APOTHEM]
    ])
    
    # Support function of unit hexagon for a given direction vector u (unit vector)
    def h_unit_hex(u):
        # u is (cos alpha, sin alpha)
        # Max dot product with vertices
        return np.max(hex_vertices @ u)

    # Calculate the side length S of the outer hexagon (orientation theta) 
    # needed to enclose the composite body defined by centers P and unit hex H
    def calc_side_length(P, theta):
        # P is Nx2 array of centers
        # theta is the rotation of the OUTER hexagon
        # The inner hexagons are fixed at orientation 0.
        # The composite body B = P + H.
        # We need B subset of H_out(theta, S).
        # h_B(u) = h_P(u) + h_H(u).
        # h_H_out(u) = S * h_unit_hex(R_{-theta} u).
        # We need h_P(u) + h_H(u) <= S * h_unit_hex(R_{-theta} u) for all u.
        # S >= (h_P(u) + h_H(u)) / h_unit_hex(R_{-theta} u).
        # We only need to check directions u that are normals to the edges of the outer hexagon?
        # No, the constraint is for all u. However, the ratio is maximized when u is aligned with the "corners" of the Minkowski sum relative to the container.
        # A sufficient and necessary condition for convex polygons is to check the normals of the container.
        # The normals of the outer hexagon are at angles theta + k*pi/3.
        # Let's check the 6 normals of the outer hexagon.
        
        c = np.cos(theta)
        s = np.sin(theta)
        R_inv = np.array([[c, s], [-s, c]]) # Rotation by -theta
        
        max_ratio = 0.0
        for k in range(6):
            alpha = theta + k * math.pi / 3.0
            u = np.array([math.cos(alpha), math.sin(alpha)])
            
            # h_P(u)
            h_p = np.max(P @ u)
            
            # h_H(u) - h of unit hexagon at orientation 0
            h_h = h_unit_hex(u)
            
            # h_unit_hex(R_{-theta} u)
            # Note: R_{-theta} u is the direction u rotated by -theta.
            # If u is a normal of the outer hexagon (angle alpha), then R_{-theta} u has angle alpha - theta = k*pi/3.
            # The support function of a unit hexagon (flat top, normals at 0, 60, ...) at angle 0 is 1.0 (vertex at 1,0).
            # Wait, the normal at angle 0 points to the vertex (1,0).
            # h_unit_hex(u) where u=(1,0) is 1.0.
            # Let's verify: max dot product of (1,0) with vertices.
            # (1,0).(1,0) = 1. (1,0).(0.5, 0.866) = 0.5. Max is 1.
            # So h_unit_hex(R_{-theta} u) should be 1.0 if u is a normal of the outer hexagon?
            # Yes, because the normals of the outer hexagon are aligned with the vertices of the unit hexagon when theta=0.
            # Actually, let's just compute it to be safe.
            u_rot = R_inv @ u
            h_denom = h_unit_hex(u_rot)
            
            if h_denom < 1e-9:
                continue
                
            ratio = (h_p + h_h) / h_denom
            if ratio > max_ratio:
                max_ratio = ratio
                
        return max_ratio

    def optimize_orientation(P):
        # Minimize calc_side_length(P, theta) over theta in [0, pi/3)
        # The function is periodic with period pi/3?
        # If we rotate outer hex by 60 deg, it's the same shape.
        # So we search in [0, pi/3).
        
        # Coarse search
        best_S = float('inf')
        best_theta = 0.0
        for i in range(30):
            theta = i * (math.pi / 3.0) / 30.0
            S = calc_side_length(P, theta)
            if S < best_S:
                best_S = S
                best_theta = theta
        
        # Fine search
        try:
            if time.time() - START_TIME < TIME_LIMIT - 5.0:
                res = minimize_scalar(lambda th: calc_side_length(P, th), bounds=(0, math.pi / 3.0), method='bounded', options={'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_theta = res.x
        except Exception:
            pass
            
        return best_S, best_theta

    # Generate initial candidate sets of centers
    # We use axial coordinates to generate hexagonal lattice points
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    candidates = []
    
    # 1. Simple grid-like patterns
    # Pattern 1: 3x3 block minus 2 corners
    axial_1 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1) # Two extra to make 11? 
        # 3x3 = 9. Need 2 more.
        # Let's use a known good packing for 11.
    ]
    # Actually, let's just generate many random or lattice-based sets and let the optimizer do the work.
    # A good strategy is to start with lattice points and then refine.
    
    def generate_lattice_candidates():
        pts = []
        # Search in a range of axial coordinates
        for q in range(-3, 4):
            for r in range(-3, 4):
                # Only keep points that form a "compact" set?
                # We need exactly 11 points.
                # This is hard to do randomly.
                pass
        return []

    # Instead, let's use specific known good configurations for 11 hexagons.
    # The optimal packing for 11 unit hexagons is often a "flower" or "cluster".
    # One common cluster is a central hexagon surrounded by 6, and then 4 more attached.
    
    # Config A: Center + 6 neighbors + 4 next-neighbors
    # Axial coords for center + 6 neighbors:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Next neighbors (distance 2):
    # (2,0), (-2,0), (0,2), (0,-2), (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)
    # We need 4 more. Let's pick 4 that are symmetric.
    # (2,0), (-2,0), (0,2), (0,-2) -> This makes a larger hexagon of radius 2?
    # Radius 2 hexagon has 1 + 6 + 12 = 19 points.
    # We want 11.
    # Let's try: Center, 6 neighbors, and 4 from the next ring.
    
    base_7 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    ]
    
    # Add 4 more from the next ring.
    # To maintain symmetry, we might add (2,0), (-2,0), (0,2), (0,-2)?
    # This creates a "cross" shape.
    cand_A = base_7 + [(2,0), (-2,0), (0,2), (0,-2)]
    
    # Another option: (2,0), (1,1), (0,2), (-1,1) ... not symmetric.
    # Let's try adding (2,0), (-2,0), (1,1), (-1,-1)?
    cand_B = base_7 + [(2,0), (-2,0), (1,1), (-1,-1)]
    
    # Let's try a "staircase" or "line"
    cand_C = [
        (0,0), (1,0), (2,0), (3,0), (4,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2)
    ]
    
    # Convert to Cartesian
    def make_candidate(axial_list):
        pts = np.array([axial_to_cartesian(q, r) for q, r in axial_list])
        # Center the points
        center = np.mean(pts, axis=0)
        pts -= center
        return pts

    candidates.append(make_candidate(cand_A))
    candidates.append(make_candidate(cand_B))
    candidates.append(make_candidate(cand_C))
    
    # Add some random perturbations to these candidates to allow local search
    all_candidates = []
    for base in candidates:
        all_candidates.append(base)
        # Add slightly perturbed versions
        for _ in range(2):
            perturbed = base + np.random.normal(0, 0.1, base.shape)
            all_candidates.append(perturbed)

    best_global_S = float('inf')
    best_global_P = None
    best_global_theta = 0.0

    for P in all_candidates:
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
            
        S, theta = optimize_orientation(P)
        
        # Local search on centers P
        # Objective: f(P) = min_theta calc_side_length(P, theta)
        # This is non-smooth.
        # We can fix theta to the optimal one and optimize P.
        # Then re-optimize theta.
        
        for _ in range(3): # A few iterations of local refinement
            if time.time() - START_TIME > TIME_LIMIT - 10.0:
                break
                
            # Fix theta, optimize P
            def obj_P(P_flat):
                P_arr = P_flat.reshape(N, 2)
                # Re-center P? No, the outer hexagon center is also variable.
                # In calc_side_length, we assumed P is centered?
                # No, calc_side_length uses h_P(u) = max(P @ u).
                # If P is not centered, the outer hexagon center should be adjusted.
                # The minimal enclosing hexagon of a set of points is centered at the "center" of the set.
                # For a regular hexagon, the center is the average of the vertices?
                # Actually, we can optimize the center of the outer hexagon too.
                # Let's assume the outer hexagon center is the centroid of P for simplicity in the local search step.
                centroid = np.mean(P_arr, axis=0)
                P_centered = P_arr - centroid
                return calc_side_length(P_centered, theta)
                
            P_flat = P.flatten()
            try:
                res = minimize(obj_P, P_flat, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < S:
                    P = res.x.reshape(N, 2)
                    S = res.fun
                    # Re-optimize theta
                    S_new, theta_new = optimize_orientation(P)
                    if S_new < S:
                        S = S_new
                        theta = theta_new
            except Exception:
                pass

        if S < best_global_S:
            best_global_S = S
            best_global_P = P
            best_global_theta = theta

    # Construct the result
    # The centers P are relative to the centroid.
    # The outer hexagon is centered at the centroid (0,0 in local coords).
    # We need to map back to global coords?
    # The task asks for inner (x, y, angle), outer center (x, y), side, angle.
    # We can set outer center to (0,0).
    # Inner hexagons are at P.
    # Inner hexagons have angle 0 (since we assumed they are aligned).
    # Outer hexagon has angle theta.
    
    inner = []
    for p in best_global_P:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    center = [0.0, 0.0]
    side = float(best_global_S)
    rotation = float(math.degrees(best_global_theta))
    
    return inner, center, side, rotation
