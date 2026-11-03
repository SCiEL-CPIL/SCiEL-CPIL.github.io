<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def get_gap(v1, v2):
        """
        Returns the maximum gap between two hexagons using SAT.
        If gap > 0, they are separated.
        If gap < 0, they overlap.
        """
        max_gap = -float('inf')
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
            
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
            
            for n in [n1, n2]:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    overlap_amount = min(max1, max2) - max(min1, min2)
                    gap = -overlap_amount
                    
                if gap > max_gap:
                    max_gap = gap
                    
        return max_gap

    def compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg):
        """
        Computes the minimum side length of the outer hexagon centered at (outer_cx, outer_cy)
        with rotation outer_angle_deg to contain all inner hexagons.
        """
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = inner_centers[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        # Unpack parameters
        # 0..35: inner centers (x, y) and angles for 12 hexagons (3 params each)
        # 36: outer cx
        # 37: outer cy
        # 38: outer angle (degrees)
        
        inner_centers = []
        inner_angles = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i + 1]
            ang = params[3*i + 2]
            inner_centers.append((cx, cy))
            inner_angles.append(ang)
        
        outer_cx = params[36]
        outer_cy = params[37]
        outer_angle_deg = params[38]
        
        # Calculate side length
        side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Calculate overlap penalty with smooth surrogate
        penalty = 0.0
        verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    # Use smooth penalty: -gap + epsilon*exp(-gap) to provide gradient
                    # This is better than squared penalty for near-zero overlaps
                    penalty += (-gap) + 0.1 * math.exp(-gap)
        
        # Weight the penalty heavily
        return side + 500.0 * penalty

    # Initial configuration: Staggered grid of 12 hexagons
    s = 1.0
    dx = 2.0 * s
    dy = math.sqrt(3) * s
    offset = s
    
    # Create a 4x3 grid (4 rows, 3 cols)
    centers = []
    for r in range(4):
        for c in range(3):
            x = c * dx
            y = r * dy
            if r % 2 == 1:
                x += offset
            centers.append((x, y))
            
    # Center the grid at (0,0)
    cx_mean = sum(c[0] for c in centers) / N
    cy_mean = sum(c[1] for c in centers) / N
    adjusted_centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # Initialize parameters
    x0 = np.zeros(39)
    for i in range(N):
        x0[3*i] = adjusted_centers[i][0]
        x0[3*i+1] = adjusted_centers[i][1]
        x0[3*i+2] = 0.0 # Initial angle 0
        
    x0[36] = 0.0
    x0[37] = 0.0
    x0[38] = 0.0
    
    best_side = float('inf')
    best_params = x0.copy()
    best_valid = False

    # Run optimization with multiple restarts
    restarts = [
        # Original grid
        (adjusted_centers, [0.0]*N, 0.0, 0.0, 0.0),
        # Slightly perturbed grid
    ]
    
    # Add perturbed initial configurations
    rng = np.random.RandomState(42)
    for _ in range(3):
        perturbed_centers = [(x + rng.normal(0, 0.1), y + rng.normal(0, 0.1)) for x, y in adjusted_centers]
        perturbed_angles = [rng.normal(0, 5) for _ in range(N)]
        restarts.append((perturbed_centers, perturbed_angles, rng.normal(0, 0.5), rng.normal(0, 0.5), rng.uniform(0, 60)))
    
    for restart_idx, (init_centers, init_angles, init_ox, init_oy, init_oor) in enumerate(restarts):
        if time.time() > deadline - 30.0:
            break
            
        x0 = np.zeros(39)
        for i in range(N):
            x0[3*i] = init_centers[i][0]
            x0[3*i+1] = init_centers[i][1]
            x0[3*i+2] = init_angles[i]
        x0[36] = init_ox
        x0[37] = init_oy
        x0[38] = init_oor
        
        # Run optimization
        try:
            # Nelder-Mead is robust for this non-smooth problem
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            
            # Evaluate result
            inner_centers = []
            inner_angles = []
            for i in range(N):
                cx = res.x[3*i]
                cy = res.x[3*i + 1]
                inner_centers.append((cx, cy))
                inner_angles.append(res.x[3*i+2])
                
            outer_cx = res.x[36]
            outer_cy = res.x[37]
            outer_angle_deg = res.x[38]
            
            side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
            
            # Check validity
            verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    gap = get_gap(verts_list[i], verts_list[j])
                    if gap < -1e-5: # Allow tiny numerical error
                        valid = False
                        break
                if not valid: break
                
            if valid and side < best_side:
                best_side = side
                best_params = res.x
                best_valid = True
        except Exception as e:
            pass

    # If optimization didn't yield a valid result, try the initial grid as fallback
    if not best_valid:
        inner_centers = adjusted_centers
        inner_angles = [0.0] * N
        outer_cx = 0.0
        outer_cy = 0.0
        outer_angle_deg = 0.0
        
        side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Check validity of this fallback
        verts_list = [hex_vertices(c[0], c[1], 0.0) for c in inner_centers]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            best_side = side
            inner_list = []
            for c, a in zip(inner_centers, inner_angles):
                inner_list.append([float(c[0]), float(c[1]), float(a)])
            return inner_list, (float(outer_cx), float(outer_cy)), float(best_side * 1.01), float(outer_angle_deg)
        else:
            # Absolute fallback: huge box
            inner_list = [[0.0, 0.0, 0.0] for _ in range(N)]
            return inner_list, (0.0, 0.0), 100.0, 0.0

    # High-precision polish pass on best solution
    if best_valid and time.time() < deadline - 10.0:
        try:
            res_polish = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'maxfev': 20000,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            inner_centers = []
            inner_angles = []
            for i in range(N):
                cx = res_polish.x[3*i]
                cy = res_polish.x[3*i + 1]
                inner_centers.append((cx, cy))
                inner_angles.append(res_polish.x[3*i+2])
                
            outer_cx = res_polish.x[36]
            outer_cy = res_polish.x[37]
            outer_angle_deg = res_polish.x[38]
            
            side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
            
            # Check validity
            verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    gap = get_gap(verts_list[i], verts_list[j])
                    if gap < -1e-6:
                        valid = False
                        break
                if not valid: break
            
            if valid and side < best_side:
                best_side = side
                best_params = res_polish.x
        except Exception as e:
            pass

    # Construct return value from best_params
    inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        ang_deg = best_params[3*i+2]
        inner.append([float(cx), float(cy), float(ang_deg)])
        
    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_angle_deg = best_params[38]
    
    # Add safety margin
    final_side = best_side * 1.005 + 1e-5
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle_deg)
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def get_gap(v1, v2):
        """
        Returns the maximum gap between two hexagons using SAT.
        If gap > 0, they are separated.
        If gap < 0, they overlap.
        """
        max_gap = -float('inf')
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
            
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
            
            for n in [n1, n2]:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    overlap_amount = min(max1, max2) - max(min1, min2)
                    gap = -overlap_amount
                    
                if gap > max_gap:
                    max_gap = gap
                    
        return max_gap

    def compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg):
        """
        Computes the minimum side length of the outer hexagon centered at (outer_cx, outer_cy)
        with rotation outer_angle_deg to contain all inner hexagons.
        """
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = inner_centers[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    def check_validity(centers, angles):
        """Check if configuration is valid (no overlaps)."""
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    return False
        return True

    # Structural Change: Analytic Hexagonal Lattice Packing
    # Instead of generic numerical optimization on 39 variables, we exploit the symmetry
    # of regular hexagons. We place them in a hexagonal lattice pattern.
    
    # Generate candidate configurations based on hexagonal lattice parameters
    # 1. Standard Hexagonal Tiling (Rotated to fit best)
    # 2. Compact Clusters
    
    best_side = float('inf')
    best_config = None # (centers, angles, outer_cx, outer_cy, outer_angle)
    
    # Helper to evaluate a configuration
    def evaluate_config(centers, angles, outer_cx, outer_cy, outer_angle):
        if not check_validity(centers, angles):
            return None, False
        side = compute_outer_side(centers, angles, outer_cx, outer_cy, outer_angle)
        return side, True

    # Configuration 1: 1-2-3-2-1 Hexagonal Cluster
    # This is a common dense packing for 12 items.
    # Rows: 1, 2, 3, 2, 1
    # Total: 9? No, 1+2+3+2+1 = 9.
    # We need 12.
    # Let's try 2-3-4-3-2? No.
    # 12 = 1 + 2 + 3 + 2 + 4? No.
    # 12 = 2 + 3 + 4 + 3? No.
    
    # Standard hexagonal numbers: 1, 7, 19.
    # 12 is not a hex number.
    # Let's try a 3x4 rectangle of hexagons in a honeycomb grid.
    # Or a distorted hexagonal cluster.
    
    # Let's generate a specific lattice:
    # Hexagonal lattice vectors:
    # u = (2, 0)
    # v = (1, sqrt(3))
    # w = (-1, sqrt(3))
    
    # We will try several geometric arrangements and optimize the outer hexagon orientation.
    
    arrangements = []
    
    # Arrangement A: 3x4 Grid in Honeycomb
    # This is a parallelogram of hexagons.
    # Let's define centers relative to origin.
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (shifted)
    # Row 2: 3 hexagons
    # Row 3: 4 hexagons? No, 3+4+3 = 10. Need 2 more.
    
    # Let's try a "Hexagon of Hexagons" minus corners?
    # Center 1, Ring 1 (6), Ring 2 (12). Total 19.
    # Remove 7 from Ring 2?
    
    # Let's just use a simple rectangular-ish honeycomb block and let the outer hexagon rotate.
    # A 2x6 strip?
    # A 3x4 block?
    
    # Let's generate a 3x4 honeycomb block.
    # Rows:
    # Row 0: 4 hexes
    # Row 1: 3 hexes (offset)
    # Row 2: 4 hexes
    # Row 3: 1 hex? No.
    
    # Let's try:
    # Row 0: 3
    # Row 1: 4
    # Row 2: 3
    # Row 3: 2
    # Total: 12.
    
    def make_honeycomb_block(counts, offset_x=0.5, offset_y=math.sqrt(3)/2):
        centers = []
        current_y = 0
        for r, count in enumerate(counts):
            for c in range(count):
                x = c * 2.0 + (offset_x if r % 2 == 1 else 0)
                y = r * 2 * offset_y
                centers.append((x, y))
        # Center it
        cx_mean = sum(c[0] for c in centers) / len(centers)
        cy_mean = sum(c[1] for c in centers) / len(centers)
        return [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # Candidate 1: 3-4-3-2
    centers_1 = make_honeycomb_block([3, 4, 3, 2])
    arrangements.append((centers_1, [0.0]*N))
    
    # Candidate 2: 2-3-4-3
    centers_2 = make_honeycomb_block([2, 3, 4, 3])
    arrangements.append((centers_2, [0.0]*N))
    
    # Candidate 3: 4-3-3-2
    centers_3 = make_honeycomb_block([4, 3, 3, 2])
    arrangements.append((centers_3, [0.0]*N))
    
    # Candidate 4: 1-4-4-3
    centers_4 = make_honeycomb_block([1, 4, 4, 3])
    arrangements.append((centers_4, [0.0]*N))
    
    # Candidate 5: Symmetric 2-4-4-2
    centers_5 = make_honeycomb_block([2, 4, 4, 2])
    arrangements.append((centers_5, [0.0]*N))
    
    # Candidate 6: 3-5-4
    centers_6 = make_honeycomb_block([3, 5, 4])
    arrangements.append((centers_6, [0.0]*N))
    
    # Candidate 7: 4-5-3
    centers_7 = make_honeycomb_block([4, 5, 3])
    arrangements.append((centers_7, [0.0]*N))
    
    # For each arrangement, we optimize the outer hexagon angle and center.
    # Since the inner hexagons are fixed relative to each other, we can just find the best
    # containing hexagon.
    
    for centers, angles in arrangements:
        if time.time() > deadline - 10.0:
            break
            
        # Try different outer angles
        # The optimal angle is likely aligned with the lattice or its dual.
        # We'll do a coarse search on angle, then refine.
        
        best_local_side = float('inf')
        best_local_angle = 0.0
        best_local_cx = 0.0
        best_local_cy = 0.0
        
        # Coarse search for outer angle
        for outer_angle_deg in np.linspace(0, 60, 13):
            # For a given outer angle, the optimal center is the centroid of the projection?
            # Actually, for a convex polygon containing a set of points, the minimal enclosing hexagon
            # with fixed orientation can be found by adjusting the center.
            # Here we assume the outer hexagon is centered at the centroid of the inner shapes
            # for a first approximation, then we might adjust.
            
            # Let's just compute side length assuming center is at (0,0) initially.
            # But we can shift the outer hexagon to minimize the side length.
            # The side length is determined by the max projection.
            # To minimize the max projection, we should center the bounding box of projections.
            
            outer_angle_rad = math.radians(outer_angle_deg)
            normals = []
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normals.append(np.array([math.cos(normal_angle), math.sin(normal_angle)]))
            
            # Compute projections of all vertices
            all_proj = []
            for i in range(N):
                verts = hex_vertices(centers[i][0], centers[i][1], angles[i])
                for n in normals:
                    proj = np.dot(verts, n)
                    all_proj.extend(proj)
            
            min_proj = min(all_proj)
            max_proj = max(all_proj)
            
            # The required "radius" (distance from center to side) is (max_proj - min_proj) / 2
            # If we center the outer hexagon at (min_proj + max_proj) / 2 along the normal?
            # No, the center must be a single point (cx, cy).
            
            # Let's approximate the optimal center as the average of the "extreme" points?
            # Or just use the centroid of the centers.
            cx = sum(c[0] for c in centers) / N
            cy = sum(c[1] for c in centers) / N
            
            # Actually, for a fixed orientation, the minimal enclosing hexagon is not necessarily centered at the centroid.
            # However, for symmetric clusters, it often is.
            # Let's calculate the side length with this center.
            
            side = compute_outer_side(centers, angles, cx, cy, outer_angle_deg)
            
            if side < best_local_side:
                best_local_side = side
                best_local_angle = outer_angle_deg
                best_local_cx = cx
                best_local_cy = cy
                
        # Refine angle around best
        if best_local_side < float('inf'):
            for outer_angle_deg in np.linspace(best_local_angle - 5, best_local_angle + 5, 11):
                cx = sum(c[0] for c in centers) / N
                cy = sum(c[1] for c in centers) / N
                side = compute_outer_side(centers, angles, cx, cy, outer_angle_deg)
                if side < best_local_side:
                    best_local_side = side
                    best_local_angle = outer_angle_deg
                    best_local_cx = cx
                    best_local_cy = cy
        
        # Check validity and update global best
        side, valid = evaluate_config(centers, angles, best_local_cx, best_local_cy, best_local_angle)
        if valid and side < best_side:
            best_side = side
            best_config = (centers, angles, best_local_cx, best_local_cy, best_local_angle)
            
    # If we found a valid config
    if best_config:
        centers, angles, cx, cy, angle = best_config
        inner = []
        for i in range(N):
            inner.append([float(centers[i][0]), float(centers[i][1]), float(angles[i])])
        
        # Add small safety margin
        final_side = best_side * 1.001 + 1e-5
        
        return inner, (float(cx), float(cy)), float(final_side), float(angle)
    else:
        # Fallback: Return a large valid configuration
        # Simple grid that definitely fits
        centers = []
        for r in range(4):
            for c in range(3):
                x = c * 2.2
                y = r * 2.0
                if r % 2 == 1:
                    x += 1.1
                centers.append((x, y))
        
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        
        inner = []
        for i in range(N):
            inner.append([float(centers[i][0]), float(centers[i][1]), 0.0])
            
        # Calculate side for this fallback
        side = compute_outer_side(centers, [0.0]*N, 0.0, 0.0, 0.0)
        
        return inner, (0.0, 0.0), float(side * 1.1), 0.0

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
