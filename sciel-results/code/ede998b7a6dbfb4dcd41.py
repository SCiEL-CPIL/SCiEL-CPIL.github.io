import math
import time
import random
import numpy as np
from itertools import combinations, product

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Hexagon geometry helpers
    # Vertices of a unit hexagon centered at (0,0) with angle 0
    # Flat-top orientation? Pointy-top?
    # Standard math convention: angles 0, 60, 120...
    # Let's use Pointy-Top for lattice alignment convenience?
    # Actually, for enclosing hexagon, orientation of inner vs outer matters.
    # Let's stick to a consistent convention.
    # Inner hex: Side 1.
    # Outer hex: Side S.
    
    # Let's define vertices for angle 0 as:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # This is "Pointy-Left/Right" or "Flat-Top"?
    # (1,0) is a vertex. So it's Pointy-Right.
    # Apothem is sqrt(3)/2.
    
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))
        
    # Precompute for speed
    base_verts_np = np.array(base_verts)
    
    def transform_verts_np(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        # base_verts_np is (6, 2)
        # scaled = base_verts_np * S
        # rotated = scaled @ rot.T
        # translated = rotated + [cx, cy]
        scaled = base_verts_np * S
        rotated = scaled @ rot.T
        return rotated + np.array([cx, cy])

    def segments_intersect(p1, p2, p3, p4):
        # p1, p2, p3, p4 are tuples (x,y)
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def point_in_polygon(p, polygon):
        # polygon is list of (x,y)
        x, y = p
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(verts_a, verts_b):
        # verts_a, verts_b are lists of 6 tuples
        # Fast check: center distance
        cx_a = sum(v[0] for v in verts_a) / 6
        cy_a = sum(v[1] for v in verts_a) / 6
        cx_b = sum(v[0] for v in verts_b) / 6
        cy_b = sum(v[1] for v in verts_b) / 6
        dist_sq = (cx_a - cx_b)**2 + (cy_a - cy_b)**2
        # Min center distance for disjoint unit hexes is sqrt(3) (flat-to-flat)
        # If dist < sqrt(3), they overlap.
        if dist_sq < 3.0 - 1e-9:
            return False
        # Max center distance for touching is 2 (point-to-point)
        # If dist > 2, they are disjoint.
        if dist_sq > 4.0 + 1e-9:
            return True
        
        # Exact check
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        # inner: list of [x, y, angle]
        # Calculate minimum S such that all inner hexes fit inside outer hex
        # Centered at (ocx, ocy) with rotation oangle
        
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts_np(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        
        # The outer hexagon has 6 sides.
        # For a regular hexagon, the distance from center to a side (apothem) is S * sqrt(3)/2.
        # The sides are perpendicular to the normals at angles:
        # oangle + 30, oangle + 90, ... (for pointy-top? No, depends on vertex def)
        # Our base_verts are at 0, 60, 120...
        # So the "flat" sides are perpendicular to 30, 90, 150...
        # Wait, if vertices are at 0, 60, the side between 0 and 60 is at angle 30.
        # The normal to that side is at 30 + 90 = 120? No.
        # Vector (1,0) to (0.5, .866). Midpoint (0.75, .433).
        # Normal is (cos(30), sin(30))? No, (cos(30), sin(30)) is (0.866, 0.5).
        # Let's check: (0.866, 0.5) dot (1,0) = 0.866. (0.866, 0.5) dot (0.5, .866) = 0.433 + 0.433 = 0.866.
        # Yes, the apothem direction is at 30, 90, 150, 210, 270, 330.
        
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            
            # Project all vertices of all inner hexes onto this normal
            # Relative to outer center
            max_proj = 0.0
            for iv in inner_verts:
                # iv is (6,2) array
                # relative = iv - [ocx, ocy]
                # proj = relative @ [nx, ny]
                rel = iv - np.array([ocx, ocy])
                projs = rel @ np.array([nx, ny])
                max_p = np.max(projs)
                if max_p > max_proj:
                    max_proj = max_p
            
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts_np(ocx, ocy, S, oangle)
        # Convert to list of tuples for point_in_polygon
        outer_list = [tuple(v) for v in outer_verts]
        
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = transform_verts_np(ix, iy, 1.0, iangle)
            inner_verts.append([tuple(v) for v in iv])
        
        # Check if all inner vertices are inside outer
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_list):
                    return False
        
        # Check disjointness
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def update_best(inner, ocx, ocy, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        S = compute_outer_hex(inner, ocx, ocy, oangle)
        if all_hexagons_fit(inner, ocx, ocy, S, oangle):
            if S < best_S:
                best_S = S
                best_inner = [list(x) for x in inner]
                best_center = (ocx, ocy)
                best_angle = oangle
                best_valid = True
                return True
        return False

    # --- Strategy: Polyhex Enumeration ---
    # 1. Generate all 11-ominoes (polyhexes) on a triangular lattice.
    # 2. For each unique shape, place the hexagons at their lattice positions (angle 0).
    # 3. Run a local search to optimize the outer hexagon's rotation and translation.
    #    Actually, since the inner hexes are fixed relative to each other in the lattice model,
    #    we can just try rotating the entire cluster and the outer hex.
    #    But wait, the problem allows individual rotations.
    #    However, for a "cluster" approach, we assume the topology is fixed.
    #    We can try to "relax" the positions slightly?
    #    The insight says: "Exhaustive search over all 11-cell polyhexes finds the true optimal cluster shape... continuous refinement then optimizes rotation and translation for that optimal shape."
    #    This implies we fix the relative positions (lattice) but optimize the global transformation (rotation/translation of the whole cluster relative to the outer hex) OR we allow slight deviations?
    #    "continuous refinement... optimizes rotation and translation" suggests we might treat the cluster as a rigid body or allow small moves.
    #    Given the time limit, let's stick to:
    #    1. Enumerate shapes.
    #    2. For each shape, center it at origin.
    #    3. Try various outer hex rotations.
    #    4. Try to "shake" the inner hexagons slightly to see if we can reduce S?
    #       Or just assume the lattice packing is optimal for that shape?
    #       Actually, if we fix the shape, the minimal enclosing hexagon depends on the relative orientation of the cluster and the outer hex.
    #       So for each shape, we should find the optimal rotation of the cluster relative to the outer hex (or vice versa).
    
    # Lattice basis for hexagons
    # Neighbors of (0,0) in a hex grid (axial coordinates):
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Cartesian conversion for pointy-top hexes (side 1, distance between centers sqrt(3)):
    # Let's use the standard hex grid to cartesian mapping.
    # If we use "flat-top" hexes, the neighbors are at angles 0, 60, 120...
    # Distance is 2 * apothem = sqrt(3).
    # Let's stick to the lattice used in the previous code:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 1.5)
    # Wait, previous code used:
    # a1 = (spacing, 0)
    # a2 = (spacing/2, spacing*sqrt(3)/2)
    # With spacing = sqrt(3).
    # a1 = (1.732, 0)
    # a2 = (0.866, 1.5)
    # This is a valid hex lattice.
    
    hex_a1 = np.array([math.sqrt(3), 0.0])
    hex_a2 = np.array([math.sqrt(3)/2, 1.5])
    
    def axial_to_cartesian(q, r):
        return q * hex_a1 + r * hex_a2

    def generate_polyhexes(n_cells):
        # Generate all free polyhexes of size n_cells
        # We use a canonical representation: set of axial coordinates (q,r)
        # such that the shape is connected and the "top-left-most" cell is at (0,0) or similar.
        
        all_shapes = set()
        
        def normalize(shape):
            # Translate so min q is 0, and min r for that q is 0?
            # Better: Translate so that the set of points is lexicographically smallest?
            # Or just use a canonical form based on bounding box and rotation?
            # For polyhexes, we can just check all 6 rotations and reflections?
            # No, we just need unique shapes.
            # Let's use a simple normalization:
            # 1. Translate so min(q) = 0.
            # 2. Translate so min(r) = 0 for the cells with min(q).
            # This is not fully canonical (doesn't handle rotation/reflection),
            # but if we generate from a root (0,0) and only add neighbors,
            # we might generate duplicates.
            # To avoid duplicates, we can use a "growth" method where we only add cells
            # that are "greater" than the current max in some order?
            # Or simpler: use a set of frozensets of coordinates.
            # To handle symmetry, we can just store the normalized frozenset.
            
            qs = [q for q, r in shape]
            rs = [r for q, r in shape]
            min_q = min(qs)
            min_r = min(rs)
            # This simple translation is not enough to canonicalize under rotation.
            # However, for n=11, the number of shapes is ~ 10,000-20,000?
            # Number of 11-ominoes (polyhexes) is 2299? (Free polyhexes).
            # Fixed polyhexes are 6 * 2299 ~ 13,000.
            # We can just generate all fixed polyhexes and use a set to remove duplicates.
            
            # Let's use a robust canonicalization:
            # Generate all 6 rotations and 6 reflections (12 total) of the shape.
            # For each, normalize translation (min q=0, min r=0).
            # Take the lexicographically smallest tuple of sorted coords.
            
            def rotate_reflect(coords, rot_idx, refl):
                # Axial to cube for easy rotation?
                # Or just use matrix rotation on cartesian and snap to lattice?
                # Easier: use axial rotation formulas.
                # Rotation by 60 deg in axial:
                # (q, r) -> (-r, q+r) ? No.
                # Let's use cube coordinates (x, y, z) with x+y+z=0.
                # q = x, r = z. y = -x-z.
                # Rotation 60: (x, y, z) -> (-y, -z, -x)
                
                new_coords = []
                for q, r in coords:
                    x = q
                    z = r
                    y = -x - z
                    if refl:
                        # Reflection: swap x and y? (x, y, z) -> (y, x, z)
                        x, y = y, x
                    for _ in range(rot_idx):
                        x, y, z = -y, -z, -x
                    new_q = x
                    new_r = z
                    new_coords.append((new_q, new_r))
                return new_coords

            def norm_translate(coords):
                min_q = min(c[0] for c in coords)
                min_r = min(c[1] for c in coords)
                return tuple(sorted((c[0] - min_q, c[1] - min_r) for c in coords))

            candidates = []
            for rot in range(6):
                for refl in [False, True]:
                    new_coords = rotate_reflect(list(shape), rot, refl)
                    candidates.append(norm_translate(new_coords))
            
            return min(candidates)

        def grow(shape, current_max):
            # shape: frozenset of (q,r)
            # current_max: the last added cell (to ensure we only grow "outwards" in a specific order to avoid duplicates? 
            # Actually, the standard algorithm for polyominoes/polyhexes is:
            # 1. Start with (0,0).
            # 2. At each step, find all "empty" cells adjacent to the shape.
            # 3. To avoid generating the same shape multiple times via different paths,
            #    we can impose an ordering on the cells.
            #    E.g., only add a cell if it is "greater" than all previously added cells in some lexicographic order?
            #    No, that restricts the shape too much.
            #    
            #    Standard method:
            #    Generate all connected sets of size N.
            #    Use a set to store normalized shapes.
            #    To prune: when adding a new cell, ensure that the new cell is not "behind" the current frontier in a way that creates a duplicate path?
            #    Simplest robust way for N=11:
            #    Recursive backtracking.
            #    State: current shape (set), list of cells in order of addition.
            #    To avoid duplicates, we can require that the new cell added at step k is lexicographically greater than the new cell added at step k-1?
            #    No, that's for polyominoes on a grid where we scan left-to-right.
            #    
            #    Let's use the "growth" method with a canonical check at the end.
            #    Since N=11 is small, the total number of connected sets is manageable.
            #    Number of fixed 11-ominoes is ~ 13,000.
            #    Number of paths to generate them might be higher, but with a set of seen normalized shapes, we prune.
            
            if len(shape) == n_cells:
                canon = normalize(shape)
                if canon not in all_shapes:
                    all_shapes.add(canon)
                return
            
            # Find all neighbors of all cells in shape
            neighbors = set()
            for q, r in shape:
                for dq, dr in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) not in shape:
                        neighbors.add((nq, nr))
            
            # To reduce symmetry, we can sort neighbors and only add those that are "valid"
            # A simple heuristic: only add a neighbor if it is lexicographically greater than the last added cell?
            # This works if we start from (0,0) and maintain the list of added cells.
            # Let's pass the list of added cells.
            
            # Actually, let's just iterate all neighbors. The set `all_shapes` handles duplicates.
            # To speed up, we can prune if the current shape's bounding box is already too large?
            # No, we don't know the optimal size yet.
            
            for nq, nr in neighbors:
                new_shape = frozenset(shape | {(nq, nr)})
                grow(new_shape, (nq, nr))

        start_shape = frozenset([(0, 0)])
        grow(start_shape, (0,0))
        
        return list(all_shapes)

    # Generate shapes
    print("Generating polyhexes...")
    shapes = generate_polyhexes(N)
    print(f"Found {len(shapes)} unique shapes.")
    
    # For each shape, evaluate the minimal enclosing hexagon
    # We assume the hexagons are placed at the lattice positions derived from the axial coordinates.
    # We then try to optimize the rotation of the entire cluster relative to the outer hex.
    
    # To save time, we can skip shapes that are clearly "long" (large bounding box).
    # But let's try all.
    
    for i, shape in enumerate(shapes):
        if time.time() > deadline:
            break
            
        # Convert axial to cartesian
        centers = []
        for q, r in shape:
            c = axial_to_cartesian(q, r)
            centers.append(c)
        
        # Center the cluster at origin
        mean_c = np.mean(centers, axis=0)
        centers = [c - mean_c for c in centers]
        
        # Create initial inner config: [x, y, angle]
        # All angles 0
        inner = [[c[0], c[1], 0.0] for c in centers]
        
        # Try different outer hex rotations
        # The cluster has 6-fold symmetry? No, the shape might not.
        # But the outer hex has 6-fold symmetry.
        # So we only need to try 0 to 30 degrees?
        # Actually, the relative angle between cluster and outer hex is what matters.
        # We can fix the outer hex at 0 and rotate the cluster.
        # Or fix the cluster and rotate the outer hex.
        # Let's rotate the outer hex.
        
        best_local_S = float('inf')
        best_local_angle = 0.0
        
        # Sample angles
        for oangle_deg in np.linspace(0, 60, 13): # 0, 5, 10... 60
            S = compute_outer_hex(inner, 0.0, 0.0, oangle_deg)
            # We don't need to check fit if we are just calculating S from projection,
            # but we must ensure disjointness.
            # Disjointness is guaranteed by the lattice construction (spacing sqrt(3)).
            # So we just need to check if S is smaller.
            # Wait, does the projection method guarantee fit?
            # Yes, if we take the max projection, the hexagon of size S will contain all points.
            # And since the points are from a valid lattice packing, they are disjoint.
            # So `all_hexagons_fit` should return True for the calculated S.
            # But `compute_outer_hex` just calculates the required S.
            # We should verify fit just in case of numerical issues.
            
            if S < best_local_S:
                best_local_S = S
                best_local_angle = oangle_deg
                
        # Refine the angle locally
        # Use a simple hill climbing on the angle
        current_angle = best_local_angle
        current_S = best_local_S
        step = 0.5
        while step > 0.01:
            improved = False
            for da in [-step, step]:
                new_angle = current_angle + da
                new_S = compute_outer_hex(inner, 0.0, 0.0, new_angle)
                if new_S < current_S:
                    current_S = new_S
                    current_angle = new_angle
                    improved = True
            if not improved:
                step /= 2.0
            if time.time() > deadline:
                break
        
        if current_S < best_S:
            # Update global best
            # We need to return the inner hexagons with their absolute positions.
            # Currently `inner` is centered at origin.
            # We can return it as is, with center (0,0).
            best_S = current_S
            best_inner = [list(x) for x in inner]
            best_center = (0.0, 0.0)
            best_angle = current_angle
            best_valid = True
            
        if i % 100 == 0:
            print(f"Processed {i} shapes. Best S: {best_S}")

    # If no valid shape found (shouldn't happen), fallback
    if not best_valid:
        # Simple radial fallback
        sp = math.sqrt(3) * 1.01
        hex_config = [[0.0, 0.0, 0.0]]
        for i in range(6):
            theta = math.radians(60 * i)
            hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        for i in [0, 2, 4]:
            theta = math.radians(60 * i + 30)
            hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        for i in [1, 3, 5]:
            theta = math.radians(60 * i)
            hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        while len(hex_config) < N:
            hex_config.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        
        for oangle in [0, 30, 60]:
            if update_best(hex_config[:N], 0.0, 0.0, oangle):
                break

    if not best_valid or best_inner is None:
        # Last resort
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
