"""Strategy: Constrained lattice optimization with fine grid search."""
import math
import time
import numpy as np

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Check if two hexagons overlap using Separating Axis Theorem (SAT)
    def is_disjoint_fast(hex1_verts, hex2_verts):
        # For unit hexagons, we can just check all 12 axes (6 from each)
        # But since we are fixing angles to 0 or 60 in the grid search, 
        # we can pre-compute or use a faster specific check.
        # However, for robustness in the final validation, we use SAT.
        
        # Get axes for hex1 (angle 0)
        axes1 = []
        for i in range(6):
            theta = math.radians(0 + i * 60 + 30)
            axes1.append((math.cos(theta), math.sin(theta)))
        
        # Get axes for hex2 (angle 0)
        # If both are angle 0, axes are identical.
        # If one is rotated, we need to handle it.
        # In our grid search, we mostly use angle 0. 
        # Let's assume general case for safety, but optimize for 0.
        
        # Actually, if we fix all angles to 0, the axes are the same for all.
        # There are only 6 unique axes.
        axes = []
        for i in range(6):
            theta = math.radians(i * 60 + 30)
            axes.append((math.cos(theta), math.sin(theta)))
            
        for axis in axes:
            nx, ny = axis
            min1 = max1 = None
            min2 = max2 = None
            for vx, vy in hex1_verts:
                proj = vx * nx + vy * ny
                if min1 is None or proj < min1:
                    min1 = proj
                if max1 is None or proj > max1:
                    max1 = proj
            for vx, vy in hex2_verts:
                proj = vx * nx + vy * ny
                if min2 is None or proj < min2:
                    min2 = proj
                if max2 is None or proj > max2:
                    max2 = proj
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    # Optimized: precompute cos/sin for outer hexagon axes
    def calc_outer_side_fast(hexagons, outer_angle_deg):
        # The 6 normal directions of the outer hexagon
        # For a regular hexagon with side S, the apothem is S * SQRT3 / 2.
        # The vertices are at distance S from center.
        # The "width" in the direction of the normal to a side is the apothem.
        # So we need max projection of any vertex onto the 6 normal directions to be <= apothem.
        
        base_angles = [30, 90, 150, 210, 270, 330] # Normals for hex with vertex at 0
        # Wait, if outer hex has vertex at angle 0, its normals are at 30, 90, etc.
        # If outer hex has vertex at angle alpha, its normals are at alpha+30, etc.
        
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem = max_proj
        # S * SQRT3 / 2 = max_proj  => S = 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # Generate candidate configurations
    # We focus on the "honeycomb" lattice structure which is known to be optimal for small N.
    # The 11 hexagons form a shape of side 2 (which has 7 hexes) plus 4 more, or side 3 (19 hexes) minus 8.
    # A compact shape for 11 is:
    #   . X .
    #  X X X X
    #   . X .
    # This is 7 hexes (side 2). We need 4 more.
    # Adding them to the ends of the 3 arms of the side-2 hexagon creates a "star" or "flower" shape.
    # Or, a 3x3 grid in axial coordinates?
    
    # Let's define a set of "template" relative positions for the 11 hexagons.
    # All angles are fixed to 0 for the inner hexagons to maximize packing density in the lattice.
    # We will search for the best "centering" and "rotation" of this cluster.
    
    # Template 1: "Flower" - Center + 6 neighbors + 4 next-neighbors (corners of the hex ring?)
    # Actually, the most compact 11-hexagon packing is often derived from the infinite honeycomb.
    # Let's take a patch of the honeycomb.
    
    # Axial coordinates (q, r) for a compact 11-hex cluster:
    # Center: (0,0)
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  -- 7 total
    # Next layer (pick 4 to minimize bounding box):
    # (2,0), (2,-1), (1,1), (0,2) ? No, let's look at the "side 2" hexagon which has 7.
    # To get 11, we add 4 to the perimeter.
    # A symmetric addition: Add to (2,0), (-2,0) is not possible directly in axial without breaking symmetry easily.
    
    # Let's use a specific known good packing for 11:
    # It fits in a hexagon of side length S.
    # If S=3, area is proportional to 9. 11 hexes have area 11. 11/9 > 1, so S must be > 3?
    # No, side length S hexagon contains 3S^2 - 3S + 1 hexes in a perfect lattice.
    # S=1: 1
    # S=2: 7
    # S=3: 19
    # So 11 hexes cannot fit in a perfect S=2 hexagon (only 7 fit).
    # They must fit in a shape that is "wider" than S=2 but "narrower" than S=3 in some directions, 
    # or the outer hexagon must be rotated.
    
    # The minimal side length for 11 unit hexagons is known to be around 3.0 to 3.2?
    # Actually, if we just take the 7 hexes of S=2, and add 4 more, the bounding box expands.
    
    # Let's generate a list of candidate clusters based on honeycomb patches.
    # We will represent the cluster by the relative positions of the 11 centers.
    
    # Cluster A: The "Hexagon of Side 2" (7 hexes) + 4 attached to the flat sides.
    # Centers in Cartesian (unit hex, pointy top or flat top? 
    # Standard: Vertices at 0, 60, 120... 
    # Neighbors are at distance SQRT3.
    # Let's use axial to Cartesian conversion.
    def axial_to_cartesian(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Define a few candidate sets of (q, r) coordinates
    candidates = []
    
    # Candidate 1: Compact cluster
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  [7]
    # Add: (2,0), (-2,0), (1,1), (-1,-1) ? 
    # Let's just brute force a small set of "shapes" defined by axial coords.
    
    shape1 = [
        (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    # Check if this is valid (no overlaps in lattice).
    # In a perfect lattice, any distinct (q,r) are non-overlapping.
    
    # Convert to Cartesian
    cart1 = [axial_to_cartesian(q, r) for q, r in shape1]
    # Center it
    cx_avg = sum(x for x, y in cart1) / N
    cy_avg = sum(y for x, y in cart1) / N
    cart1_centered = [(x - cx_avg, y - cy_avg) for x, y in cart1]
    candidates.append(cart1_centered)
    
    # Candidate 2: A different 11-hex shape
    # Row r=-1: q=-1, 0, 1
    # Row r=0: q=-2, -1, 0, 1, 2
    # Row r=1: q=-1, 0, 1
    # This is 3+5+3 = 11. This is a "rhombus" or "parallelogram" like shape.
    shape2 = [
        (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1)
    ]
    cart2 = [axial_to_cartesian(q, r) for q, r in shape2]
    cx_avg = sum(x for x, y in cart2) / N
    cy_avg = sum(y for x, y in cart2) / N
    cart2_centered = [(x - cx_avg, y - cy_avg) for x, y in cart2]
    candidates.append(cart2_centered)
    
    # Candidate 3: "Star" shape
    shape3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 2), (-2, 1) # Adding 3? Need 4.
        # Let's add (0,2) or similar.
        # Actually, let's just use the first two which are very compact.
    ]
    # If shape3 is not 11, discard.
    if len(shape3) == 11:
         cart3 = [axial_to_cartesian(q, r) for q, r in shape3]
         cx_avg = sum(x for x, y in cart3) / N
         cy_avg = sum(y for x, y in cart3) / N
         cart3_centered = [(x - cx_avg, y - cy_avg) for x, y in cart3]
         candidates.append(cart3_centered)
         
    # Add a random perturbation candidate to break symmetry if needed?
    # No, the lattice is the best bet.
    
    best_side = float('inf')
    best_config = None
    best_outer_angle = 0.0
    
    # For each candidate shape, we search for the best outer hexagon angle.
    # We also allow a small global translation (centering is already done, but we can shift the whole cluster relative to the outer hex center).
    # Since the outer hex is centered at (0,0) in our calculation, shifting the cluster is equivalent to shifting the outer hex, 
    # but the problem asks for inner coords relative to outer center.
    # So we can just try shifting the cluster by (dx, dy).
    
    # Grid search for outer angle (0 to 60 degrees)
    # And grid search for cluster shift (dx, dy) in a small range.
    
    # To save time, we first find the best angle for the centered cluster.
    # Then refine.
    
    for i, cluster in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
            
        # Pre-compute vertices for this cluster (angle 0)
        hexagons = []
        for x, y in cluster:
            hexagons.append(get_hex_vertices(x, y, 0.0))
            
        # Check validity of this specific cluster (should be valid by construction, but just in case)
        valid = True
        for a in range(N):
            for b in range(a+1, N):
                if not is_disjoint_fast(hexagons[a], hexagons[b]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Search outer angle
        # The bounding box size depends on the angle.
        # We scan 0 to 60 with step 0.5 degrees.
        best_local_side = float('inf')
        best_local_angle = 0.0
        
        # We can also try shifting the cluster slightly to optimize the fit.
        # Let's try a few shifts: (0,0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)
        shifts = [(0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1), (0.05, 0.05), (-0.05, -0.05)]
        
        for dx, dy in shifts:
            # Shifted hexagons
            shifted_hexagons = []
            for x, y in cluster:
                shifted_hexagons.append(get_hex_vertices(x + dx, y + dy, 0.0))
                
            for oa_deg in range(0, 600, 5): # 0.1 deg steps? 600 steps is fast.
                oa = oa_deg * 0.1
                side = calc_outer_side_fast(shifted_hexagons, oa)
                if side < best_local_side:
                    best_local_side = side
                    best_local_angle = oa
                    best_local_shift = (dx, dy)
                    
        if best_local_side < best_side:
            best_side = best_local_side
            best_config = cluster
            best_outer_angle = best_local_angle
            best_shift = best_local_shift
            
    # Refine the best found angle with finer steps
    if best_config is not None:
        # Use the best shift found
        dx, dy = best_shift
        refined_hexagons = []
        for x, y in best_config:
            refined_hexagons.append(get_hex_vertices(x + dx, y + dy, 0.0))
            
        # Fine scan around best_outer_angle
        # Range: best_angle +/- 2 degrees, step 0.01
        best_refined_side = float('inf')
        best_refined_angle = best_outer_angle
        
        start_a = max(0, best_outer_angle - 2.0)
        end_a = min(60, best_outer_angle + 2.0)
        
        # If we hit the boundary, wrap around? 
        # Hexagon symmetry is 60 deg.
        # If best_outer_angle is near 0, check near 60 too.
        
        angles_to_check = []
        for a in np.arange(start_a, end_a, 0.05):
            angles_to_check.append(a)
        if best_outer_angle < 2.0:
            for a in np.arange(58.0, 60.0, 0.05):
                angles_to_check.append(a)
        if best_outer_angle > 58.0:
            for a in np.arange(0.0, 2.0, 0.05):
                angles_to_check.append(a)
                
        for oa in angles_to_check:
            side = calc_outer_side_fast(refined_hexagons, oa)
            if side < best_refined_side:
                best_refined_side = side
                best_refined_angle = oa
                
        if best_refined_side < best_side:
            best_side = best_refined_side
            best_outer_angle = best_refined_angle

    # Finalize result
    # best_config is the list of (x,y) relative to cluster center.
    # We need to return inner list with (x, y, angle).
    # Angle is 0 for all.
    # Outer center is (0,0).
    # Outer side is best_side.
    # Outer angle is best_outer_angle.
    
    inner_list = []
    for x, y in best_config:
        inner_list.append([x + best_shift[0], y + best_shift[1], 0.0])
        
    return (inner_list, (0.0, 0.0), best_side, best_outer_angle)

# EVOLVE-BLOCK-END
