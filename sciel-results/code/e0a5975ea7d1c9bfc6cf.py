import numpy as np
import time
import itertools

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Generate a candidate set of centers on a hexagonal lattice
    # The optimal packing for hexagons is typically a honeycomb structure.
    # We generate centers on a lattice with spacing slightly less than the
    # touching distance (SQRT3) to allow for "sliding" or tighter packing
    # if rotations are involved, but mostly we rely on the lattice structure.
    
    # Lattice basis vectors for hexagonal packing (side length 1)
    # Distance between touching centers is SQRT3.
    # We'll use a finer grid to catch non-standard offsets if rotations allow it.
    # However, for hexagons, the "hard" constraint is usually lattice-like.
    # Let's generate a grid of potential centers.
    
    max_radius = 10.0
    step = 0.1 # Fine enough to catch small shifts, coarse enough to be fast?
    # Actually, 12 hexagons. If we use a grid, the number of candidates can explode.
    # Let's restrict the search space to a bounding box of the expected cluster.
    # Expected cluster radius is around 3-4 units.
    
    # Instead of a dense grid, let's use the exact hexagonal lattice points
    # and their neighbors.
    # Axial coordinates (q, r)
    # x = SQRT3 * (q + r/2)
    # y = 1.5 * r
    
    # We will search over subsets of these lattice points.
    # How many lattice points to consider?
    # A cluster of 12 hexagons fits in a hexagon of radius 2 (7 in center, 6 in ring 1? No, 7+6=13).
    # So we need lattice points up to radius 2 or 3.
    
    lattice_points = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            lattice_points.append((x, y))
    
    lattice_points = np.array(lattice_points)
    
    # To allow for small deviations from the perfect lattice (which might be optimal
    # if the outer hexagon is rotated or if we use non-lattice rotations),
    # we could add small perturbations, but that increases combinatorial complexity.
    # Let's first try strict lattice points.
    
    # 2. Define relative orientations
    # Hexagons have 6-fold symmetry. Rotations by 60 degrees are equivalent.
    # However, relative rotation between two hexagons matters.
    # Possible relative angles: 0, 30, 60, 90, 120, 150.
    # Due to symmetry, 0, 30, 60 are the distinct ones?
    # If hex A is at 0, hex B at 30, it's different from B at 0.
    # Let's consider global rotation of the whole system as well.
    # For now, let's assume all inner hexagons are aligned with the outer one (angle 0)
    # OR aligned with each other.
    # The most efficient packing for regular hexagons is usually aligned (angle 0).
    # Let's focus on angle=0 for all inner hexagons first.
    # If that fails to improve, we can try mixed angles.
    
    # Strategy:
    # Iterate over all combinations of N lattice points.
    # This is C(M, N) where M is number of lattice points.
    # M is roughly 9x9 = 81. C(81, 12) is huge.
    
    # We need a smarter search.
    # Local search / Hill Climbing on the lattice.
    # Start with a valid packing (e.g., a patch of the honeycomb).
    # Move one hexagon to a nearby lattice point if it reduces the outer side length
    # and remains valid (disjoint).
    
    # Initial configuration: A compact cluster of 12.
    # Center (0,0).
    # Ring 1: 6 hexagons.
    # Ring 2: 5 hexagons (to make 12).
    
    def get_lattice_index(q, r):
        # Map (q,r) to index in lattice_points
        # This is slow, let's build a dict
        pass

    # Let's just use the array and find closest.
    # Better: Generate specific candidate sets.
    
    # Candidate Set 1: The "Honeycomb Patch"
    # Coordinates in axial (q, r)
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1)  <-- 5 of the 6 in ring 2
    
    base_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    def axial_to_xy(q, r):
        return (SQRT3 * (q + r / 2.0), 1.5 * r)
    
    init_positions = np.array([axial_to_xy(q, r) for q, r in base_coords])
    
    # Define a function to calculate the minimal outer hexagon side length
    # for a given set of positions (all angle 0).
    
    def calc_outer_side(positions):
        # positions: (N, 2)
        # Vertices of each inner hexagon (side 1, angle 0)
        # (1, 0), (0.5, SQRT3/2), (-0.5, SQRT3/2), (-1, 0), (-0.5, -SQRT3/2), (0.5, -SQRT3/2)
        
        # For a regular hexagon (outer) centered at origin, side S, angle 0.
        # A point (x, y) is inside if:
        # max( |x|, |y|*2/SQRT3, |x + y/SQRT3|, |x - y/SQRT3| ) <= S ?
        # Let's re-verify the inequality for angle 0.
        # Vertices: (S,0), (S/2, S*SQRT3/2), ...
        # Side 1: x = S. Normal (1,0). Distance |x| <= S.
        # Side 2: -SQRT3*x + y = S*SQRT3 ? No.
        # Line through (S,0) and (S/2, S*SQRT3/2).
        # Slope = (S*SQRT3/2 - 0) / (S/2 - S) = (SQRT3/2) / (-1/2) = -SQRT3.
        # Eq: y - 0 = -SQRT3 (x - S) => SQRT3 x + y = S*SQRT3.
        # Distance from origin to this line: |0 + 0 - S*SQRT3| / sqrt(3+1) = S*SQRT3/2.
        # Wait, the apothem is S*SQRT3/2.
        # So the constraint is (SQRT3 x + y) / 2 <= S*SQRT3/2 ?
        # SQRT3 x + y <= S*SQRT3.
        
        # Let's use the support function approach.
        # For a regular hexagon with side S, the width in direction theta is:
        # W(theta) = S * (1 + cos(60 - (theta mod 60)))? No.
        
        # Simpler: The distance from center to boundary in direction (nx, ny) is:
        # If we normalize (nx, ny) to unit vector u.
        # The hexagon is the intersection of 6 half-planes.
        # For angle 0 (vertex at right):
        # Normals to sides are at angles 0, 60, 120, 180, 240, 300?
        # No, side midpoints are at 30, 90, 150, 210, 270, 330.
        # Side 1 (right-most): x = S. Normal (1, 0).
        # Side 2 (top-right): SQRT3 x + y = S*SQRT3. Normal (SQRT3, 1). Norm: 2.
        #   Constraint: (SQRT3 x + y) / 2 <= S*SQRT3/2.
        #   => SQRT3 x + y <= S*SQRT3.
        
        # So for a point (x, y), the required S is the max of:
        # 1. x (from x <= S)
        # 2. -x (from -x <= S)
        # 3. (SQRT3 x + y) / SQRT3 (from SQRT3 x + y <= S*SQRT3)
        # 4. (-SQRT3 x + y) / SQRT3 (from -SQRT3 x + y <= S*SQRT3)
        # 5. -y (from -y <= S*SQRT3/2 ? No, let's check bottom side)
        #    Bottom side: y = -S*SQRT3/2. Normal (0, -1).
        #    Constraint: -y <= S*SQRT3/2 => y >= -S*SQRT3/2.
        #    So S >= -y * 2/SQRT3.
        #    Similarly top side: y <= S*SQRT3/2 => S >= y * 2/SQRT3.
        
        # Let's list all 6 constraints for S:
        # 1. x <= S  => S >= x
        # 2. -x <= S => S >= -x
        # 3. SQRT3*x + y <= S*SQRT3 => S >= (SQRT3*x + y)/SQRT3 = x + y/SQRT3
        # 4. -SQRT3*x + y <= S*SQRT3 => S >= (-SQRT3*x + y)/SQRT3 = -x + y/SQRT3
        # 5. y <= S*SQRT3/2 => S >= 2*y/SQRT3
        # 6. -y <= S*SQRT3/2 => S >= -2*y/SQRT3
        
        # So S_point = max(x, -x, x + y/SQRT3, -x + y/SQRT3, 2*y/SQRT3, -2*y/SQRT3)
        # Note: x and -x are just |x|.
        # x + y/SQRT3 and -x + y/SQRT3 are not just |x| + |y|/SQRT3.
        # Example: x=1, y=1.
        # |x|=1.
        # x+y/SQRT3 = 1 + 0.577 = 1.577.
        # -x+y/SQRT3 = -1 + 0.577 = -0.423.
        # 2y/SQRT3 = 1.154.
        # Max is 1.577.
        
        # Let's compute this for all vertices of all inner hexagons.
        
        max_s = 0.0
        for i in range(N):
            cx, cy = positions[i]
            # Vertices of inner hexagon at (cx, cy) with side 1, angle 0
            # (1, 0), (0.5, SQRT3/2), (-0.5, SQRT3/2), (-1, 0), (-0.5, -SQRT3/2), (0.5, -SQRT3/2)
            offsets = [
                (1.0, 0.0),
                (0.5, SQRT3/2),
                (-0.5, SQRT3/2),
                (-1.0, 0.0),
                (-0.5, -SQRT3/2),
                (0.5, -SQRT3/2)
            ]
            for dx, dy in offsets:
                x = cx + dx
                y = cy + dy
                s_req = max(
                    x, -x,
                    x + y/SQRT3,
                    -x + y/SQRT3,
                    2.0*y/SQRT3,
                    -2.0*y/SQRT3
                )
                if s_req > max_s:
                    max_s = s_req
        return max_s

    # Check disjointness for lattice-aligned hexagons.
    # Two hexagons at centers c1, c2 with angle 0 are disjoint if
    # the distance between centers is >= SQRT3 (touching) or > SQRT3 (separated).
    # Actually, for regular hexagons, they touch if distance == SQRT3.
    # If distance < SQRT3, they overlap.
    # So we just need to check pairwise distances.
    
    def is_valid(positions):
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < SQRT3 - 1e-9: # Overlapping
                    return False
        return True

    # Local Search on Lattice
    # We represent the state as a set of indices into a larger lattice grid.
    # To make it easier, let's define a function to find the nearest lattice point
    # to a given coordinate.
    
    # But we are restricted to the lattice points we generated?
    # No, we should allow moving to ANY lattice point.
    # Let's generate a larger set of lattice points to allow movement.
    
    large_lattice = []
    for q in range(-10, 11):
        for r in range(-10, 11):
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            large_lattice.append((x, y))
    large_lattice = np.array(large_lattice)
    
    # Map from (q,r) to index in large_lattice
    # q = -10..10, r = -10..10
    # Index = (q + 10) * 21 + (r + 10)
    def q_r_to_idx(q, r):
        return (q + 10) * 21 + (r + 10)
    
    def idx_to_q_r(idx):
        q = idx // 21 - 10
        r = idx % 21 - 10
        return q, r
    
    # Initialize with base_coords
    current_qr = [tuple(c) for c in base_coords]
    current_positions = np.array([axial_to_xy(q, r) for q, r in current_qr])
    
    if not is_valid(current_positions):
        # Fallback: spread them out
        # This shouldn't happen with the base config, but just in case.
        pass
        
    current_score = calc_outer_side(current_positions)
    best_score = current_score
    best_qr = list(current_qr)
    
    # Neighborhood: Move one hexagon to an adjacent lattice point.
    # Adjacent in axial coordinates: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Hill Climbing
    improved = True
    iterations = 0
    max_iterations = 10000
    
    while improved and iterations < max_iterations and time.time() < deadline:
        improved = False
        iterations += 1
        
        for i in range(N):
            q, r = current_qr[i]
            for dq, dr in neighbors:
                new_q, new_r = q + dq, r + dr
                
                # Check if this point is already occupied by another hexagon
                # We need to check against all other current_qr[j]
                occupied = False
                for j in range(N):
                    if i != j and current_qr[j] == (new_q, new_r):
                        occupied = True
                        break
                if occupied:
                    continue
                
                # Propose move
                old_qr = current_qr[i]
                current_qr[i] = (new_q, new_r)
                new_pos = axial_to_xy(new_q, new_r)
                
                # Update positions array for validity check
                # Instead of updating the whole array, just check the moved one against others
                # and calculate new score.
                
                # Check validity of the moved hexagon against others
                valid = True
                for j in range(N):
                    if i != j:
                        other_pos = axial_to_xy(current_qr[j])
                        dist = np.linalg.norm(new_pos - other_pos)
                        if dist < SQRT3 - 1e-9:
                            valid = False
                            break
                
                if not valid:
                    current_qr[i] = old_qr
                    continue
                
                # Calculate new score
                # We can optimize this by only recalculating for the moved hexagon?
                # No, the max over all points might change.
                # But we can keep a list of s_req for each hexagon's vertices?
                # Simpler: just recalc. N=12, 72 points. Fast.
                
                # Update positions for calc
                temp_positions = current_positions.copy()
                temp_positions[i] = new_pos
                
                new_score = calc_outer_side(temp_positions)
                
                if new_score < best_score - 1e-9:
                    # Accept move
                    best_score = new_score
                    best_qr = list(current_qr)
                    current_positions[i] = new_pos
                    improved = True
                else:
                    # Reject move
                    current_qr[i] = old_qr
                    
    # After local search, try a few random restarts if time permits?
    # Or just return the best found.
    
    # Construct final output
    inner = []
    for q, r in best_qr:
        x, y = axial_to_xy(q, r)
        inner.append([float(x), float(y), 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side length best_score
    # Angle 0.0
    
    return inner, (0.0, 0.0), float(best_score), 0.0
# EVOLVE-BLOCK-END
