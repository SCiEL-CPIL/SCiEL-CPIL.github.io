import math
import random
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # Helper functions for geometry
    
    def get_hex_vertices(x, y, theta):
        """Return list of (vx, vy) for a unit hexagon centered at (x,y) with rotation theta (radians)."""
        verts = []
        for i in range(6):
            angle = theta + i * math.pi / 3.0
            verts.append((x + math.cos(angle), y + math.sin(angle)))
        return verts

    def get_hex_normals(theta):
        """Return list of (nx, ny) for the 6 outward normals of a unit hexagon with rotation theta."""
        normals = []
        # The normal to the edge connecting vertex i and i+1.
        # Vertex i is at angle theta + i*pi/3.
        # The edge midpoint is at angle theta + (i + 0.5)*pi/3.
        # The outward normal is in the same direction as the midpoint from center.
        for i in range(6):
            angle = theta + (i + 0.5) * math.pi / 3.0
            normals.append((math.cos(angle), math.sin(angle)))
        return normals

    def project_hex(vertices, nx, ny):
        """Project vertices onto normal (nx, ny). Return (min_proj, max_proj)."""
        projs = [vx * nx + vy * ny for vx, vy in vertices]
        return min(projs), max(projs)

    def check_overlap(hex1, hex2):
        """
        hex1, hex2: (x, y, theta)
        Returns True if they overlap, False if separated.
        Uses SAT.
        """
        x1, y1, t1 = hex1
        x2, y2, t2 = hex2
        
        verts1 = get_hex_vertices(x1, y1, t1)
        verts2 = get_hex_vertices(x2, y2, t2)
        
        normals1 = get_hex_normals(t1)
        normals2 = get_hex_normals(t2)
        
        # Check separation along normals of hex1
        for nx, ny in normals1:
            min1, max1 = project_hex(verts1, nx, ny)
            min2, max2 = project_hex(verts2, nx, ny)
            # Add small epsilon for robustness
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False # Separated
        
        # Check separation along normals of hex2
        for nx, ny in normals2:
            min1, max1 = project_hex(verts1, nx, ny)
            min2, max2 = project_hex(verts2, nx, ny)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False # Separated
        
        return True # Overlapping

    def get_outer_vertices(cx, cy, side, theta):
        """Return vertices of the outer hexagon."""
        # Side length is 'side'. Distance from center to vertex is 'side'.
        verts = []
        for i in range(6):
            angle = theta + i * math.pi / 3.0
            verts.append((cx + side * math.cos(angle), cy + side * math.sin(angle)))
        return verts

    def is_inside_inner(hex_inner, outer_verts):
        """Check if all vertices of inner hex are inside outer hex."""
        x, y, theta = hex_inner
        verts_inner = get_hex_vertices(x, y, theta)
        
        # Check if each inner vertex is inside the outer polygon.
        # For a convex polygon, a point is inside if it is on the same side of all edges.
        # Outer hexagon is centered at (cx, cy) with side 'side'.
        # We can just check against the 6 half-planes defined by the outer edges.
        # Outer edge normals point INWARDS for the containment check? 
        # Let's define outer normals pointing OUTWARDS.
        # Point P is inside if dot(P - V_i, N_i) <= 0 for all edges?
        # Or simpler: Use the support function approach.
        # The distance from center to edge (apothem) of outer hex is a_out = side * sqrt(3)/2.
        # For each of the 6 directions (normals of outer hex), the projection of any inner point
        # must be <= a_out (relative to center) and >= -a_out.
        
        # Let's stick to standard point-in-convex-polygon using cross products or dot products.
        # Since outer is regular, we can use the 6 outward normals.
        # Normal angle for edge k is theta_out + (k + 0.5)*pi/3.
        # Wait, the normal to the edge between vertex k and k+1 is at angle theta + (k+0.5)*pi/3.
        # The edge is at distance a_out from center along this normal.
        
        # Let's compute outer normals and apothem.
        # We don't have theta_out explicitly passed in the helper, but we can assume outer is fixed at 0 rot, 0 center for simplicity?
        # The interface allows outer center and rotation. However, by symmetry, optimal solution likely has outer centered at 0,0 and aligned.
        # I will fix outer center to (0,0) and rotation to 0.0 to reduce search space.
        
        cx, cy = 0.0, 0.0
        theta_out = 0.0
        
        # Outer apothem
        a_out = side * math.sqrt(3) / 2.0
        
        # Check each vertex of inner hex
        for vx, vy in verts_inner:
            # Relative to outer center
            rx = vx - cx
            ry = vy - cy
            
            # Check against 6 half-planes
            # Normal angles: (k + 0.5) * pi/3
            for k in range(6):
                angle = (k + 0.5) * math.pi / 3.0
                nx = math.cos(angle)
                ny = math.sin(angle)
                
                # The boundary is at distance a_out along normal (nx, ny)
                # The point (rx, ry) must satisfy: rx*nx + ry*ny <= a_out
                # AND rx*nx + ry*ny >= -a_out (implied by checking all 6 normals which cover 360 deg)
                # Actually, checking all 6 outward normals with <= a_out is sufficient for convex body.
                proj = rx * nx + ry * ny
                if proj > a_out + 1e-9:
                    return False
        return True

    def calculate_energy(state, side):
        """
        state: list of (x, y, theta)
        side: outer side length
        Returns energy (penalty for violations).
        """
        penalty = 0.0
        
        # Check containment
        for h in state:
            if not is_inside_inner(h, side):
                # Penalty proportional to how far it is out?
                # Simple binary penalty is okay for SA if we handle overlaps separately.
                # Let's use a soft penalty: distance outside.
                # For now, just a large constant to encourage moving in.
                penalty += 1000.0
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                if check_overlap(state[i], state[j]):
                    # Penalty for overlap.
                    # Calculate minimum distance to separate? 
                    # Approximation: Penalty based on center distance vs sum of apothems?
                    # Sum of apothems = sqrt(3).
                    # If centers are closer than sqrt(3), they might overlap.
                    # Let's just use a constant penalty for now, or better, estimate overlap depth.
                    # For simplicity in this implementation, constant penalty.
                    penalty += 100.0
                    
        return penalty

    # Initial State
    # Place hexagons in a hexagonal grid pattern.
    # Spacing: 2 * apothem = sqrt(3) ~ 1.732.
    # Let's use spacing 1.8 to be safe.
    spacing = 1.8
    state = []
    
    # Generate grid coordinates
    # Hex grid offsets
    # Row 0: 0
    # Row 1: -1, 0, 1
    # etc.
    # We need 11 hexagons.
    # Center (0,0).
    # (0,0)
    # (1.8, 0), (-1.8, 0)
    # (0.9, 1.5), (0.9, -1.5), (-0.9, 1.5), (-0.9, -1.5)
    # (2.7, 1.5), (2.7, -1.5), (-2.7, 1.5), (-2.7, -1.5)
    # This is 1 + 2 + 4 + 4 = 11.
    
    coords = [
        (0.0, 0.0),
        (spacing, 0.0), (-spacing, 0.0),
        (spacing/2, spacing*math.sqrt(3)/2), (spacing/2, -spacing*math.sqrt(3)/2),
        (-spacing/2, spacing*math.sqrt(3)/2), (-spacing/2, -spacing*math.sqrt(3)/2),
        (spacing*1.5, spacing*math.sqrt(3)/2), (spacing*1.5, -spacing*math.sqrt(3)/2),
        (-spacing*1.5, spacing*math.sqrt(3)/2), (-spacing*1.5, -spacing*math.sqrt(3)/2)
    ]
    
    for x, y in coords:
        state.append([x, y, 0.0])
        
    # Initial outer side length
    # Max extent of initial grid:
    # x range: -2.7 to 2.7 -> width 5.4
    # y range: -1.5 to 1.5 -> height 3.0
    # Hexagon width (flat-to-flat) for side S is S*sqrt(3).
    # Hexagon height (vertex-to-vertex) for side S is 2*S.
    # To contain width 5.4, S*sqrt(3) >= 5.4 => S >= 3.1
    # To contain height 3.0, 2*S >= 3.0 => S >= 1.5
    # So S ~ 3.5 is a good start.
    side = 4.0
    
    best_state = [h[:] for h in state]
    best_side = side
    best_energy = calculate_energy(state, side)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate a random move
        move_type = random.random()
        
        # Copy state for modification
        new_state = [h[:] for h in state]
        new_side = side
        
        if move_type < 0.5:
            # Move a random hexagon
            i = random.randint(0, N - 1)
            new_state[i][0] += random.gauss(0, 0.5)
            new_state[i][1] += random.gauss(0, 0.5)
        elif move_type < 0.7:
            # Rotate a random hexagon
            i = random.randint(0, N - 1)
            new_state[i][2] += random.gauss(0, 0.2)
        elif move_type < 0.85:
            # Adjust outer side length
            # Try to decrease side length if energy is low
            new_side = side + random.gauss(0, 0.1)
            if new_side < 1.0: new_side = 1.0
        else:
            # Perturb outer center/rotation? 
            # We fixed outer to (0,0,0) in is_inside_inner. 
            # To allow outer movement, we'd need to pass it to is_inside_inner.
            # For now, stick to fixed outer to simplify.
            pass

        new_energy = calculate_energy(new_state, new_side)
        
        # Acceptance criterion
        delta_e = new_energy - best_energy
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            state = new_state
            side = new_side
            best_energy = new_energy
            
            # Update best if no overlaps and contained
            if best_energy == 0:
                if side < best_side:
                    best_side = side
                    best_state = [h[:] for h in state]
        
        T *= cooling
        
    # Final polish: Try to reduce side further with smaller steps
    # Simple local search on side length
    current_side = best_side
    current_state = [h[:] for h in best_state]
    
    for _ in range(100):
        if calculate_energy(current_state, current_side) > 0:
            break
        # Try to shrink
        test_side = current_side * 0.999
        if test_side < 1.0: break
        if calculate_energy(current_state, test_side) == 0:
            current_side = test_side
        else:
            # Try to adjust positions slightly to allow shrinking?
            # For now, just stop if can't shrink.
            break

    return [ [h[0], h[1], math.degrees(h[2])] for h in current_state ], (0.0, 0.0), current_side, 0.0
# EVOLVE-BLOCK-END
