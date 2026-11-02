"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def _hex_apothem(side):
    return side * math.sqrt(3) / 2

def _calc_enclosing_hex_side(positions, center, angle_deg):
    """Calculate the side length of the regular hexagon centered at `center`
    with rotation `angle_deg` that encloses all `positions` (centers).
    Returns the side length. Assumes positions are centers of unit hexagons.
    The actual required side must account for the radius of the inner hexagons."""
    theta = math.radians(angle_deg)
    max_proj = 0
    # 6 outward normals
    for k in range(6):
        alpha = theta + k * math.pi / 3
        nx = math.cos(alpha)
        ny = math.sin(alpha)
        for px, py in positions:
            # Project relative to center
            dx = px - center[0]
            dy = py - center[1]
            proj = dx * nx + dy * ny
            if proj > max_proj:
                max_proj = proj
        
    # Apothem of outer hexagon must be at least max_proj + 1 (radius of inner hex)
    apothem = max_proj + 1.0
    side = 2 * apothem / math.sqrt(3)
    return side

def _is_disjoint(positions, min_dist):
    """Check if all pairs of hexagons are at least `min_dist` apart."""
    n = len(positions)
    for i in range(n):
        xi, yi = positions[i]
        for j in range(i + 1, n):
            xj, yj = positions[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            if dist_sq < min_dist * min_dist - 1e-9:
                return False
    return True

def _init_hex_grid(n, spacing):
    """Generate initial hex grid positions."""
    positions = []
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1
    for k in range(6):
        angle = k * math.pi / 3
        # Axial to cartesian for flat-top hex grid
        # q = cos(angle), r = sin(angle) is not quite right for axial.
        # Let's use the standard axial coords for ring 1:
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        coords = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        q, r = coords[k]
        x = spacing * (q + r / 2.0)
        y = spacing * (math.sqrt(3) / 2.0) * r
        positions.append((x, y))
        
    # Ring 2 (partial)
    # We need 4 more.
    # Axial coords for ring 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Pick 4 that are compact.
    # (2,0), (0,2), (-2,0), (0,-2) are far.
    # (1,1), (-1,1), (-1,-1), (1,-1) are closer?
    # Distance from center for (1,1): x = s(1+0.5)=1.5s, y = s(sqrt(3)/2). dist = s * sqrt(2.25 + 0.75) = s*sqrt(3).
    # Distance for (2,0): x=2s, y=0. dist=2s.
    # So (1,1) etc are closer.
    ring2_coords = [(1,1), (-1,1), (-1,-1), (1,-1)]
    for q, r in ring2_coords:
        x = spacing * (q + r / 2.0)
        y = spacing * (math.sqrt(3) / 2.0) * r
        positions.append((x, y))
        
    return positions[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Minimum distance for non-overlapping unit hexagons
    min_dist = math.sqrt(3)
    # Add a small margin for safety in floating point
    min_dist_safe = min_dist * 1.01
    
    # Initial positions
    positions = _init_hex_grid(N, min_dist_safe)
    
    # Calculate initial center and best side
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    center = (cx, cy)
    
    # We will alternate between optimizing positions and outer rotation
    best_side = float('inf')
    best_positions = [p[:] for p in positions]
    best_center = center
    best_angle = 0.0
    
    # Try a few initial angles for the outer hexagon
    candidate_angles = [0.0, 15.0, 30.0, 45.0]
    
    # Local Search
    # We'll do a simple random perturbation search
    # Since N is small, we can try many iterations
    
    # Pre-calculate some random steps
    # We will try to move one hexagon at a time
    
    iterations = 0
    max_iterations = 5000 # Adjust based on time
    
    while time.time() < deadline:
        iterations += 1
        
        # 1. Optimize Outer Rotation for current positions
        # Try a few angles around the current best
        # Or just scan 0-60 in small steps
        # To save time, scan 0-60 in 5 degree steps, then refine
        current_best_angle = 0
        current_min_side = float('inf')
        
        # Coarse scan
        for a_deg in range(0, 60, 5):
            side = _calc_enclosing_hex_side(positions, center, a_deg)
            if side < current_min_side:
                current_min_side = side
                current_best_angle = a_deg
        
        # Fine scan around best
        for a_deg_10 in range(int(current_best_angle*10) - 10, int(current_best_angle*10) + 10):
            a_deg = a_deg_10 / 10.0
            if a_deg < 0 or a_deg >= 60: continue
            side = _calc_enclosing_hex_side(positions, center, a_deg)
            if side < current_min_side:
                current_min_side = side
                current_best_angle = a_deg
        
        if current_min_side < best_side:
            best_side = current_min_side
            best_angle = current_best_angle
            best_positions = [p[:] for p in positions]
            # Update center to centroid of best_positions? 
            # The enclosing hexagon is centered at `center`. 
            # Should we move `center` to the centroid of `positions`?
            # Yes, for a given set of points, the optimal center for a symmetric shape like a hexagon
            # is often near the centroid, but not necessarily.
            # However, moving the center changes the projections.
            # Let's assume center = centroid for now, as optimizing center is another dimension.
            # To keep it simple and fast, we fix center = centroid(positions).
            cx = sum(p[0] for p in positions) / N
            cy = sum(p[1] for p in positions) / N
            center = (cx, cy)
            # Recalculate side with new center? 
            # _calc_enclosing_hex_side uses the center passed to it.
            # So we should pass the new center.
            # Let's re-evaluate the best angle with the new center.
            # Actually, the angle optimization above used the OLD center.
            # This is an approximation.
            # For a more robust search, we should update center and re-optimize angle.
            # But this might oscillate.
            # Let's just update best_side if it's better, using the side calculated with the center at that time.
            # Wait, if I change center, the side changes.
            # So `current_min_side` was calculated with the old center.
            # I should recalculate the side with the new center and the best angle.
            side_with_new_center = _calc_enclosing_hex_side(positions, center, current_best_angle)
            if side_with_new_center < best_side:
                best_side = side_with_new_center
                best_angle = current_best_angle
                best_positions = [p[:] for p in positions]
                best_center = center
        
        # 2. Optimize Positions
        # Try to move one random hexagon
        if N > 0:
            idx = random.randint(0, N - 1)
            orig_x, orig_y = positions[idx]
            
            # Random move
            step_size = 0.2
            new_x = orig_x + random.uniform(-step_size, step_size)
            new_y = orig_y + random.uniform(-step_size, step_size)
            
            # Check bounds? No explicit bounds, but we can limit to reasonable area
            # e.g., within 10 units of center
            
            positions[idx] = (new_x, new_y)
            
            if _is_disjoint(positions, min_dist_safe):
                # Calculate new side
                # Update center
                cx = sum(p[0] for p in positions) / N
                cy = sum(p[1] for p in positions) / N
                new_center = (cx, cy)
                
                # Use current best angle as a quick check
                side = _calc_enclosing_hex_side(positions, new_center, best_angle)
                
                # If this looks promising (better than current best or close), do a full angle optimization
                if side < best_side * 1.05: # 5% margin
                    # Full angle optimization
                    current_best_angle = best_angle
                    current_min_side = float('inf')
                    
                    # Coarse scan
                    for a_deg in range(0, 60, 5):
                        s = _calc_enclosing_hex_side(positions, new_center, a_deg)
                        if s < current_min_side:
                            current_min_side = s
                            current_best_angle = a_deg
                            
                    # Fine scan
                    for a_deg_10 in range(int(current_best_angle*10) - 10, int(current_best_angle*10) + 10):
                        a_deg = a_deg_10 / 10.0
                        if a_deg < 0 or a_deg >= 60: continue
                        s = _calc_enclosing_hex_side(positions, new_center, a_deg)
                        if s < current_min_side:
                            current_min_side = s
                            current_best_angle = a_deg
                            
                    if current_min_side < best_side:
                        best_side = current_min_side
                        best_angle = current_best_angle
                        best_positions = [p[:] for p in positions]
                        best_center = new_center
                    else:
                        # Revert
                        positions[idx] = (orig_x, orig_y)
                else:
                    # Revert
                    positions[idx] = (orig_x, orig_y)
            else:
                # Revert
                positions[idx] = (orig_x, orig_y)
                
    # Final check
    if best_side == float('inf'):
        # Fallback
        return [[p[0], p[1], 0.0] for p in positions], (0.0, 0.0), 10.0, 0.0

    inner = [[p[0], p[1], 0.0] for p in best_positions]
    return inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
