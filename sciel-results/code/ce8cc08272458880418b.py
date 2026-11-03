"""Topological graph optimization for hexagon packing."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill
SQRT3 = np.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_vertices_batch(positions, angles, side=1.0):
    """Vectorized: return (N*6, 2) array of all vertices for N hexagons."""
    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    offsets_x = side * np.cos(total_angles)
    offsets_y = side * np.sin(total_angles)
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    return np.column_stack([verts_x.ravel(), verts_y.ravel()])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s = side
    if lx > s + 1e-10 or lx < -s - 1e-10:
        return False
    if SQRT3 * lx + ly > SQRT3 * s + 1e-10:
        return False
    if -SQRT3 * lx + ly > SQRT3 * s + 1e-10:
        return False
    if -SQRT3 * lx - ly > SQRT3 * s + 1e-10:
        return False
    if SQRT3 * lx - ly > SQRT3 * s + 1e-10:
        return False
    return True

def _min_side_vectorized(all_verts, cx, cy, outer_angle_deg):
    """
    Compute minimum outer side length S such that all vertices in all_verts are inside
    the outer hexagon centered at (cx,cy) with rotation outer_angle_deg.
    """
    verts = np.asarray(all_verts, dtype=np.float64)
    n_pts = len(verts)
    if n_pts == 0:
        return 0.0
    
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / SQRT3
    c4 = -lx + ly / SQRT3
    c5 = -lx - ly / SQRT3
    c6 = lx - ly / SQRT3
    
    required = np.maximum(
        np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)),
        np.maximum(c5, c6)
    )
    return float(np.max(required))

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _check_no_overlap_fast(positions, angles, tol=1e-7):
    """Check that no two inner hexagons overlap. Returns True if valid."""
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol):
                return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    """Find minimum outer side length such that all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.asarray(inner_angles, dtype=np.float64)
    all_verts = _hex_vertices_batch(pos, ang, side=1.0)
    return _min_side_vectorized(all_verts, cx, cy, outer_angle)

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _hexagon_overlap(verts_list[i], verts_list[j]):
                return False
    
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _get_positions_from_params(r1, r2, theta):
    """
    Generate positions for the 11 hexagons based on topological cluster parameters.
    r1: radius of first ring (6 hexagons)
    r2: radius of second ring (4 hexagons)
    theta: global rotation angle of the cluster
    """
    positions = []
    angles = []
    
    # Center hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # First ring: 6 hexagons
    for i in range(6):
        a = theta + i * np.pi / 3
        x = r1 * np.cos(a)
        y = r1 * np.sin(a)
        positions.append([x, y])
        angles.append(0.0)
    
    # Second ring: 4 hexagons (at 0, 90, 180, 270 relative to cluster axis? No, hexagonal symmetry suggests 0, 60, 120, 180, 240, 300)
    # Wait, the standard hexagonal packing of 19 hexagons has rings of 1, 6, 12.
    # We have 11 hexagons: 1 center, 6 ring 1, 4 ring 2.
    # To maintain symmetry and minimize bounding box, the 4 ring 2 hexagons should be placed
    # to fill the "gaps" or corners. In a hex grid, the second ring neighbors are at distance 2*r1.
    # The angles for the 4 hexagons in the second ring of a 11-hex cluster (which is a subset of the 19-hex cluster)
    # are typically 0, 60, 120, 180, 240, 300. We pick 4 of them. To be symmetric, we might pick 0, 90, 180, 270? No, that breaks hex symmetry.
    # The 19-hex cluster has 12 in the second ring. We have 4.
    # A symmetric choice for 4 hexagons in the second ring of a hexagonal cluster is to place them at 0, 90, 180, 270 degrees? No, that's square symmetry.
    # Hexagonal symmetry group is D6. A subset of 4 points invariant under D6? Impossible.
    # However, we can rotate the whole cluster. The "shape" of the 4 points matters.
    # Let's assume the 4 points are at angles 0, 60, 120, 180 (4 consecutive) or 0, 90, 180, 270 (square)?
    # Actually, for minimal bounding box, we want the points to be as "circular" as possible.
    # Let's try placing them at 0, 90, 180, 270 relative to the cluster rotation.
    # Or better: The 11-hex cluster is often formed by 1 center, 6 neighbors, and 4 of the 12 second-neighbors.
    # To keep it "round", we should pick every 3rd one? 0, 120, 240? That's 3. We need 4.
    # Let's pick 0, 60, 120, 180.
    # Actually, let's allow the angles of the second ring to be optimized too?
    # The prompt says: "fixing the adjacency structure... and solving for the minimal outer side length by optimizing the global rotation angle... rather than optimizing individual positions."
    # This implies the relative positions are fixed by the topology.
    # Topology: 1 center, 6 neighbors, 4 second neighbors.
    # In a regular hex grid, the second neighbors are at distance 2*sqrt(3) (if r1=2*sqrt(3)? No, r1 is center-to-center distance).
    # For unit hexagons, center-to-center distance is 2*sqrt(3) for non-overlapping neighbors?
    # No, for unit side length s, distance between centers of touching hexagons is 2*s*sqrt(3)/2 * 2?
    # Distance between centers of two touching unit hexagons is 2 * sqrt(3) * (1/2) * 2?
    # Height of hexagon is sqrt(3). Distance between parallel sides is sqrt(3).
    # Distance between centers is 2 * (sqrt(3)/2) * 2? No.
    # Let's recall: For regular hexagons of side 1, the distance between centers of two adjacent hexagons is 2 * sqrt(3) * (1/2) ?
    # The apothem is sqrt(3)/2. Distance between centers is 2 * apothem = sqrt(3).
    # Wait. Vertices are at distance 1 from center.
    # Distance between centers of two hexagons sharing a side:
    # The side length is 1. The distance from center to side is sqrt(3)/2.
    # So distance between centers is 2 * (sqrt(3)/2) = sqrt(3).
    # So r1 should be at least sqrt(3) ~ 1.732.
    # If r1 = sqrt(3), they touch.
    # The second ring hexagons in a hex grid are at distance 2*sqrt(3) from center?
    # No, in a hex grid, the neighbors of neighbors are at various distances.
    # The "second ring" in the 19-hex cluster (center + 6 + 12) has 12 hexagons.
    # 6 of them are at distance 2*sqrt(3) (directly opposite the first ring gaps? No).
    # Let's visualize.
    # Center (0,0).
    # Ring 1: 6 hexagons at distance sqrt(3), angles 0, 60, ...
    # Ring 2: 12 hexagons.
    # 6 of them are at distance 2*sqrt(3), angles 0, 60, ... (aligned with ring 1).
    # 6 of them are at distance sqrt(3)*sqrt(3) = 3? No.
    # The other 6 are at distance sqrt( (sqrt(3))^2 + (sqrt(3))^2 )?
    # Let's use coordinates.
    # Hex 1 center: (sqrt(3), 0).
    # Hex 2 center: (sqrt(3)/2, 1.5).
    # Vector to Hex 1: (sqrt(3), 0).
    # Vector to Hex 2: (sqrt(3)/2, 1.5).
    # Sum: (3*sqrt(3)/2, 1.5). Distance: sqrt(27/4 + 2.25) = sqrt(6.75 + 2.25) = sqrt(9) = 3.
    # So the "corner" second neighbors are at distance 3.
    # The "side" second neighbors (aligned with ring 1) are at distance 2*sqrt(3) ~ 3.464.
    # So the 12 second neighbors are 6 at distance 3 and 6 at distance 2*sqrt(3).
    # We have 4 hexagons to place in the second ring.
    # To minimize the bounding box, we should probably pick the 4 that are most "compact".
    # Or we can define the topology such that the 4 are at specific angles.
    # Let's assume the 4 second-ring hexagons are at angles 0, 90, 180, 270 relative to the cluster rotation?
    # This would create a cross shape.
    # Or 0, 60, 120, 180?
    # Let's optimize the angles of the second ring as well?
    # The directive says: "optimizing the global rotation angle... rather than optimizing individual positions."
    # This suggests the relative positions are FIXED by the topology.
    # I will fix the second ring hexagons at angles 0, 60, 120, 180 (4 consecutive) and distance 3 (the "corner" ones) or 2*sqrt(3) (the "side" ones)?
    # Actually, a common 11-hex packing is the center, 6 neighbors, and 4 of the 12 second neighbors.
    # To be symmetric, we might pick 0, 90, 180, 270? No, that's not hex symmetric.
    # Let's pick 0, 60, 120, 180.
    # And distance? If we pick the "side" ones (distance 2*sqrt(3)), they are aligned with the first ring.
    # If we pick the "corner" ones (distance 3), they are between the first ring hexagons.
    # Let's try placing the 4 second-ring hexagons at angles 0, 90, 180, 270 and distance 3?
    # No, let's stick to the hex grid topology.
    # I will define the 4 second-ring hexagons to be at angles 0, 60, 120, 180 and distance 2*sqrt(3).
    # Wait, if I rotate the cluster, the angles change.
    # So the parameters are: r1 (fixed to sqrt(3) for touching?), r2 (fixed to 2*sqrt(3)?), theta (rotation).
    # But if I fix r1 and r2, I only optimize theta. That's a 1D optimization.
    # Maybe I should allow r1 and r2 to vary slightly to allow "breathing" if it helps?
    # But they must not overlap.
    # Let's optimize r1, r2, and theta.
    # Constraints:
    # 1. No overlap between any pair.
    # 2. r1 >= sqrt(3) (center to ring 1).
    # 3. r2 >= 2*sqrt(3) (center to ring 2)? No, distance between ring 1 and ring 2 also matters.
    # Distance between a ring 1 hex and a ring 2 hex:
    # If ring 2 is at angle 0 and ring 1 is at angle 0, distance is |r2 - r1|.
    # If ring 2 is at angle 60 and ring 1 is at angle 0, distance is sqrt(r1^2 + r2^2 - 2*r1*r2*cos(60)).
    # We need these distances to be at least sqrt(3).
    
    # Let's define the objective function.
    
    positions = []
    angles = [0.0] * N
    
    # Center
    positions.append([0.0, 0.0])
    
    # Ring 1
    for i in range(6):
        a = theta + i * np.pi / 3
        positions.append([r1 * np.cos(a), r1 * np.sin(a)])
    
    # Ring 2: 4 hexagons. Let's place them at 0, 90, 180, 270 relative to theta?
    # No, let's place them at 0, 60, 120, 180 relative to theta.
    # And distance r2.
    for i in range(4):
        a = theta + i * np.pi / 3
        positions.append([r2 * np.cos(a), r2 * np.sin(a)])
        
    positions = np.array(positions)
    return positions, angles

def _objective(params):
    """
    Objective function for differential evolution.
    params: [r1, r2, theta]
    """
    r1, r2, theta = params
    
    # Bounds check
    if r1 < 1.732 or r2 < 3.0: # sqrt(3) ~ 1.732, 2*sqrt(3) ~ 3.464
        return 100.0
        
    positions, angles = _get_positions_from_params(r1, r2, theta)
    
    # Check overlaps
    if not _check_no_overlap_fast(positions, angles):
        return 100.0
        
    # Calculate outer side length
    side = _min_outer_side(positions, angles, (0.0, 0.0), 0.0)
    
    return side

def construct():
    global START_TIME, DEADLINE
    
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Initial guess
    r1_init = np.sqrt(3)
    r2_init = 2 * np.sqrt(3)
    theta_init = 0.0
    
    # Define bounds
    # r1: min sqrt(3), max 5
    # r2: min 3.0, max 6
    # theta: 0 to 60 degrees (due to symmetry, 0-60 is enough)
    bounds = [(np.sqrt(3), 5.0), (3.0, 6.0), (0.0, np.pi/3)]
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-5,
            polish=True
        )
        
        if result.success:
            r1_opt, r2_opt, theta_opt = result.x
            positions, angles = _get_positions_from_params(r1_opt, r2_opt, theta_opt)
            best_side = _min_outer_side(positions, angles, (outer_cx, outer_cy), outer_angle)
        else:
            # Fallback to initial
            positions, angles = _get_positions_from_params(r1_init, r2_init, theta_init)
            best_side = _min_outer_side(positions, angles, (outer_cx, outer_cy), outer_angle)
            
    except Exception:
        positions, angles = _get_positions_from_params(r1_init, r2_init, theta_init)
        best_side = _min_outer_side(positions, angles, (outer_cx, outer_cy), outer_angle)
        
    # Final verification
    inner_list = [[float(positions[i][0]), float(positions[i][1]), float(angles[i])] for i in range(N)]
    
    # Recompute the exact side length
    best_side = _min_outer_side(positions, angles, (outer_cx, outer_cy), outer_angle)
    
    if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
        # Fallback: use initial packing
        pos_init, ang_init = _get_positions_from_params(np.sqrt(3), 2*np.sqrt(3), 0.0)
        inner_list = [[float(pos_init[i][0]), float(pos_init[i][1]), float(ang_init[i])] for i in range(N)]
        best_side = _min_outer_side(pos_init, ang_init, (outer_cx, outer_cy), outer_angle)
        
        if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
            # Last resort: spread them out
            inner_list = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner_list, (0.0, 0.0), 40.0, 0.0
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(best_side), float(outer_angle)
