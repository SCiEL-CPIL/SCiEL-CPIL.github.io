"""
Strategy: Rigid Hexagonal Cluster Optimization
Fixes the adjacency structure of the 11 hexagons as a specific hexagonal cluster
(1 central, 6 in a ring, 4 in the second ring).
Optimizes the global rotation angle of the inner cluster relative to the outer hexagon
to minimize the outer side length.
"""
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 120s limit

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

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    s = side
    # Half-plane method for hexagon with vertices at angles 0, 60, ...
    # In local frame (unrotated), vertices are at (s,0), (s/2, s*sqrt(3)/2), etc.
    # Inequalities:
    # 1. x <= s
    # 2. -x <= s  (x >= -s)
    # 3. sqrt(3)*y + x <= sqrt(3)*s  ? No.
    
    # Let's derive the 6 inequalities for a regular hexagon with side s, center (0,0), vertex at (s,0).
    # Vertices:
    # v0: (s, 0)
    # v1: (s/2, s*sqrt(3)/2)
    # v2: (-s/2, s*sqrt(3)/2)
    # v3: (-s, 0)
    # v4: (-s/2, -s*sqrt(3)/2)
    # v5: (s/2, -s*sqrt(3)/2)
    
    # Edge v0-v1: vector (-s/2, s*sqrt(3)/2). Normal inward: (s*sqrt(3)/2, s/2) -> (sqrt(3), 1).
    # Equation: sqrt(3)*x + y <= sqrt(3)*s
    # Edge v1-v2: vector (-s, 0). Normal inward: (0, -s) -> (0, -1).
    # Equation: -y <= -s*sqrt(3)/2  => y >= -s*sqrt(3)/2 ... wait.
    # Let's check v1: (s/2, s*sqrt(3)/2). -y = -s*sqrt(3)/2. LHS = -s*sqrt(3)/2. RHS = -s*sqrt(3)/2. OK.
    # Center (0,0): 0 <= -s*sqrt(3)/2? No. 0 is not <= negative.
    # Ah, the normal should point inward.
    # Edge v1->v2 is leftwards. Inward is downwards. Normal (0, -1).
    # Dot product with (p - v1) <= 0?
    # (0, -1) . (lx - s/2, ly - s*sqrt(3)/2) <= 0
    # -(ly - s*sqrt(3)/2) <= 0 => ly >= s*sqrt(3)/2.
    # This is the top edge.
    
    # Let's just use the standard form for a hexagon with flat top/bottom or pointy top.
    # Our vertices are pointy top/bottom? No, v0 is (s,0), so it's pointy left/right.
    # Width 2s, Height sqrt(3)s.
    
    # Inequalities for pointy left/right hexagon:
    # |x| <= s
    # |y| <= s * sqrt(3) / 2  is NOT sufficient.
    # The sides are slanted.
    # Top-Right edge: from (s,0) to (s/2, s*sqrt(3)/2).
    # Line: y - 0 = (s*sqrt(3)/2 - 0) / (s/2 - s) * (x - s)
    # y = (sqrt(3)/2) / (-1/2) * (x - s) = -sqrt(3) (x - s) = -sqrt(3)x + sqrt(3)s
    # So y <= -sqrt(3)x + sqrt(3)s  => sqrt(3)x + y <= sqrt(3)s
    
    # Top-Left edge: from (s/2, s*sqrt(3)/2) to (-s/2, s*sqrt(3)/2).
    # y = s*sqrt(3)/2
    
    # Bottom-Left edge: from (-s/2, s*sqrt(3)/2) to (-s, 0).
    # Symmetric to Top-Right: sqrt(3)(-x) + y <= sqrt(3)s ? No.
    # Line: y - 0 = (s*sqrt(3)/2 - 0) / (-s/2 - (-s)) * (x - (-s))
    # y = (sqrt(3)/2) / (1/2) * (x + s) = sqrt(3)(x+s) = sqrt(3)x + sqrt(3)s
    # So y <= sqrt(3)x + sqrt(3)s => -sqrt(3)x + y <= sqrt(3)s
    
    # Bottom-Right edge: from (-s,0) to (-s/2, -s*sqrt(3)/2).
    # y = -sqrt(3)(x+s) => -sqrt(3)x - y <= sqrt(3)s ?
    # Let's check (-s,0): -sqrt(3)(-s) - 0 = sqrt(3)s. OK.
    # Let's check (-s/2, -s*sqrt(3)/2): -sqrt(3)(-s/2) - (-s*sqrt(3)/2) = s*sqrt(3)/2 + s*sqrt(3)/2 = s*sqrt(3). OK.
    # So -sqrt(3)x - y <= sqrt(3)s => sqrt(3)x + y >= -sqrt(3)s
    
    # Bottom edge: y = -s*sqrt(3)/2 => -y <= -s*sqrt(3)/2 => y >= -s*sqrt(3)/2
    
    # Top edge: y = s*sqrt(3)/2 => y <= s*sqrt(3)/2
    
    # Left edge: x = -s => x >= -s
    # Right edge: x = s => x <= s
    
    # Summary of 6 inequalities:
    # 1. x <= s
    # 2. x >= -s
    # 3. y <= s * sqrt(3) / 2
    # 4. y >= -s * sqrt(3) / 2
    # 5. sqrt(3)*x + y <= s * sqrt(3)
    # 6. -sqrt(3)*x + y <= s * sqrt(3)  (Top-Left)
    # 7. sqrt(3)*x - y <= s * sqrt(3)  (Bottom-Left? No, Bottom-Right was -sqrt(3)x - y <= sqrt(3)s => sqrt(3)x + y >= -sqrt(3)s)
    
    # Let's re-verify the slanted ones.
    # TR: sqrt(3)x + y <= s*sqrt(3)
    # TL: -sqrt(3)x + y <= s*sqrt(3)
    # BL: -sqrt(3)x - y <= s*sqrt(3)  (Check: (-s,0) -> s*sqrt(3) <= s*sqrt(3). OK. (-s/2, -s*sqrt(3)/2) -> s*sqrt(3)/2 + s*sqrt(3)/2 = s*sqrt(3). OK.)
    # BR: sqrt(3)x - y <= s*sqrt(3)   (Check: (s,0) -> s*sqrt(3). OK. (s/2, -s*sqrt(3)/2) -> s*sqrt(3)/2 + s*sqrt(3)/2 = s*sqrt(3). OK.)
    
    sqrt3 = np.sqrt(3)
    
    if lx > s + 1e-9: return False
    if lx < -s - 1e-9: return False
    if ly > s * sqrt3 / 2 + 1e-9: return False
    if ly < -s * sqrt3 / 2 - 1e-9: return False
    if sqrt3 * lx + ly > s * sqrt3 + 1e-9: return False
    if -sqrt3 * lx + ly > s * sqrt3 + 1e-9: return False
    if -sqrt3 * lx - ly > s * sqrt3 + 1e-9: return False
    if sqrt3 * lx - ly > s * sqrt3 + 1e-9: return False
    
    return True

def _min_outer_side_for_cluster(cluster_positions, cluster_angles, outer_center, outer_angle):
    """
    Calculate the minimum outer side length required to contain the cluster.
    cluster_positions: list of (x, y) relative to cluster center.
    cluster_angles: list of angles for each hexagon.
    """
    cx, cy = outer_center
    all_verts = []
    for i in range(len(cluster_positions)):
        px, py = cluster_positions[i]
        ang = cluster_angles[i]
        verts = _hex_vertices(px, py, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Binary search on side length
    lo, hi = 1.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        ok = True
        for v in all_verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, mid, outer_angle):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    global DEADLINE
    
    # Define the rigid cluster of 11 hexagons
    # Center: (0,0)
    # Ring 1 (6 hexagons):
    #   Angles: 0, 60, 120, 180, 240, 300
    #   Distance from center: 2 * side * cos(30) = 2 * 1 * sqrt(3)/2 = sqrt(3)
    #   Wait, for hexagons of side 1, the distance between centers of adjacent hexagons in a lattice is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    #   Yes, distance is sqrt(3).
    
    # Ring 2 (4 hexagons):
    #   We need 4 more. The full second ring has 12.
    #   To minimize the bounding box, we should pick 4 that are symmetric.
    #   The "corners" of the hexagonal cluster are at angles 30, 90, 150, 210, 270, 330?
    #   Let's check the lattice coordinates.
    #   Center (0,0).
    #   Ring 1 centers:
    #   (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    #   Note: 1.5 = 3/2. sqrt(3)/2 * 2 = sqrt(3).
    #   Distance check: (sqrt(3)/2)^2 + (1.5)^2 = 3/4 + 9/4 = 12/4 = 3. dist = sqrt(3). Correct.
    
    #   Ring 2 candidates (distance 2*sqrt(3) from center? No, next shell).
    #   In hex lattice, next shell distance is 2*sqrt(3).
    #   The 12 positions in Ring 2 are:
    #   (2*sqrt(3), 0), (3*sqrt(3)/2, 3), (sqrt(3), 3), (0, 3), (-sqrt(3), 3), (-3*sqrt(3)/2, 3), ...
    #   Let's verify (2*sqrt(3), 0) is adjacent to (sqrt(3), 0)?
    #   Dist: sqrt(3). Yes.
    #   (sqrt(3), 3) adjacent to (sqrt(3)/2, 1.5)?
    #   dx = sqrt(3)/2, dy = 1.5. dist^2 = 3/4 + 9/4 = 3. Yes.
    #   (0, 3) adjacent to (sqrt(3)/2, 1.5)?
    #   dx = sqrt(3)/2, dy = 1.5. dist^2 = 3. Yes.
    
    #   So Ring 2 positions include:
    #   P1: (2*sqrt(3), 0)
    #   P2: (3*sqrt(3)/2, 3)
    #   P3: (sqrt(3), 3)
    #   P4: (0, 3)
    #   P5: (-sqrt(3), 3)
    #   P6: (-3*sqrt(3)/2, 3)
    #   P7: (-2*sqrt(3), 0)
    #   ... and bottom ones.
    
    #   We need to pick 4. To maintain symmetry and minimize bounding box,
    #   picking the 4 "cardinal" directions of the hexagon (0, 120, 240, 360? No, hex has 6-fold symmetry).
    #   If we pick P1 (0 deg), P4 (90 deg? No, angle is atan2(3,0)=90), P7 (180 deg), P10 (270 deg).
    #   Let's check angles.
    #   P1: (2*sqrt(3), 0) -> 0 deg
    #   P4: (0, 3) -> 90 deg
    #   P7: (-2*sqrt(3), 0) -> 180 deg
    #   P10: (0, -3) -> 270 deg
    #   This forms a cross shape.
    #   Is this valid?
    #   Center (0,0).
    #   Ring 1: 6 hexagons.
    #   Ring 2: 4 hexagons at 0, 90, 180, 270 degrees (in terms of position angle).
    #   Total 11.
    
    #   Let's define the positions.
    sqrt3 = np.sqrt(3)
    
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1
    for i in range(6):
        theta = i * np.pi / 3
        x = sqrt3 * np.cos(theta)
        y = sqrt3 * np.sin(theta)
        positions.append([x, y])
        angles.append(0.0) # Hexagons are aligned with the lattice (0 deg rotation)
        
    # Ring 2 (4 positions: 0, 90, 180, 270 degrees)
    # Note: The lattice is not square, so "90 degrees" in Cartesian is not a lattice direction.
    # But we can place hexagons anywhere as long as they don't overlap.
    # However, if we place them at Cartesian 90 degrees, they might not touch Ring 1 optimally or might overlap.
    # Let's check overlap for P4 (0, 3).
    # Nearest Ring 1 neighbors: (sqrt(3)/2, 1.5) and (-sqrt(3)/2, 1.5).
    # Dist from (0,3) to (sqrt(3)/2, 1.5):
    # dx = sqrt(3)/2, dy = 1.5. dist = sqrt(3/4 + 9/4) = sqrt(3) = 1.732.
    # Required distance for non-overlap is sqrt(3).
    # So they touch. This is a valid packing.
    
    # So the 4 positions are:
    # (2*sqrt(3), 0)
    # (0, 3)
    # (-2*sqrt(3), 0)
    # (0, -3)
    
    positions.append([2*sqrt3, 0.0])
    angles.append(0.0)
    
    positions.append([0.0, 3.0])
    angles.append(0.0)
    
    positions.append([-2*sqrt3, 0.0])
    angles.append(0.0)
    
    positions.append([0.0, -3.0])
    angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    # The cluster is centered at (0,0).
    # We will rotate the entire cluster by angle `theta` and place it in the outer hexagon.
    # The outer hexagon is centered at (0,0) and has angle 0 (aligned with axes).
    # We can also rotate the outer hexagon, but since the problem is symmetric,
    # fixing outer angle to 0 and rotating the inner cluster is equivalent.
    
    # Objective: Minimize outer side length S.
    # For a given rotation theta of the cluster, we calculate the required S.
    
    def cost(theta):
        if time.time() > DEADLINE:
            return np.inf
        
        # Rotate positions
        ca = np.cos(theta)
        sa = np.sin(theta)
        
        rotated_pos = np.zeros_like(positions)
        rotated_pos[:, 0] = positions[:, 0] * ca - positions[:, 1] * sa
        rotated_pos[:, 1] = positions[:, 0] * sa + positions[:, 1] * ca
        
        # Angles of hexagons also rotate
        rotated_ang = angles + np.degrees(theta)
        
        # Calculate min outer side
        # Outer center is (0,0), outer angle is 0.
        s = _min_outer_side_for_cluster(rotated_pos, rotated_ang, (0.0, 0.0), 0.0)
        return s
    
    # Optimize theta in [0, pi/3] due to symmetry
    # Use minimize_scalar
    
    best_theta = 0.0
    best_side = np.inf
    
    try:
        res = minimize_scalar(cost, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-4})
        if res.success:
            best_theta = res.x
            best_side = res.fun
    except Exception:
        pass
    
    # If optimization failed or didn't improve, try a few discrete angles
    if best_side == np.inf or not np.isfinite(best_side):
        for t in np.linspace(0, np.pi/3, 20):
            s = cost(t)
            if s < best_side:
                best_side = s
                best_theta = t
    
    # Construct final positions
    ca = np.cos(best_theta)
    sa = np.sin(best_theta)
    
    final_pos = np.zeros_like(positions)
    final_pos[:, 0] = positions[:, 0] * ca - positions[:, 1] * sa
    final_pos[:, 1] = positions[:, 0] * sa + positions[:, 1] * ca
    
    final_ang = angles + np.degrees(best_theta)
    
    # Verify validity
    # Check overlaps
    verts_list = []
    for i in range(N):
        verts = _hex_vertices(final_pos[i][0], final_pos[i][1], 1.0, final_ang[i])
        verts_list.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            # SAT check
            axes = []
            for verts in [verts_list[i], verts_list[j]]:
                for k in range(6):
                    v1 = verts[k]
                    v2 = verts[(k+1)%6]
                    edge = v2 - v1
                    nx, ny = edge[1], -edge[0]
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm > 1e-12:
                        axes.append([nx/norm, ny/norm])
            overlap = False
            for axis in axes:
                proj1 = np.dot(verts_list[i], axis)
                proj2 = np.dot(verts_list[j], axis)
                if proj1.max() < proj2.min() - 1e-7 or proj2.max() < proj1.min() - 1e-7:
                    overlap = True
                    break
            if not overlap:
                valid = False
                break
        if not valid:
            break
    
    # Check containment
    if valid:
        outer_verts = _hex_vertices(0.0, 0.0, best_side, 0.0)
        for verts in verts_list:
            for v in verts:
                if not _point_in_hexagon(v[0], v[1], 0.0, 0.0, best_side, 0.0):
                    valid = False
                    break
            if not valid:
                break
    
    # If invalid, fallback to a known valid packing (e.g. line)
    if not valid:
        # Fallback: 11 hexagons in a row
        # Side length approx 11 * 2 = 22?
        # Positions: (2*i, 0)
        fallback_pos = np.array([[2.0 * i, 0.0] for i in range(-5, 6)])
        fallback_ang = np.zeros(11)
        # Min side for this
        # Extent x: -10 to 10. y: -sqrt(3)/2 to sqrt(3)/2.
        # For hexagon with side S, width 2S.
        # We need 2S >= 20 + 2*1? No.
        # Let's just use a large safe value.
        return [[float(fallback_pos[i][0]), float(fallback_pos[i][1]), float(fallback_ang[i])] for i in range(N)], (0.0, 0.0), 25.0, 0.0
    
    inner_list = [[float(final_pos[i][0]), float(final_pos[i][1]), float(final_ang[i])] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
