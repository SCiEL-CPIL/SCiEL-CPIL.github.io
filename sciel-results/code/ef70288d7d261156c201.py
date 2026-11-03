"""Packing 11 unit regular hexagons in a regular hexagon.

Strategy:
- Build a connected cluster topology of 11 hexagons (a 3-3-3-2 arrangement).
- Use scipy.optimize.minimize (BFGS) on global rotation + translation to minimize
  the outer hexagon side length required to contain the cluster.
- The objective computes the minimum side length of a regular hexagon (with given
  rotation) that contains all vertices of all inner hexagons.
- Multiple restarts with different topologies / initial rotations.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 150.0  # evaluator kills at 150 s; keep margin

def _hex_vertices(cx, cy, angle_rad):
    """Return the 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_rad."""
    # A regular hexagon with side length 1 has circumradius 1.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top orientation at angle=0).
    angles = np.array([0.0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]) + angle_rad
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_hex_side(points, outer_angle):
    """
    Given a set of 2D points, find the minimum side length s of a regular hexagon
    (with rotation outer_angle) that contains all points.
    
    A regular hexagon with side length s and rotation theta has support function:
    h(theta + k*pi/3) = s * cos(pi/6) = s * sqrt(3)/2  for k=0..5 (for flat-top)
    Actually, for a regular hexagon with circumradius R = s (side length s for unit hex),
    the support function in direction phi is R * max_k cos(phi - (theta + k*pi/3)).
    
    Wait: for a regular hexagon with side length s, the circumradius is s.
    The support function h(phi) = s * max_{k=0..5} cos(phi - theta - k*pi/3).
    
    To contain all points, we need for each point p and each direction:
    p . u <= h(u) for all unit vectors u.
    This is equivalent to: max_{p in points} p . u <= h(u) for all u.
    By convexity, it's enough to check the 6 directions of the hexagon normals.
    
    Actually, a point p is inside the hexagon centered at origin with rotation theta and side s iff:
    For each of the 6 normal directions n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3)):
    p . n_k <= s * cos(pi/6) = s * sqrt(3)/2
    
    So the minimum s is: max over all points p and all k of (p . n_k) / (sqrt(3)/2).
    But we also need to handle the negative side: |p . n_k| <= s * sqrt(3)/2.
    So s_min = max_{p, k} |p . n_k| / (sqrt(3)/2).
    """
    theta = outer_angle
    normals = []
    for k in range(6):
        a = theta + k * np.pi / 3
        normals.append(np.array([np.cos(a), np.sin(a)]))
    normals = np.array(normals)  # shape (6, 2)
    
    # points: (M, 2)
    # Compute all dot products: (M, 6)
    dots = points @ normals.T  # (M, 6)
    abs_dots = np.abs(dots)
    max_abs = np.max(abs_dots)
    s = max_abs / (np.sqrt(3.0) / 2.0)
    return s

def _build_cluster_topology(topo_type=0):
    """
    Build initial positions for 11 unit hexagons in a connected cluster.
    Returns list of (dx, dy) relative offsets from cluster center.
    
    Topology 0: 3-3-3-2 rows (flat-top hexagons, angle=0)
    For flat-top hexagons with side 1:
    - Horizontal distance between adjacent hex centers in same row: 2 (since width = 2)
    - Vertical distance between rows: sqrt(3)
    - Offset between rows: 1 (half of horizontal spacing)
    """
    positions = []
    if topo_type == 0:
        # 3-3-3-2 arrangement
        # Row 0: 3 hexes at y=0
        for i in range(3):
            positions.append((i * 2.0, 0.0))
        # Row 1: 3 hexes at y=sqrt(3), offset by 1
        for i in range(3):
            positions.append((i * 2.0 + 1.0, np.sqrt(3)))
        # Row 2: 3 hexes at y=2*sqrt(3)
        for i in range(3):
            positions.append((i * 2.0, 2.0 * np.sqrt(3)))
        # Row 3: 2 hexes at y=3*sqrt(3), offset by 1
        for i in range(2):
            positions.append((i * 2.0 + 1.0, 3.0 * np.sqrt(3)))
    elif topo_type == 1:
        # 2-3-3-3 arrangement (mirror)
        for i in range(2):
            positions.append((i * 2.0 + 1.0, 0.0))
        for i in range(3):
            positions.append((i * 2.0, np.sqrt(3)))
        for i in range(3):
            positions.append((i * 2.0 + 1.0, 2.0 * np.sqrt(3)))
        for i in range(3):
            positions.append((i * 2.0, 3.0 * np.sqrt(3)))
    elif topo_type == 2:
        # Compact 4-3-4 or similar
        # Let's try a different arrangement: 4-3-4
        for i in range(4):
            positions.append((i * 2.0, 0.0))
        for i in range(3):
            positions.append((i * 2.0 + 1.0, np.sqrt(3)))
        for i in range(4):
            positions.append((i * 2.0, 2.0 * np.sqrt(3)))
    else:
        # Fallback: linear
        for i in range(N):
            positions.append((i * 2.0, 0.0))
    
    # Center the cluster
    pos_arr = np.array(positions)
    centroid = np.mean(pos_arr, axis=0)
    pos_arr = pos_arr - centroid
    return pos_arr

def _objective(params, rel_positions, inner_angles):
    """
    params: [cx, cy, global_rotation]
    rel_positions: (N, 2) relative offsets
    inner_angles: (N,) individual hex angles (relative to global rotation)
    """
    cx, cy, rot = params
    total_angle = rot  # global rotation applied to all
    
    all_points = []
    for i in range(N):
        dx, dy = rel_positions[i]
        # Rotate the relative position by global rotation
        ca, sa = np.cos(total_angle), np.sin(total_angle)
        rx = dx * ca - dy * sa
        ry = dx * sa + dy * ca
        px = cx + rx
        py = cy + ry
        angle_i = inner_angles[i] + total_angle
        verts = _hex_vertices(px, py, angle_i)
        all_points.append(verts)
    
    all_points = np.vstack(all_points)  # (N*6, 2)
    
    # Try multiple outer hexagon rotations to find the best
    # The outer hexagon rotation is also a variable, but for speed we can
    # optimize it separately or include it in params.
    # For now, let's include outer rotation as a 4th parameter.
    # Actually, let's restructure: params = [cx, cy, inner_rot, outer_rot]
    # But inner_rot applies to the whole cluster.
    
    # For simplicity, let's just compute with outer_rot = 0 and let the optimizer
    # handle it via the objective. But that's not quite right.
    
    # Better: include outer rotation in params.
    # Let's redefine: params = [cx, cy, cluster_rot, outer_rot]
    # But we already unpacked. Let's fix this by making the objective take 4 params.
    
    # This is getting complicated. Let's simplify: fix outer_rot = 0 and
    # just optimize cx, cy, cluster_rot. Then post-optimize outer_rot.
    
    outer_rot = 0.0
    s = _min_hex_side(all_points, outer_rot)
    return s

def _objective4(params, rel_positions, inner_angles):
    """params = [cx, cy, cluster_rot, outer_rot]"""
    cx, cy, cluster_rot, outer_rot = params
    
    all_points = []
    ca, sa = np.cos(cluster_rot), np.sin(cluster_rot)
    for i in range(N):
        dx, dy = rel_positions[i]
        rx = dx * ca - dy * sa
        ry = dx * sa + dy * ca
        px = cx + rx
        py = cy + ry
        angle_i = inner_angles[i] + cluster_rot
        verts = _hex_vertices(px, py, angle_i)
        all_points.append(verts)
    
    all_points = np.vstack(all_points)
    s = _min_hex_side(all_points, outer_rot)
    return s

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Try different topologies and initial conditions
    topo_types = [0, 1, 2]
    initial_rots = [0.0, np.pi/6, np.pi/3, np.pi/4]
    
    for topo in topo_types:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        rel_pos = _build_cluster_topology(topo)
        # All inner hexagons have the same orientation (flat-top)
        inner_angles = np.zeros(N)
        
        for init_rot in initial_rots:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            x0 = np.array([0.0, 0.0, init_rot, 0.0])
            
            try:
                res = minimize(
                    _objective4,
                    x0,
                    args=(rel_pos, inner_angles),
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                
                if res.success or res.fun < best_score:
                    cx, cy, crot, orot = res.x
                    score = res.fun
                    if score < best_score:
                        best_score = score
                        # Build the result
                        ca, sa = np.cos(crot), np.sin(crot)
                        inner = []
                        for i in range(N):
                            dx, dy = rel_pos[i]
                            rx = dx * ca - dy * sa
                            ry = dx * sa + dy * ca
                            px = cx + rx
                            py = cy + ry
                            angle_deg = np.degrees(inner_angles[i] + crot)
                            inner.append([px, py, angle_deg])
                        best_result = (inner, (cx, cy), score, np.degrees(orot))
            except Exception:
                pass
    
    if best_result is None:
        # Fallback to the original
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

