"""
Strategy: Rigid-Body Rotation Optimization with Lattice Packing.

The key insight is that for 11 unit hexagons, the most compact arrangement
is a hexagonal lattice cluster (1 center + 6 ring-1 + 4 ring-2).
Rather than optimizing all 33 degrees of freedom (positions + angles),
we treat the cluster as a rigid body with fixed internal geometry,
and optimize only the global rotation angle relative to the outer hexagon.

This reduces the problem from ~33D to 1D (plus small translation adjustments),
making it far more tractable and allowing us to find the optimal orientation
that minimizes the enclosing hexagon side length.
"""

import time
import numpy as np
import math

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin for 150s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_overlap(pos, ang, tol=1e-6):
    """
    Check if any two hexagons overlap using SAT (Separating Axis Theorem).
    Returns True if overlap exists.
    """
    verts_list = []
    for i in range(N):
        verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        verts_list.append(verts)
    
    bboxes = []
    for v in verts_list:
        min_x, min_y = np.min(v, axis=0)
        max_x, max_y = np.max(v, axis=0)
        bboxes.append((min_x, min_y, max_x, max_y))
    
    for i in range(N):
        for j in range(i+1, N):
            if (bboxes[i][0] > bboxes[j][2] or bboxes[i][2] < bboxes[j][0] or
                bboxes[i][1] > bboxes[j][3] or bboxes[i][3] < bboxes[j][1]):
                continue
            
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            axes = []
            for k in range(6):
                edge1 = v1[(k+1)%6] - v1[k]
                normal = np.array([-edge1[1], edge1[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
                
                edge2 = v2[(k+1)%6] - v2[k]
                normal = np.array([-edge2[1], edge2[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
            
            overlap = True
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                    overlap = False
                    break
            
            if overlap:
                return True
    return False

def _create_lattice_cluster():
    """
    Create a compact hexagonal lattice cluster of 11 unit hexagons.
    
    Structure:
    - Node 0: Center at (0, 0)
    - Nodes 1-6: Ring 1 at distance sqrt(3) * 2 = 2*sqrt(3)
    - Nodes 7-10: Ring 2 at distance 6 (4 hexagons at specific angles)
    
    For unit hexagons on a hexagonal lattice:
    - Center-to-center distance in ring 1: 2*sqrt(3) ≈ 3.464
    - Ring 2 positions: at distance 6 from center
    """
    sqrt3 = math.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # Node 0: Center
    pos[0] = [0.0, 0.0]
    ang[0] = 0.0
    
    # Nodes 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        ang[i+1] = 0.0
    
    # Nodes 7-10: Ring 2 (4 hexagons)
    # These are placed at the "gaps" between ring 1 hexagons
    # In a hexagonal lattice, ring 2 hexagons are at distance 6 from center
    # and at angles offset by 30 degrees from ring 1
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        ang[i+7] = 0.0
    
    return pos, ang

def _try_different_ring2_config():
    """
    Try a different configuration for ring 2 hexagons.
    Instead of 4 hexagons at 30, 150, 210, 330 degrees,
    try other arrangements that might pack tighter.
    """
    sqrt3 = math.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # Node 0: Center
    pos[0] = [0.0, 0.0]
    ang[0] = 0.0
    
    # Nodes 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        ang[i+1] = 0.0
    
    # Alternative: Place ring 2 hexagons at the same angles as ring 1
    # but further out (distance 4*sqrt(3) ≈ 6.928)
    # This creates a more symmetric pattern
    ring2_angles = [0, 1, 2, 3, 4, 5]  # Same angles as ring 1
    for i in range(4):
        a = ring2_angles[i] * np.pi / 3
        pos[i+7] = [4*sqrt3 * np.cos(a), 4*sqrt3 * np.sin(a)]
        ang[i+7] = 0.0
    
    return pos, ang

def _optimize_cluster_rotation(pos, ang, n_samples=360):
    """
    Optimize the global rotation of the cluster to minimize the outer hexagon side length.
    We also try translating the cluster to center it optimally.
    
    Returns: best_pos, best_ang, best_side, best_rotation_deg, best_center
    """
    best_side = float('inf')
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_rot = 0.0
    best_center = (0.0, 0.0)
    
    # First, center the cluster
    centroid = np.mean(pos, axis=0)
    pos_centered = pos - centroid
    
    for rot_deg in np.linspace(0, 60, n_samples):
        rot_rad = np.radians(rot_deg)
        c, s = np.cos(rot_rad), np.sin(rot_rad)
        rot_mat = np.array([[c, -s], [s, c]])
        
        pos_rotated = pos_centered @ rot_mat.T
        ang_rotated = ang + rot_deg
        
        # Try a few outer hexagon rotations
        for outer_rot in np.linspace(0, 30, 7):
            side = _min_outer_side(pos_rotated, ang_rotated, (0.0, 0.0), outer_rot)
            if side < best_side:
                # Verify no overlaps
                if not _check_overlap(pos_rotated, ang_rotated):
                    best_side = side
                    best_pos = pos_rotated.copy()
                    best_ang = ang_rotated.copy()
                    best_rot = rot_deg
                    best_center = (0.0, 0.0)
    
    return best_pos, best_ang, best_side, best_rot, best_center

def _local_refinement(pos, ang, side, max_iters=100):
    """
    Perform local refinement by slightly adjusting positions.
    This helps fine-tune the packing.
    """
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_side = side
    centroid = np.mean(best_pos, axis=0)
    pos_centered = best_pos - centroid
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        improved = False
        # Try small random perturbations
        for trial in range(10):
            pos_trial = pos_centered + np.random.normal(0, 0.05, size=pos_centered.shape)
            ang_trial = best_ang + np.random.normal(0, 5, size=best_ang.shape)
            
            if not _check_overlap(pos_trial, ang_trial):
                for outer_rot in np.linspace(0, 30, 7):
                    side_trial = _min_outer_side(pos_trial, ang_trial, (0.0, 0.0), outer_rot)
                    if side_trial < best_side - 1e-6:
                        best_side = side_trial
                        best_pos = pos_trial.copy()
                        best_ang = ang_trial.copy()
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_pos, best_ang, best_side

def construct():
    """
    Main entry point. Returns:
    - inner: list of (x, y, angle_degrees) for each inner hexagon
    - center: (x, y) of outer hexagon center
    - side: side length of outer hexagon
    - angle: rotation angle of outer hexagon in degrees
    """
    global START_TIME, DEADLINE
    
    # Start with the standard lattice configuration
    pos, ang = _create_lattice_cluster()
    
    # Optimize rotation
    pos, ang, side, rot, center = _optimize_cluster_rotation(pos, ang, n_samples=180)
    
    # Local refinement
    pos, ang, side = _local_refinement(pos, ang, side)
    
    # Try the alternative ring 2 configuration
    pos2, ang2 = _try_different_ring2_config()
    pos2, ang2, side2, rot2, center2 = _optimize_cluster_rotation(pos2, ang2, n_samples=180)
    pos2, ang2, side2 = _local_refinement(pos2, ang2, side2)
    
    # Choose the better configuration
    if side2 < side:
        pos, ang, side = pos2, ang2, side2
        center = center2
    else:
        center = center
    
    # Final verification
    if _check_overlap(pos, ang):
        # Fallback to the simple lattice
        pos, ang = _create_lattice_cluster()
        centroid = np.mean(pos, axis=0)
        pos = pos - centroid
        side = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)
        center = (0.0, 0.0)
    
    # Find best outer rotation
    best_outer_rot = 0.0
    best_final_side = side
    for rot_deg in np.linspace(0, 60, 13):
        s = _min_outer_side(pos, ang, (0.0, 0.0), rot_deg)
        if s < best_final_side:
            best_final_side = s
            best_outer_rot = rot_deg
    
    # Construct output
    inner_list = [[float(pos[i,0]), float(pos[i,1]), float(ang[i])] for i in range(N)]
    
    return inner_list, (float(center[0]), float(center[1])), float(best_final_side), float(best_outer_rot)

if __name__ == "__main__":
    construct()
