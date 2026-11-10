# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_enclosing_hex_side(points, angle_deg):
    """
    Find the smallest regular hexagon of given orientation containing all points.
    Returns the side length.
    """
    a = np.radians(angle_deg)
    # Edge normals: perpendicular to edges. For hex with vertices at a+60i,
    # edges are between consecutive vertices. Normal to edge i is at angle a+30+60i.
    normals = np.array([[np.cos(a + np.pi/6 + i*np.pi/3), 
                         np.sin(a + np.pi/6 + i*np.pi/3)] for i in range(6)])
    
    def side_for_center(c):
        """Compute required side length for center c."""
        max_r = 0.0
        for n in normals:
            proj = points @ n - c @ n
            r_i = max(np.max(proj), -np.min(proj))
            max_r = max(max_r, r_i)
        # r is inradius, side = 2*r/sqrt(3)
        return 2.0 * max_r / np.sqrt(3.0)
    
    # Optimize center to minimize side
    c0 = np.array([np.mean(points[:,0]), np.mean(points[:,1])])
    res = minimize(side_for_center, c0, method='Nelder-Mead',
                   options={'xatol':1e-10, 'fatol':1e-10, 'maxiter':50000})
    return res.fun


def hexagon_packing_11():
    """
    Packs 11 unit regular hexagons in a flower pattern and finds minimal enclosing regular hexagon.
    """
    # Flower pattern: 1 center, 6 ring (dist 2), 4 gaps
    ring_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    ring_pos = 2.0 * np.column_stack([np.cos(ring_angles), np.sin(ring_angles)])
    
    # Gap hexagons touch two adjacent ring hexagons
    # Distance from center: 2*sqrt(3)
    gap_dist = 2.0 * np.sqrt(3.0)
    # Pick 4 gaps for 11 total. Use gaps at 30, 150, 210, 330 degrees
    gap_angles = np.deg2rad([30, 150, 210, 330])
    gap_pos = gap_dist * np.column_stack([np.cos(gap_angles), np.sin(gap_angles)])
    
    # Initial positions
    positions = np.vstack([[0,0], ring_pos, gap_pos])  # 11x2
    angles = np.zeros(11)  # all aligned
    
    # Try different outer hexagon orientations
    best_side = np.inf
    best_config = None
    
    for outer_angle in [0, 30, 60, 15, 45, 75, 10, 20, 50]:
        # Compute enclosing hex side for this orientation
        all_verts = []
        for i in range(11):
            all_verts.append(_hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]))
        all_verts = np.vstack(all_verts)
        
        side = _min_enclosing_hex_side(all_verts, outer_angle)
        
        if side < best_side:
            best_side = side
            best_config = (positions.copy(), angles.copy(), outer_angle)
    
    positions, angles, outer_angle = best_config
    
    # Compute final enclosing hexagon
    all_verts = []
    for i in range(11):
        all_verts.append(_hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]))
    all_verts = np.vstack(all_verts)
    
    # Find optimal center
    a = np.radians(outer_angle)
    normals = np.array([[np.cos(a + np.pi/6 + i*np.pi/3), 
                         np.sin(a + np.pi/6 + i*np.pi/3)] for i in range(6)])
    
    def side_for_center(c):
        max_r = 0.0
        for n in normals:
            proj = all_verts @ n - c @ n
            r_i = max(np.max(proj), -np.min(proj))
            max_r = max(max_r, r_i)
        return 2.0 * max_r / np.sqrt(3.0)
    
    c0 = np.array([np.mean(all_verts[:,0]), np.mean(all_verts[:,1])])
    res = minimize(side_for_center, c0, method='Nelder-Mead',
                   options={'xatol':1e-10, 'fatol':1e-10, 'maxiter':50000})
    
    cx, cy = res.x
    side = res.fun
    
    # Shift to center at origin
    positions[:,0] -= cx
    positions[:,1] -= cy
    
    inner_hex_data = np.column_stack([positions[:,0], positions[:,1], angles])
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    
    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
