# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _point_in_hex(v, outer_verts):
    """Check if point v is inside or on outer hexagon (vertices in order)."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = np.cross(p2 - p1, v - p1)
        if cross < -1e-6:
            return False
    return True


def _check_all_inside(inner_verts, outer_verts):
    """Check all inner vertices are inside outer hexagon."""
    for iv in inner_verts:
        if not _point_in_hex(iv, outer_verts):
            return False
    return True


def _min_containing_s(centers, outer_angle_deg):
    """Find minimum outer hexagon side length s (centered at origin) with given angle
    that contains all unit hexagons at given centers."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for cx, cy in centers:
        verts = _hex_vertices(cx, cy, 1.0, 0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Binary search on s
    lo = np.sqrt(12.0)
    hi = 20.0
    
    for _ in range(50):
        mid = (lo + hi) / 2
        outer_verts = _hex_vertices(0, 0, mid, outer_angle_deg)
        if _check_all_inside(all_verts, outer_verts):
            hi = mid
        else:
            lo = mid
    
    return hi


def _objective(params):
    """Objective: minimize outer hexagon side length.
    params = [outer_angle_deg, cx, cy]
    Centers are fixed relative topology, shifted by (cx, cy)."""
    outer_angle = params[0]
    cx = params[1]
    cy = params[2]
    
    # Fixed relative positions (hexagonal lattice pattern for 12 hexagons)
    # Using a known good topology: center + ring of 6 + partial second ring
    # Actually, let's use a compact arrangement:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # With hexagonal offsets
    
    # Better: use the standard hexagonal packing
    # For 12 hexagons, a good arrangement is:
    # 3-4-3-2 pattern or similar
    
    # Let me use a well-known compact arrangement:
    # Positions in a hex grid (axial coordinates converted to cartesian)
    # q, r -> x = 1.5*q, y = sqrt(3)*(r + q/2)
    
    # A compact 12-hexagon cluster:
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance 2*sqrt(3) (for unit hexes touching)
    # Ring 2: 5 more hexagons
    
    # For unit regular hexagons, center-to-center distance when touching = 2*sqrt(3)/2 * 2 = 2*sqrt(3)? 
    # No: the width of a unit hexagon (flat-to-flat) = 2*sqrt(3)/2 * 2 = sqrt(3)*2? 
    # Actually, for a regular hexagon with side 1, the distance from center to vertex = 1,
    # and the distance from center to edge (apothem) = sqrt(3)/2.
    # Two adjacent hexagons (sharing an edge) have centers 2*apothem = sqrt(3) apart.
    
    # Hex grid positions (axial to cartesian, with side=1 hexagons):
    # x = 1.5 * q, y = sqrt(3) * (r + 0.5*q)
    # Adjacent centers are sqrt(3) apart.
    
    sqrt3 = np.sqrt(3)
    
    # Compact 12-hexagon arrangement (3-4-3-2 or similar):
    # Let me use a proven compact layout:
    # Positions (q, r) in axial coordinates:
    axial_positions = [
        (0, 0),      # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # ring 1 (6 hexes)
        (2, 0), (-2, 0), (2, -1), (-2, 1), (1, 1),           # 5 more for total 12
    ]
    
    # Convert to cartesian
    rel_centers = np.array([[1.5 * q, sqrt3 * (r + 0.5 * q)] for q, r in axial_positions])
    
    # Shift by (cx, cy)
    centers = rel_centers + np.array([cx, cy])
    
    s = _min_containing_s(centers, outer_angle)
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    sqrt3 = np.sqrt(3)
    
    # Compact 12-hexagon arrangement in axial coordinates
    axial_positions = [
        (0, 0),      # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # ring 1
        (2, 0), (-2, 0), (2, -1), (-2, 1), (1, 1),           # ring 2 partial
    ]
    
    # Convert to cartesian (relative to cluster center)
    rel_centers = np.array([[1.5 * q, sqrt3 * (r + 0.5 * q)] for q, r in axial_positions])
    
    # Optimize: outer_angle, cx, cy
    # Try multiple initial angles
    best_s = float('inf')
    best_params = None
    
    # Initial guesses for outer angle (0 to 60 degrees, symmetric)
    for angle_init in [0, 15, 30, 45, 60]:
        x0 = np.array([angle_init, 0.0, 0.0])
        
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: just use the fixed arrangement with a large enough hexagon
        centers = rel_centers
        s = _min_containing_s(centers, 0)
        inner_hex_data = np.column_stack([centers, np.zeros(12)])
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, s
    
    outer_angle = best_params[0]
    cx = best_params[1]
    cy = best_params[2]
    
    centers = rel_centers + np.array([cx, cy])
    s = best_s
    
    # Construct inner hexagon data
    inner_hex_data = np.column_stack([centers, np.zeros(12)])
    
    # Outer hexagon centered at origin
    outer_hex_data = np.array([0, 0, outer_angle])
    
    # Verify and adjust if needed
    outer_verts = _hex_vertices(0, 0, s, outer_angle)
    all_inner_verts = np.vstack([_hex_vertices(cx, cy, 1.0, 0) for cx, cy in centers])
    
    if not _check_all_inside(all_inner_verts, outer_verts):
        # Increase s slightly
        s *= 1.001
        outer_hex_data = np.array([0, 0, outer_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
