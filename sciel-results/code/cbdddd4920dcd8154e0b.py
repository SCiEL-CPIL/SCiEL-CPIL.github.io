# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side(inner_vertices, outer_angle_deg):
    """
    Given inner hexagon vertices (12x6x2) and outer hexagon angle,
    compute the minimum side length s of an outer regular hexagon (centered at origin)
    that contains all inner vertices.
    """
    a_outer = np.radians(outer_angle_deg)
    min_s = 0.0
    for k in range(6):
        angle_k = a_outer + np.radians(30 + 60 * k)
        n_k = np.array([np.cos(angle_k), np.sin(angle_k)])
        # Project all inner vertices onto this normal
        projections = inner_vertices @ n_k  # shape (72,)
        max_proj = np.max(projections)
        # s >= 2/sqrt(3) * max_proj
        s_needed = 2.0 / np.sqrt(3.0) * max_proj
        if s_needed > min_s:
            min_s = s_needed
    return min_s


def _get_lattice_points(a):
    """
    Returns 12 points on a triangular lattice with spacing a.
    We use a compact arrangement:
    1 center, 6 neighbors, and 5 from the next ring.
    To be more symmetric, we might choose a different set.
    Let's use the set:
    (0,0)
    (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)  [6 neighbors]
    (2,0), (-2,0), (0,2), (0,-2), (1,1) [5 extra]
    This is the same as before. Let's try to make it more symmetric.
    Actually, a 3x4 rectangle on the lattice might be better?
    Points:
    i in -1..1, j in -1..2 ? No.
    
    Let's stick to the "flower + 5" but ensure the optimization can rotate it.
    The specific choice of 5 points from the second ring matters.
    If we pick (2,0), (-2,0), (0,2), (0,-2), (1,1), it's asymmetric.
    
    Alternative: A rhombus of side 3?
    Or a parallelogram?
    
    Let's try a different set of 12 points that is more symmetric:
    Take the 19-point hexagon (side 2) and remove 7 points from the corners?
    No, that breaks the lattice.
    
    Let's use the standard "12 hexagons" packing which is often a 2x2x3 block in a rectangular grid?
    No, triangular is denser.
    
    Let's use the following 12 points on the triangular lattice (i,j):
    (0,0)
    (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    (2,0), (-2,0), (0,2), (0,-2), (1,1)
    
    Wait, if we allow the lattice to be optimized, the "asymmetry" of the point set might be corrected by skewing the lattice?
    But we are constrained to triangular lattice.
    
    Let's try a different point set:
    (0,0)
    (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    (2,-1), (-2,1), (1,1), (-1,-1), (0,2)
    This is also asymmetric.
    
    Actually, the optimal packing of 12 circles is known. It's not a simple subset of the infinite lattice in a symmetric way, but often close to a rectangular or rhombic arrangement.
    For hexagons, it's similar.
    
    Let's just use the current point set but optimize $a$ and rotations.
    The key is that the current code allows $a \neq b$ and $\theta \neq 60$.
    Restricting to triangular lattice ($a=b, \theta=60$) reduces parameters and might find a better local minimum if the previous search was too broad.
    """
    indices = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
        (1, 1)
    ]
    pts = np.array(indices, dtype=float)
    # Triangular lattice basis
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.5, np.sqrt(3)/2])
    
    centers = pts[:, 0] * e1 + pts[:, 1] * e2
    centers *= a
    return centers


def _check_overlaps(vertices):
    """
    Check for overlaps between inner hexagons using SAT.
    Returns True if no overlaps.
    """
    for i in range(12):
        for j in range(i + 1, 12):
            vi = vertices[i]
            vj = vertices[j]
            
            overlap = True
            for vset in [vi, vj]:
                for k in range(6):
                    p1 = vset[k]
                    p2 = vset[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm_len = np.linalg.norm(normal)
                    if norm_len < 1e-10:
                        continue
                    normal = normal / norm_len
                    
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        overlap = False
                        break
                if not overlap:
                    break
            
            if overlap:
                return False
    return True


def _objective(params):
    """
    params: (a, rot_inner, rot_outer)
    """
    a, rot_inner, rot_outer = params
    
    if a < 1.5:
        return 1e10
        
    centers = _get_lattice_points(a)
    
    # Apply rotation to centers
    rot_rad = np.radians(rot_inner)
    R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
    centers = centers @ R.T
    
    # Compute vertices
    vertices = np.zeros((12, 6, 2))
    for idx in range(12):
        cx, cy = centers[idx]
        vertices[idx] = _hex_vertices(cx, cy, 1.0, rot_inner)
    
    # Check overlaps
    if not _check_overlaps(vertices):
        return 1e10
    
    # Compute outer side
    s = _min_outer_side(vertices.reshape(72, 2), rot_outer)
    
    if s < np.sqrt(12) - 1e-6:
        return 1e10
        
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = 620
    
    best_s = 1e10
    best_params = [2.0, 0, 0]
    
    # Initial guesses
    # a is the distance between lattice points.
    # For non-overlapping unit hexagons, min distance is 2*sqrt(3) ~ 3.464?
    # No, the distance between centers of touching hexagons is 2 * apothem = 2 * sqrt(3)/2 * 2?
    # Apothem of unit hexagon is sqrt(3)/2.
    # Distance between centers of touching hexagons is 2 * sqrt(3)/2 = sqrt(3) ~ 1.732?
    # Wait.
    # Hexagon side 1.
    # Distance from center to vertex is 1.
    # Distance from center to edge is sqrt(3)/2.
    # Two hexagons touching edge-to-edge have center distance 2 * sqrt(3)/2 = sqrt(3).
    # So a should be at least sqrt(3) ~ 1.732.
    
    guesses = [
        [1.732, 0, 0],
        [1.732, 30, 0],
        [1.8, 0, 0],
        [1.8, 30, 0],
        [1.9, 0, 0],
        [1.9, 30, 0],
    ]
    
    for guess in guesses:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(_objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Also try Powell
        try:
            res = minimize(_objective, guess, method='Powell', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
        except Exception:
            pass
            
    a, rot_inner, rot_outer = best_params
    
    # Final calculation
    centers = _get_lattice_points(a)
    rot_rad = np.radians(rot_inner)
    R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
    centers = centers @ R.T
    
    vertices = np.zeros((12, 6, 2))
    for idx in range(12):
        cx, cy = centers[idx]
        vertices[idx] = _hex_vertices(cx, cy, 1.0, rot_inner)
    
    s = _min_outer_side(vertices.reshape(72, 2), rot_outer)
    
    # Fallback
    if s < np.sqrt(12) - 1e-6 or s >= 1e9:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        s = 8.0
        return inner_hex_data, outer_hex_data, s
    
    inner_hex_data = np.column_stack([centers, np.full(12, rot_inner)])
    outer_hex_data = np.array([0, 0, rot_outer])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
