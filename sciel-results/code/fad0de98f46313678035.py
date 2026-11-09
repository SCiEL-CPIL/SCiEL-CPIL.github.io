# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s, tol=1e-6, edge_tol=1e-8):
    """Check validity of a packing configuration."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons have unit edges
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > edge_tol:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            # Get edge normals for both hexagons
            normals = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n = n / n_norm
                        normals.append(n)
            
            # SAT: if there exists a normal where the projections don't overlap, they are separated
            separated = False
            for n in normals:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap > -tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _build_honeycomb_cluster(rotation_deg=0.0):
    """
    Build a compact honeycomb arrangement of 12 unit hexagons.
    The hexagons are arranged in a 3-4-3-2 pattern or similar compact shape.
    All hexagons have angle = rotation_deg (they rotate together).
    
    Returns (12, 3) array of (x, y, angle_deg) for the cluster centered at origin.
    """
    # Honeycomb lattice vectors for unit hexagons (side 1):
    # Adjacent hexagons are separated by distance 2*sqrt(3)/2 * ... 
    # Actually, for unit regular hexagons (side 1), the distance between centers
    # of adjacent hexagons in a honeycomb is 2 * (apothem) = 2 * sqrt(3)/2 * 2? 
    # Let me think: a regular hexagon with side 1 has apothem = sqrt(3)/2.
    # Two adjacent hexagons in a honeycomb share an edge, so center-to-center distance = 2 * apothem = sqrt(3).
    # Wait, that's not right either. Let me reconsider.
    # A regular hexagon with side length s has:
    # - Inradius (apothem) = s * sqrt(3)/2
    # - Circumradius = s
    # For two hexagons sharing an edge, the distance between centers is 2 * inradius = s * sqrt(3).
    # For s=1: distance = sqrt(3).
    
    # Honeycomb lattice: two basis vectors at 60 degrees
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    # Actually, let me use the standard honeycomb lattice.
    # The honeycomb lattice has a triangular Bravais lattice with a 2-point basis.
    # But for our purposes, we can just place hexagons at specific positions.
    
    # Let's use a simple compact arrangement:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # For unit hexagons in honeycomb:
    # Horizontal spacing between centers in same row: sqrt(3)
    # Vertical spacing between rows: 1.5 (for offset rows)
    # Actually, let me be more careful.
    
    # For a regular hexagon with side 1, oriented with a flat top:
    # The vertices are at angles 0, 60, 120, 180, 240, 300 degrees from center.
    # The "width" (distance between parallel edges) = 2 * apothem = sqrt(3)
    # The "height" (distance between opposite vertices) = 2 * circumradius = 2
    
    # In a honeycomb packing of flat-top hexagons:
    # - Adjacent hexagons in the same row are separated by sqrt(3) horizontally
    # - Adjacent rows are separated by 1.5 vertically, and offset by sqrt(3)/2 horizontally
    
    # Let's define positions for a 3-4-3-2 arrangement:
    # Using flat-top orientation (angle 0)
    
    positions = []
    
    # Row 0 (top): 3 hexagons
    for i in range(3):
        x = (i - 1) * math.sqrt(3)
        y = 0.0
        positions.append((x, y))
    
    # Row 1: 4 hexagons, offset by sqrt(3)/2, down by 1.5
    for i in range(4):
        x = (i - 1.5) * math.sqrt(3)
        y = -1.5
        positions.append((x, y))
    
    # Row 2: 3 hexagons
    for i in range(3):
        x = (i - 1) * math.sqrt(3)
        y = -3.0
        positions.append((x, y))
    
    # Row 3: 2 hexagons, offset
    for i in range(2):
        x = (i - 0.5) * math.sqrt(3)
        y = -4.5
        positions.append((x, y))
    
    # Center the cluster
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    cx_mean = (xs.min() + xs.max()) / 2
    cy_mean = (ys.min() + ys.max()) / 2
    
    positions = [(x - cx_mean, y - cy_mean) for x, y in positions]
    
    # Apply rotation
    rot_rad = math.radians(rotation_deg)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    
    rotated_positions = []
    for x, y in positions:
        xr = x * cos_r - y * sin_r
        yr = x * sin_r + y * cos_r
        rotated_positions.append((xr, yr))
    
    inner = np.zeros((12, 3))
    for i, (x, y) in enumerate(rotated_positions):
        inner[i, 0] = x
        inner[i, 1] = y
        inner[i, 2] = rotation_deg
    
    return inner

def _compute_min_s(inner, outer_angle_deg=0.0):
    """
    Compute the minimum side length s of a regular hexagon (with given angle)
    that can contain all vertices of the inner hexagons, centered at origin.
    
    For a regular hexagon with side s, angle a, centered at (0,0):
    The hexagon is the intersection of 6 half-planes.
    For each edge, we can compute the required s such that all points are inside.
    
    A regular hexagon with side s and angle a has edges that are at distance s*sqrt(3)/2 
    from the center (the apothem). The outward normal of edge i points in direction 
    a + 60*i + 90 degrees (or similar).
    
    Actually, let's think about this differently.
    For a regular hexagon centered at origin with side s and angle a:
    The vertices are at:
    (s*cos(a+60k), s*sin(a+60k)) for k=0..5
    
    The edges connect consecutive vertices. The edge between vertex k and k+1
    has a midpoint at distance (apothem) = s*sqrt(3)/2 from center, in the direction
    of angle a + 60*(k+0.5).
    
    The half-plane constraint for edge k is:
    dot(point, normal_k) <= s*sqrt(3)/2
    where normal_k is the unit vector in direction a + 60*(k+0.5).
    
    So for a set of points P, the minimum s is:
    s = max over all points p and all edge directions k of:
        dot(p, normal_k) / (sqrt(3)/2)
    
    But we need dot(p, normal_k) to be non-negative for the constraint to make sense.
    Actually, the hexagon contains all points p such that for all k:
    dot(p, normal_k) <= s * sqrt(3)/2
    
    So s >= max_k max_p dot(p, normal_k) / (sqrt(3)/2)
    
    But we also need to consider that dot(p, normal_k) could be negative, which is fine
    (the point is on the interior side of that edge's line).
    
    So: s_min = max_k (max_p dot(p, normal_k)) / (sqrt(3)/2)
    
    But wait, we need s to be positive, so we take max with 0.
    """
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    a = math.radians(outer_angle_deg)
    s_min = 0.0
    
    for k in range(6):
        # Normal direction for edge k
        # Edge k is between vertex k and vertex k+1
        # The outward normal points in direction a + 60*(k+0.5)
        normal_angle = a + math.radians(60.0 * (k + 0.5))
        normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
        
        # Max dot product
        dots = all_verts @ normal
        max_dot = dots.max()
        
        # Required s for this edge
        if max_dot > 0:
            s_required = max_dot / (math.sqrt(3) / 2)
            s_min = max(s_min, s_required)
    
    return s_min

def _objective(params):
    """
    Objective: minimize s for the given configuration.
    params: [rotation_deg, outer_angle_deg, tx, ty]
    tx, ty: translation of the cluster
    """
    rot = params[0]
    outer_angle = params[1]
    tx = params[2]
    ty = params[3]
    
    inner = _build_honeycomb_cluster(rot)
    inner[:, 0] += tx
    inner[:, 1] += ty
    
    s = _compute_min_s(inner, outer_angle)
    
    # Check if the configuration is valid with this s
    outer = np.array([0.0, 0.0, 0.0])
    
    if not _is_valid(inner, outer, s):
        return s + 100.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s limit
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    def try_config(inner, outer_angle_deg):
        nonlocal best_s, best_inner, best_outer
        s = _compute_min_s(inner, outer_angle_deg)
        outer = np.array([0.0, 0.0, 0.0])
        
        if s >= best_s:
            return False
        
        if _is_valid(inner, outer, s):
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()
            return True
        return False
    
    # Phase 1: Grid search over rotation and outer angle
    rotations = np.linspace(0, 60, 13)  # 0 to 60 degrees in 5-degree steps
    outer_angles = np.linspace(0, 60, 13)
    
    for rot in rotations:
        if time.time() > deadline - 60:
            break
        inner = _build_honeycomb_cluster(rot)
        for oang in outer_angles:
            s_test = _compute_min_s(inner, oang)
            if s_test < best_s:
                try_config(inner, oang)
    
    # Phase 2: Local optimization from best found
    if best_inner is not None:
        # Extract best parameters
        # We need to figure out which rot and oang gave the best result
        # For now, just do local optimization from a few starting points
        
        def local_obj(params):
            rot = params[0]
            oang = params[1]
            tx = params[2]
            ty = params[3]
            inner = _build_honeycomb_cluster(rot)
            inner[:, 0] += tx
            inner[:, 1] += ty
            s = _compute_min_s(inner, oang)
            outer = np.array([0.0, 0.0, 0.0])
            if not _is_valid(inner, outer, s):
                return s + 100.0
            return s
        
        # Try local optimization from several starting points
        start_points = [
            [0.0, 0.0, 0.0, 0.0],
            [15.0, 0.0, 0.0, 0.0],
            [30.0, 0.0, 0.0, 0.0],
            [0.0, 30.0, 0.0, 0.0],
            [15.0, 30.0, 0.0, 0.0],
        ]
        
        for x0 in start_points:
            if time.time() > deadline - 30:
                break
            try:
                result = minimize(local_obj, x0, method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if result.fun < best_s:
                    rot, oang, tx, ty = result.x
                    inner = _build_honeycomb_cluster(rot)
                    inner[:, 0] += tx
                    inner[:, 1] += ty
                    try_config(inner, oang)
            except Exception:
                pass
    
    # Phase 3: Differential evolution for broader search
    if time.time() < deadline - 60:
        bounds = [(-30, 30), (-30, 30), (-3, 3), (-3, 3)]
        
        def de_callback(xk, convergence):
            val = _objective(xk)
            rot, oang, tx, ty = xk
            inner = _build_honeycomb_cluster(rot)
            inner[:, 0] += tx
            inner[:, 1] += ty
            try_config(inner, oang)
            return time.time() > deadline - 30
        
        try:
            res = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                tol=1e-8,
                mut=(0.5, 1.5),
                recomb=0.7,
                seed=42,
                polish=True,
                callback=de_callback
            )
            if res.fun < best_s:
                rot, oang, tx, ty = res.x
                inner = _build_honeycomb_cluster(rot)
                inner[:, 0] += tx
                inner[:, 1] += ty
                try_config(inner, oang)
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_inner is None:
        inner = _build_honeycomb_cluster(0.0)
        outer = np.array([0.0, 0.0, 0.0])
        s = _compute_min_s(inner, 0.0)
        # Make sure it's at least sqrt(12)
        s = max(s, math.sqrt(12) + 1e-6)
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
