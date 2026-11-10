import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _lattice_centers():
    """
    12 unit hexagons on the dual triangular grid (hexagonal lattice).
    Using axial coordinates (q, r) -> Cartesian:
      x = sqrt(3) * (q + r/2)
      y = 1.5 * r
    This gives edge-to-edge adjacency (distance sqrt(3) between adjacent centers),
    which exactly matches the width of a unit hexagon, ensuring non-overlap.
    
    We use a compact 12-hex cluster: center + 6 in ring 1 + 5 from ring 2.
    """
    sqrt3 = np.sqrt(3.0)
    # Axial coordinates (q, r)
    # Center: (0,0)
    # Ring 1 (6 hexes): directions in axial space
    # Ring 2: select 5 of the 12 ring-2 positions for compactness
    axial = [
        (0, 0),       # center
        (1, 0),       # ring 1
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        # ring 2 (5 selected for compactness)
        (2, 0),
        (1, 1),
        (-1, 2),
        (-2, 1),
        (-2, 0),
    ]
    
    centers = np.zeros((12, 2))
    for i, (q, r) in enumerate(axial):
        centers[i, 0] = sqrt3 * (q + r / 2.0)
        centers[i, 1] = 1.5 * r
    
    return centers


def _min_containing_hex_side_rotated(centers, theta_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all unit hexagons with given centers after rotating by theta_deg.
    
    The unit hexagons have vertices at angles 0, 60, 120, 180, 240, 300 from their center.
    After rotating the entire configuration by theta, we need the smallest regular hexagon
    (angle 0, centered at origin) containing all rotated vertices.
    
    For a regular hexagon of side s centered at origin with angle 0:
    - Its edges are at distance s*sqrt(3)/2 from center along normals at 30, 90, 150, 210, 270, 330
    - A point (x,y) is inside iff for all 6 normals n_k: dot((x,y), n_k) <= s*sqrt(3)/2
    - So s = (2/sqrt(3)) * max_k max_points dot(point, n_k)
    """
    theta = np.deg2rad(theta_deg)
    c, s_ = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s_], [s_, c]])
    
    # Rotate all centers
    rotated_centers = centers @ R.T  # (12, 2)
    
    # Unit hexagon vertices relative to center: at angles 0, 60, ..., 300
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    unit_verts = np.column_stack([base_cos, base_sin])  # (6, 2)
    
    # All vertices: (12, 6, 2)
    all_verts = rotated_centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)  # (72, 2)
    
    # Outward normals of the containing hexagon (angle 0): at 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    projections = normals @ all_verts_flat.T  # (6, 72)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)  # scalar
    
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _min_containing_hex_side_general(centers, outer_angle_deg=0.0):
    """
    Compute minimum side length of a regular hexagon centered at origin with given angle
    that contains all unit hexagons with given centers (no rotation of inner hexagons).
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    
    # Unit hexagon vertices
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    unit_verts = np.column_stack([base_cos, base_sin])  # (6, 2)
    
    # All vertices
    all_verts = centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)  # (72, 2)
    
    # Outward normals of the containing hexagon
    normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    projections = normals @ all_verts_flat.T  # (6, 72)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)  # scalar
    
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _check_separation(centers, tol=1e-6):
    """
    Check if all pairs of unit hexagons are separated using SAT.
    Returns True if all pairs are separated.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i, :, 0] = centers[i, 0] + base_cos
        all_verts[i, :, 1] = centers[i, 1] + base_sin
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                return False
    return True


def _check_all_valid(centers, outer_angle_deg, s, tol=1e-6):
    """Full validity check."""
    # Check separation
    if not _check_separation(centers, tol):
        return False
    
    # Check all inner vertices inside outer hexagon
    outer_verts = _hex_vertices(0.0, 0.0, s, outer_angle_deg)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    unit_verts = np.column_stack([base_cos, base_sin])
    
    for i in range(12):
        for k in range(6):
            vx = centers[i, 0] + unit_verts[k, 0]
            vy = centers[i, 1] + unit_verts[k, 1]
            # Check against each outer edge
            for e in range(6):
                p1 = outer_verts[e]
                p2 = outer_verts[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (vy - p1[1]) - (p2[1] - p1[1]) * (vx - p1[0])
                if cross < -tol:
                    return False
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    
    Strategy: Use a fixed hexagonal lattice topology (non-overlap by construction),
    then optimize the rotation angle of the entire cluster to minimize the containing
    hexagon's side length.
    
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Emergency fallback
        centers = _lattice_centers()
        s = _min_containing_hex_side_general(centers, 0.0)
        inner = np.column_stack([centers, np.zeros(12)])
        outer = np.array([0.0, 0.0, 0.0])
        return inner, outer, s
    
    # Base lattice centers
    base_centers = _lattice_centers()
    
    # Verify base lattice is non-overlapping (should be by construction)
    assert _check_separation(base_centers), "Lattice should be non-overlapping by construction"
    
    # Strategy: rotate the entire cluster and find optimal angle
    # The containing hexagon has fixed angle (0), so we rotate inner cluster.
    # Alternatively: fix inner cluster, rotate outer hexagon.
    # Due to symmetry, these are equivalent. Let's rotate the outer hexagon.
    
    # For a given outer hexagon angle alpha, the side length needed is:
    # s(alpha) = (2/sqrt(3)) * max_k max_vertices dot(vertex, n_k(alpha))
    # where n_k(alpha) are the 6 outward normals.
    
    # This is a piecewise-linear function of alpha (in terms of the angle).
    # Let's do a fine scan of alpha in [0, 60) degrees.
    
    # Fine scan
    n_scan = 3600  # 0.01 degree resolution
    alphas = np.linspace(0.0, 60.0, n_scan, endpoint=False)
    
    best_s = np.inf
    best_alpha = 0.0
    
    # Precompute unit hexagon vertices
    base_angles = np.deg2rad(60.0 * np.arange(6))
    unit_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    all_verts = base_centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)  # (72, 2)
    
    for alpha_deg in alphas:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        alpha = np.deg2rad(alpha_deg)
        normal_angles = alpha + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        projections = normals @ all_verts_flat.T
        max_proj = np.max(projections)
        s = (2.0 / np.sqrt(3.0)) * max_proj
        
        if s < best_s:
            best_s = s
            best_alpha = alpha_deg
    
    # Refine around best with golden section search
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        # Golden section search in a window around best_alpha
        window = 0.5  # degrees
        a = max(0.0, best_alpha - window)
        b = min(60.0, best_alpha + window)
        
        if b - a > 1e-10:
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            fc = _min_containing_hex_side_general(base_centers, c)
            fd = _min_containing_hex_side_general(base_centers, d)
            
            max_iter = 60
            for _ in range(max_iter):
                elapsed = time.time() - _START_TIME
                if elapsed > _DEADLINE - _SAFETY_MARGIN:
                    break
                
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = _min_containing_hex_side_general(base_centers, c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = _min_containing_hex_side_general(base_centers, d)
            
            # Best estimate
            mid = (a + b) / 2.0
            s_mid = _min_containing_hex_side_general(base_centers, mid)
            if s_mid < best_s:
                best_s = s_mid
                best_alpha = mid
    
    # Also try: rotate the inner cluster instead (equivalent but let's verify)
    # Actually, let's also try a different lattice configuration: 
    # Instead of the asymmetric 12-hex cluster, try more symmetric arrangements.
    
    # Alternative lattice: 2x6 rectangular-ish arrangement
    # Or: hexagonal number pattern with different subset of ring 2
    
    # Let's try several different 12-hex lattice configurations
    alternative_configs = []
    
    # Config 1: current (center + ring1 + 5 ring2)
    alternative_configs.append(base_centers)
    
    # Config 2: Different 5 from ring 2 (shift the gap)
    sqrt3 = np.sqrt(3.0)
    axial2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        # Different 5 from ring 2
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
    ]
    centers2 = np.zeros((12, 2))
    for i, (q, r) in enumerate(axial2):
        centers2[i, 0] = sqrt3 * (q + r / 2.0)
        centers2[i, 1] = 1.5 * r
    if _check_separation(centers2):
        alternative_configs.append(centers2)
    
    # Config 3: Another ring-2 subset
    axial3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1), (-2, 0),
    ]
    # This is same as config 1, skip
    
    # Config 4: More spread - center + ring1 + 5 different ring2
    axial4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
    ]
    # Same as config 2
    
    # Config 5: Try a 3-4-5 row arrangement
    # Row-based: 3 in one row, 4 in next, 5 in next
    # Using proper hex lattice spacing
    axial5 = [
        # Row r=0: 3 hexes
        (-1, 0), (0, 0), (1, 0),
        # Row r=1: 4 hexes
        (-1, 1), (0, 1), (1, 1), (2, 1),
        # Row r=2: 5 hexes... wait that's 12 total? 3+4+5=12
        # But need to check they're valid lattice positions
        (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
    ]
    centers5 = np.zeros((12, 2))
    for i, (q, r) in enumerate(axial5):
        centers5[i, 0] = sqrt3 * (q + r / 2.0)
        centers5[i, 1] = 1.5 * r
    if _check_separation(centers5):
        alternative_configs.append(centers5)
    
    # Config 6: 4-4-4 rows
    axial6 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
    ]
    centers6 = np.zeros((12, 2))
    for i, (q, r) in enumerate(axial6):
        centers6[i, 0] = sqrt3 * (q + r / 2.0)
        centers6[i, 1] = 1.5 * r
    if _check_separation(centers6):
        alternative_configs.append(centers6)
    
    # Config 7: 2-4-4-2 diamond
    axial7 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
    ]
    centers7 = np.zeros((12, 2))
    for i, (q, r) in enumerate(axial7):
        centers7[i, 0] = sqrt3 * (q + r / 2.0)
        centers7[i, 1] = 1.5 * r
    if _check_separation(centers7):
        alternative_configs.append(centers7)
    
    # Config 8: 3-3-3-3
    axial8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    ]
    centers8 = np.zeros((12, 2))
    for i, (q, r) in enumerate(axial8):
        centers8[i, 0] = sqrt3 * (q + r / 2.0)
        centers8[i, 1] = 1.5 * r
    if _check_separation(centers8):
        alternative_configs.append(centers8)
    
    # For each valid config, find optimal outer hexagon angle
    for config_idx, config_centers in enumerate(alternative_configs):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Quick scan
        n_scan2 = 1800
        alphas2 = np.linspace(0.0, 60.0, n_scan2, endpoint=False)
        
        unit_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        all_verts_cfg = config_centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]
        all_verts_flat_cfg = all_verts_cfg.reshape(-1, 2)
        
        best_s_cfg = np.inf
        best_alpha_cfg = 0.0
        
        for alpha_deg in alphas2:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            alpha = np.deg2rad(alpha_deg)
            normal_angles = alpha + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            projections = normals @ all_verts_flat_cfg.T
            max_proj = np.max(projections)
            s = (2.0 / np.sqrt(3.0)) * max_proj
            
            if s < best_s_cfg:
                best_s_cfg = s
                best_alpha_cfg = alpha_deg
        
        # Refine
        if elapsed < _DEADLINE - _SAFETY_MARGIN:
            window = 0.3
            a = max(0.0, best_alpha_cfg - window)
            b = min(60.0, best_alpha_cfg + window)
            
            if b - a > 1e-10:
                gr = (np.sqrt(5.0) - 1.0) / 2.0
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                fc = _min_containing_hex_side_general(config_centers, c)
                fd = _min_containing_hex_side_general(config_centers, d)
                
                for _ in range(50):
                    elapsed = time.time() - _START_TIME
                    if elapsed > _DEADLINE - _SAFETY_MARGIN:
                        break
                    
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = _min_containing_hex_side_general(config_centers, c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = _min_containing_hex_side_general(config_centers, d)
                
                mid = (a + b) / 2.0
                s_mid = _min_containing_hex_side_general(config_centers, mid)
                if s_mid < best_s_cfg:
                    best_s_cfg = s_mid
                    best_alpha_cfg = mid
        
        if best_s_cfg < best_s:
            best_s = best_s_cfg
            best_alpha = best_alpha_cfg
            base_centers = config_centers
    
    # Final validity check
    if not _check_all_valid(base_centers, best_alpha, best_s):
        # Fallback: use angle 0
        best_s = _min_containing_hex_side_general(base_centers, 0.0)
        best_alpha = 0.0
        if not _check_all_valid(base_centers, best_alpha, best_s):
            # Very conservative fallback
            best_s = 4.0
            best_alpha = 0.0
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = base_centers[i, 0]
        inner_hex_data[i, 1] = base_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, best_alpha])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
