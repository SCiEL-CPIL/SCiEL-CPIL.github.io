# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _compute_min_outer(verts, rot_deg):
    """
    Compute minimum side length s and center (ox, oy) of outer hexagon
    with rotation rot_deg containing all vertices in verts (N, 2).
    Returns (s, ox, oy).
    """
    sqrt3 = np.sqrt(3)
    rot = np.radians(rot_deg)
    
    # Edge normals of outer hexagon: perpendicular to edges
    # Edge i goes from vertex i to vertex i+1
    # Vertex i is at angle (rot + 60*i)
    # Edge from vertex i to vertex i+1 has direction (rot + 60*i + 30)
    # Normal is perpendicular: (rot + 60*i + 30 + 90) = (rot + 60*i + 120)
    # Actually, let me think more carefully.
    # Vertex i: angle = rot + 60*i
    # Edge from vertex i to vertex i+1: the outward normal direction
    # For a regular hexagon, the outward normal of edge i is at angle rot + 60*i + 30
    normal_angles = rot + np.radians(60 * np.arange(6) + 30)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # For each normal direction, project all vertices
    projections = verts @ normals.T  # (N, 6)
    p_min = projections.min(axis=0)  # (6,)
    p_max = projections.max(axis=0)  # (6,)
    
    # The center projection along normal i should be (p_min[i] + p_max[i]) / 2
    # center @ normals[i] = (p_min[i] + p_max[i]) / 2
    b = (p_min + p_max) / 2.0  # (6,)
    
    # Solve for center
    center, _, _, _ = np.linalg.lstsq(normals, b, rcond=None)
    ox, oy = center
    
    # Compute required side length
    center_projections = normals @ center  # (6,)
    s = 0.0
    for i in range(6):
        d_proj = projections[:, i] - center_projections[i]
        s = max(s, np.abs(d_proj).max() * 2.0 / sqrt3)
    
    return s, ox, oy


def _separation_sat(c1, a1, c2, a2, eps=1e-7):
    """
    Check if two unit hexagons with centers c1, c2 and angles a1, a2 are separated.
    Uses Separating Axis Theorem. Returns True if separated.
    """
    a1_rad = np.radians(a1)
    a2_rad = np.radians(a2)
    
    # Edge normals for hex1
    n1 = a1_rad + np.radians(60 * np.arange(6))
    normals1 = np.column_stack([np.cos(n1), np.sin(n1)])
    
    # Edge normals for hex2
    n2 = a2_rad + np.radians(60 * np.arange(6))
    normals2 = np.column_stack([np.cos(n2), np.sin(n2)])
    
    all_normals = np.vstack([normals1, normals2])  # (12, 2)
    
    verts1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    verts2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    proj1 = verts1 @ all_normals.T  # (6, 12)
    proj2 = verts2 @ all_normals.T  # (6, 12)
    
    p1_min = proj1.min(axis=0)
    p1_max = proj1.max(axis=0)
    p2_min = proj2.min(axis=0)
    p2_max = proj2.max(axis=0)
    
    # Check if any axis separates
    if np.any(p1_max + eps < p2_min) or np.any(p2_max + eps < p1_min):
        return True
    
    return False


def _validate(inner, outer, s, eps=1e-6):
    """Full validation of packing."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ox, oy, oang = outer
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check separation between all pairs of inner hexagons
    for i in range(12):
        for j in range(i+1, 12):
            if not _separation_sat(inner[i, :2], inner[i, 2], inner[j, :2], inner[j, 2], eps):
                return False
    
    # Check containment: every inner vertex must be inside or on the outer hexagon
    for i in range(12):
        cx, cy, aang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, aang)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    return False
    
    return True


def _make_lattice_config(lattice_coords, e1, e2, angles=None):
    """Create centers and angles from lattice coordinates."""
    n = len(lattice_coords)
    if angles is None:
        angles = np.zeros(n)
    centers = np.array([c[0]*e1 + c[1]*e2 for c in lattice_coords])
    return centers, angles


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Uses rigid cluster representation with multiple seed configurations.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    sqrt3 = np.sqrt(3)
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    def try_config(centers, angles, rot_deg):
        nonlocal best_s, best_inner, best_outer
        if time.time() > deadline - 5:
            return
        n_inner = len(centers)
        all_verts = np.zeros((n_inner * 6, 2))
        for i in range(n_inner):
            all_verts[6*i:6*(i+1)] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        
        s, ox, oy = _compute_min_outer(all_verts, rot_deg)
        
        if s < best_s:
            inner = np.column_stack([centers, angles])
            outer = np.array([ox, oy, rot_deg])
            if _validate(inner, outer, s):
                best_s = s
                best_inner = inner.copy()
                best_outer = outer.copy()
    
    # === Define multiple seed configurations ===
    
    # Config 1: Standard hexagonal lattice (center + 6 + 5 outer ring)
    # Using axial coordinates (q, r) with e1 = (2, 0), e2 = (1, sqrt(3))
    e1_std = np.array([2.0, 0.0])
    e2_std = np.array([1.0, sqrt3])
    
    lattice1 = [(0, 0),
                (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                (-1, 2), (-2, 1), (2, -1), (1, -2), (0, 2)]
    
    # Config 2: Slightly different arrangement - 2-3-4-3 rows
    # This is a more compact packing
    # Row structure: 2, 3, 4, 3 hexagons
    lattice2 = [
        # Row 0 (2 hex): offset
        (-1, 0), (0, 0),
        # Row 1 (3 hex)
        (-1, 1), (0, 1), (1, 1),
        # Row 2 (4 hex)
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        # Row 3 (3 hex)
        (-1, 3), (0, 3), (1, 3),
    ]
    
    # Config 3: 3-4-3-2 rows
    lattice3 = [
        # Row 0 (3 hex)
        (-1, 0), (0, 0), (1, 0),
        # Row 1 (4 hex)
        (-2, 1), (-1, 1), (0, 1), (1, 1),
        # Row 2 (3 hex)
        (-1, 2), (0, 2), (1, 2),
        # Row 3 (2 hex)
        (0, 3), (1, 3),
    ]
    
    # Config 4: Triangular-ish arrangement
    lattice4 = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1), (0, -1),
        (0, 2), (1, 2), (-1, 2),
        (2, 0), (-2, 0),
    ]
    
    seeds = [
        (lattice1, e1_std, e2_std, "std"),
        (lattice2, e1_std, e2_std, "2343"),
        (lattice3, e1_std, e2_std, "3432"),
        (lattice4, e1_std, e2_std, "tri"),
    ]
    
    # Phase 1: Coarse rotation search for each seed
    for lattice, e1, e2, name in seeds:
        if time.time() > deadline - 120:
            break
        centers, angles = _make_lattice_config(lattice, e1, e2)
        
        # Check if this is a valid non-overlapping configuration
        valid_seed = True
        for i in range(12):
            for j in range(i+1, 12):
                if not _separation_sat(centers[i], angles[i], centers[j], angles[j], 1e-7):
                    valid_seed = False
                    break
            if not valid_seed:
                break
        
        if not valid_seed:
            # Try with slight expansion
            for scale in [1.001, 1.002, 1.005]:
                e1s = e1 * scale
                e2s = e2 * scale
                centers, angles = _make_lattice_config(lattice, e1s, e2s)
                valid_seed = True
                for i in range(12):
                    for j in range(i+1, 12):
                        if not _separation_sat(centers[i], angles[i], centers[j], angles[j], 1e-7):
                            valid_seed = False
                            break
                    if not valid_seed:
                        break
                if valid_seed:
                    break
            if not valid_seed:
                continue
        
        for rot in np.linspace(0, 60, 31):
            if time.time() > deadline - 100:
                break
            try_config(centers, angles, rot)
    
    # Phase 2: Fine rotation search around best found
    if best_s < np.inf:
        # Find best rotation for each seed more finely
        for lattice, e1, e2, name in seeds:
            if time.time() > deadline - 80:
                break
            centers, angles = _make_lattice_config(lattice, e1, e2)
            
            valid_seed = True
            for i in range(12):
                for j in range(i+1, 12):
                    if not _separation_sat(centers[i], angles[i], centers[j], angles[j], 1e-7):
                        valid_seed = False
                        break
                if not valid_seed:
                    break
            if not valid_seed:
                continue
            
            # Fine search
            for rot in np.linspace(0, 60, 121):
                if time.time() > deadline - 60:
                    break
                try_config(centers, angles, rot)
    
    # Phase 3: Optimize lattice parameters using Nelder-Mead
    if time.time() < deadline - 100:
        for lattice, e1_init, e2_init, name in seeds:
            if time.time() > deadline - 80:
                break
            
            def obj(params):
                if time.time() > deadline - 70:
                    return 1e10
                e1_len, e1_ang, e2_len, e2_ang = params
                e1_v = np.array([e1_len * np.cos(np.radians(e1_ang)), e1_len * np.sin(np.radians(e1_ang))])
                e2_v = np.array([e2_len * np.cos(np.radians(e2_ang)), e2_len * np.sin(np.radians(e2_ang))])
                
                c = np.array([p[0]*e1_v + p[1]*e2_v for p in lattice])
                ang = np.zeros(12)
                
                # Check separation
                for i in range(12):
                    for j in range(i+1, 12):
                        if not _separation_sat(c[i], ang[i], c[j], ang[j], 1e-7):
                            return 1e10
                
                # Find best rotation
                best_local = 1e10
                all_verts = np.zeros((72, 2))
                for i in range(12):
                    all_verts[6*i:6*(i+1)] = _hex_vertices(c[i, 0], c[i, 1], 1.0, ang[i])
                
                for rot in np.linspace(0, 60, 13):
                    s, _, _ = _compute_min_outer(all_verts, rot)
                    best_local = min(best_local, s)
                
                return best_local
            
            try:
                x0 = [np.linalg.norm(e1_init), 0.0, np.linalg.norm(e2_init), 60.0]
                res = minimize(obj, x0, method='Nelder-Mead',
                              options={'maxiter': 80, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < 1e9:
                    e1_len, e1_ang, e2_len, e2_ang = res.x
                    e1_v = np.array([e1_len * np.cos(np.radians(e1_ang)), e1_len * np.sin(np.radians(e1_ang))])
                    e2_v = np.array([e2_len * np.cos(np.radians(e2_ang)), e2_len * np.sin(np.radians(e2_ang))])
                    c_opt = np.array([p[0]*e1_v + p[1]*e2_v for p in lattice])
                    ang_opt = np.zeros(12)
                    
                    for rot in np.linspace(0, 60, 61):
                        if time.time() > deadline - 40:
                            break
                        try_config(c_opt, ang_opt, rot)
            except Exception:
                pass
    
    # Phase 4: Optimize individual hexagon positions (free parameters)
    if time.time() < deadline - 60:
        # Start from best configuration found
        if best_inner is not None:
            # Use the best configuration as starting point
            # Parameterize: for each hex, (dx, dy, dangle) relative to current position
            # But this is 36 parameters - too many for Nelder-Mead
            # Instead, try small perturbations of the lattice
            
            pass
        
        # Try optimizing with fewer parameters: global rotation + scale + offset
        for lattice, e1_init, e2_init, name in seeds[:2]:
            if time.time() > deadline - 40:
                break
            
            def obj2(params):
                if time.time() > deadline - 35:
                    return 1e10
                scale, rot_e1, rot_e2, dx, dy = params
                e1_v = e1_init * scale
                e2_v = e2_init * scale
                # Apply small rotation
                theta1 = np.radians(rot_e1)
                theta2 = np.radians(rot_e2)
                R1 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])
                R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])
                e1_v = R1 @ e1_v
                e2_v = R2 @ e2_v
                
                c = np.array([p[0]*e1_v + p[1]*e2_v + np.array([dx, dy]) for p in lattice])
                ang = np.zeros(12)
                
                # Check separation
                for i in range(12):
                    for j in range(i+1, 12):
                        if not _separation_sat(c[i], ang[i], c[j], ang[j], 1e-7):
                            return 1e10
                
                all_verts = np.zeros((72, 2))
                for i in range(12):
                    all_verts[6*i:6*(i+1)] = _hex_vertices(c[i, 0], c[i, 1], 1.0, ang[i])
                
                best_local = 1e10
                for rot in np.linspace(0, 60, 13):
                    s, _, _ = _compute_min_outer(all_verts, rot)
                    best_local = min(best_local, s)
                
                return best_local
            
            try:
                x0 = [1.0, 0.0, 0.0, 0.0, 0.0]
                res = minimize(obj2, x0, method='Nelder-Mead',
                              options={'maxiter': 60, 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < 1e9:
                    scale, rot_e1, rot_e2, dx, dy = res.x
                    e1_v = e1_init * scale
                    e2_v = e2_init * scale
                    theta1 = np.radians(rot_e1)
                    theta2 = np.radians(rot_e2)
                    R1 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])
                    R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])
                    e1_v = R1 @ e1_v
                    e2_v = R2 @ e2_v
                    
                    c_opt = np.array([p[0]*e1_v + p[1]*e2_v + np.array([dx, dy]) for p in lattice])
                    ang_opt = np.zeros(12)
                    
                    for rot in np.linspace(0, 60, 61):
                        if time.time() > deadline - 20:
                            break
                        try_config(c_opt, ang_opt, rot)
            except Exception:
                pass
    
    # Fallback: ensure we have a valid result
    if best_inner is None:
        centers, angles = _make_lattice_config(lattice1, e1_std * 1.001, e2_std * 1.001)
        s, ox, oy = _compute_min_outer(
            np.vstack([_hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i]) for i in range(12)]), 0)
        best_s = s
        best_inner = np.column_stack([centers, angles])
        best_outer = np.array([ox, oy, 0.0])
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
