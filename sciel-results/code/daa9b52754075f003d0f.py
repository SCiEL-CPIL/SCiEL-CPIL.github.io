import numpy as np
import math
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # safety margin below 150s wall-clock
_SQRT3 = math.sqrt(3.0)
_HEX_ANGLES = np.radians(np.array([0, 60, 120, 180, 240, 300], dtype=np.float64))
# Outward normals for unit hexagon edges (flat-top orientation, rot=0)
# Edge from vertex i to vertex i+1, normal points outward
_OUTER_NORMAL_ANGLES = np.radians(np.array([30, 90, 150, 210, 270, 330], dtype=np.float64))


def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = _HEX_ANGLES + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))


def _hex_edge_normals(rot_deg):
    """Outward edge normals for a unit regular hexagon with given rotation."""
    base = _OUTER_NORMAL_ANGLES + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))


def _min_separation(v1, v2, n1, n2):
    """
    Compute minimum separation between two hexagons using SAT.
    Returns positive if separated, negative if overlapping.
    v1, v2: (6,2) vertex arrays
    n1, n2: (6,2) normal arrays
    """
    all_normals = np.vstack([n1, n2])
    min_pen = float('inf')
    for i in range(12):
        nrm = all_normals[i]
        proj1 = v1 @ nrm
        min1, max1 = proj1.min(), proj1.max()
        proj2 = v2 @ nrm
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2 or max2 < min1:
            return max(min2 - max1, min1 - max2)
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_pen:
            min_pen = overlap
    
    return -min_pen


def _outer_hex_side(pts, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330], dtype=np.float64) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = pts @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / _SQRT3) * max_proj


def _make_hex_grid(rows, cols, rot=0.0, gap=0.05):
    """Create a grid of hexagons with given rows and cols."""
    pts = []
    spacing_x = 2.0 + gap
    spacing_y = _SQRT3 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot])
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts


def _make_ring(gap=0.05):
    """Create a ring configuration: center + 6 around + 5 more."""
    pts = []
    d1 = _SQRT3 + gap
    d2 = 2 * _SQRT3 + gap * 1.2
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = math.radians(60 * k)
        pts.append([d1 * math.cos(ang), d1 * math.sin(ang), 0.0])
    for k in range(5):
        ang = math.radians(60 * k + 30)
        pts.append([d2 * math.cos(ang), d2 * math.sin(ang), 0.0])
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts


def _make_cluster(gap=0.05):
    """Create a compact cluster: 2 rows of 3 + 2 rows of 3 offset."""
    pts = []
    sx = 2.0 + gap
    sy = _SQRT3 + gap * 0.4
    positions = [
        (0, 0), (sx, 0), (2*sx, 0),
        (sx/2, sy), (3*sx/2, sy), (5*sx/2, sy),
        (0, 2*sy), (sx, 2*sy), (2*sx, 2*sy),
        (sx/2, 3*sy), (3*sx/2, 3*sy), (5*sx/2, 3*sy),
    ]
    for x, y in positions:
        pts.append([x, y, 0.0])
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts


def _make_strip(gap=0.05):
    """Create a strip: 2 rows of 6."""
    pts = []
    sx = 2.0 + gap
    sy = _SQRT3 + gap * 0.4
    for r in range(2):
        for c in range(6):
            x = c * sx
            y = r * sy
            if r % 2 == 1:
                x += sx / 2.0
            pts.append([x, y, 0.0])
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts


def _compute_forces(params, outer_rot):
    """
    Compute repulsive forces between hexagons and attractive force toward center.
    Returns forces array of shape (12, 2) and total penalty.
    """
    inner = params[:36].reshape(12, 3)
    ox, oy = params[36], params[37]
    
    # Compute vertices and normals for each hexagon
    verts = []
    norms = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        n = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(n)
    
    forces = np.zeros((12, 2))
    penalty = 0.0
    
    for i in range(12):
        for j in range(i + 1, 12):
            d = _min_separation(verts[i], verts[j], norms[i], norms[j])
            if d < 0:
                # Overlapping: strong repulsion
                overlap = -d
                penalty += overlap * overlap * 1000.0
                # Force direction: along the axis of minimum penetration
                # Approximate by pushing apart along center-to-center direction
                dx = inner[j, 0] - inner[i, 0]
                dy = inner[j, 1] - inner[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 1e-10:
                    fx = (dx / dist) * overlap * 50.0
                    fy = (dy / dist) * overlap * 50.0
                else:
                    # Perturb randomly if exactly overlapping
                    angle = (i * 137.5 + j * 73.1) % 360
                    fx = math.cos(math.radians(angle)) * overlap * 50.0
                    fy = math.sin(math.radians(angle)) * overlap * 50.0
                forces[i] += [-fx, -fy]
                forces[j] += [fx, fy]
    
    # Gentle attraction toward outer center to keep things compact
    for i in range(12):
        dx = ox - inner[i, 0]
        dy = oy - inner[i, 1]
        forces[i] += [dx * 0.1, dy * 0.1]
    
    return forces, penalty


def _force_relax(params, max_iters=500, tol=1e-6):
    """
    Apply force-based relaxation to reduce overlaps.
    Returns relaxed params and success flag.
    """
    params = params.copy()
    outer_rot = params[38]
    
    # Adaptive step size
    step = 0.05
    best_penalty = float('inf')
    best_params = params.copy()
    stall_count = 0
    
    for it in range(max_iters):
        forces, penalty = _compute_forces(params, outer_rot)
        
        if penalty < best_penalty:
            best_penalty = penalty
            best_params = params.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        if penalty < 1e-10:
            break
        
        if stall_count > 30:
            step *= 0.5
            if step < 1e-8:
                break
            stall_count = 0
        
        # Apply forces
        for i in range(12):
            params[i*3] += forces[i, 0] * step
            params[i*3+1] += forces[i, 1] * step
        
        # Update outer center to centroid
        params[36] = params[:36:3].mean()
        params[37] = params[1:36:3].mean()
    
    return best_params, best_penalty < 1e-8


def _local_optimize(params, time_budget=10.0):
    """
    Local optimization using Nelder-Mead on the smooth objective.
    """
    t0 = time.time()
    
    def objective(p):
        if time.time() - t0 > time_budget:
            return 1e6
        
        inner = p[:36].reshape(12, 3)
        ox, oy, oang = p[36], p[37], p[38]
        
        verts = []
        norms = []
        for i in range(12):
            cx, cy, rot = inner[i]
            v = _hex_vertices(cx, cy, 1.0, rot)
            n = _hex_edge_normals(rot)
            verts.append(v)
            norms.append(n)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                d = _min_separation(verts[i], verts[j], norms[i], norms[j])
                if d < 0:
                    penalty += (-d) ** 2 * 500.0
        
        all_pts = np.vstack(verts)
        all_pts[:, 0] -= ox
        all_pts[:, 1] -= oy
        L = _outer_hex_side(all_pts, oang)
        
        return L + penalty
    
    try:
        res = minimize(
            objective,
            params,
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        return res.x
    except Exception:
        return params


def _check_disjoint(inner):
    """Verify all hexagons are disjoint."""
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        v = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        nrm = _hex_edge_normals(inner[i][2])
        verts.append(v)
        norms.append(nrm)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_separation(verts[i], verts[j], norms[i], norms[j])
            if d < -1e-9:
                return False
    return True


def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Generate diverse initial configurations
    configs = []
    
    # Grid configurations with various gaps
    for gap in [0.0, 0.05, 0.1, 0.15]:
        configs.append(_make_hex_grid(4, 3, gap=gap))
        configs.append(_make_hex_grid(3, 4, gap=gap))
        configs.append(_make_hex_grid(2, 6, gap=gap))
        configs.append(_make_hex_grid(6, 2, gap=gap))
    
    # Ring and cluster configurations
    for gap in [0.0, 0.05, 0.1]:
        configs.append(_make_ring(gap=gap))
        configs.append(_make_cluster(gap=gap))
        configs.append(_make_strip(gap=gap))
    
    # Rotated variants
    base43 = _make_hex_grid(4, 3, gap=0.05)
    rotated43 = base43.copy()
    rotated43[:, 2] = 30.0
    configs.append(rotated43)
    
    base34 = _make_hex_grid(3, 4, gap=0.05)
    rotated34 = base34.copy()
    rotated34[:, 2] = 30.0
    configs.append(rotated34)
    
    # Deterministic random starts
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.randn(12, 3) * 2.5
        pts[:, 2] = np.random.choice([0, 30], size=12)
        pts[:, 0] -= pts[:, 0].mean()
        pts[:, 1] -= pts[:, 1].mean()
        configs.append(pts)
    
    best_L = float('inf')
    best_params = None
    
    outer_rots = [0.0, 30.0]
    
    for cfg in configs:
        if time_left() < 8:
            break
        
        for orot in outer_rots:
            if time_left() < 4:
                break
            
            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, orot]))
            
            # First: force relaxation to remove overlaps
            relaxed, success = _force_relax(init_params, max_iters=300)
            
            if success or relaxed is not None:
                # Check if relaxed config is valid
                inner_relax = relaxed[:36].reshape(12, 3)
                verts_relax = [_hex_vertices(inner_relax[i,0], inner_relax[i,1], 1.0, inner_relax[i,2]) for i in range(12)]
                norms_relax = [_hex_edge_normals(inner_relax[i,2]) for i in range(12)]
                
                valid = True
                for i in range(12):
                    for j in range(i + 1, 12):
                        if _min_separation(verts_relax[i], verts_relax[j], norms_relax[i], norms_relax[j]) < -1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    all_pts = np.vstack(verts_relax)
                    all_pts[:, 0] -= relaxed[36]
                    all_pts[:, 1] -= relaxed[37]
                    L_val = _outer_hex_side(all_pts, relaxed[38])
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = relaxed.copy()
                    
                    # Local optimization to refine
                    if time_left() > 5:
                        refined = _local_optimize(relaxed, time_budget=min(8.0, time_left() - 3))
                        inner_ref = refined[:36].reshape(12, 3)
                        verts_ref = [_hex_vertices(inner_ref[i,0], inner_ref[i,1], 1.0, inner_ref[i,2]) for i in range(12)]
                        norms_ref = [_hex_edge_normals(inner_ref[i,2]) for i in range(12)]
                        
                        valid_ref = True
                        for i in range(12):
                            for j in range(i + 1, 12):
                                if _min_separation(verts_ref[i], verts_ref[j], norms_ref[i], norms_ref[j]) < -1e-9:
                                    valid_ref = False
                                    break
                            if not valid_ref:
                                break
                        
                        if valid_ref:
                            all_pts_ref = np.vstack(verts_ref)
                            all_pts_ref[:, 0] -= refined[36]
                            all_pts_ref[:, 1] -= refined[37]
                            L_ref = _outer_hex_side(all_pts_ref, refined[38])
                            
                            if L_ref < best_L:
                                best_L = L_ref
                                best_params = refined.copy()
    
    # Fallback if nothing found
    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.3)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    # Extract final solution
    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    # Final verification
    if not _check_disjoint(inner):
        # Try to fix with force relaxation
        params = np.array([float(x) for x in np.concatenate((
            np.array(inner).flatten(),
            [outer_center[0], outer_center[1], outer_angle]
        ))])
        relaxed, _ = _force_relax(params, max_iters=200)
        inner_fixed = []
        for i in range(12):
            inner_fixed.append([float(relaxed[i*3]), float(relaxed[i*3+1]), float(relaxed[i*3+2])])
        
        if _check_disjoint(inner_fixed):
            inner = inner_fixed
            outer_center = (float(relaxed[36]), float(relaxed[37]))
            outer_angle = float(relaxed[38])
        else:
            # Ultimate fallback
            cfg = _make_hex_grid(4, 3, gap=0.5)
            inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
            outer_center = (0.0, 0.0)
            outer_angle = 0.0
    
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side(all_pts, outer_angle)
    
    # Small epsilon for numerical safety
    outer_side *= 1.000005
    
    return inner, outer_center, outer_side, outer_angle
