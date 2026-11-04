import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 135.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + rot_deg)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _sat_overlap(v1, v2, n1, n2):
    """SAT overlap: returns separation (positive=separated, negative=overlapping)."""
    min_overlap = float('inf')
    all_normals = np.vstack((n1, n2))
    for n in all_normals:
        proj1 = v1 @ n
        min1, max1 = proj1.min(), proj1.max()
        proj2 = v2 @ n
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    return -min_overlap

def _outer_side_needed(points, outer_rot_deg):
    normals = _hex_edge_normals(outer_rot_deg)
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner):
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        cx, cy, ang = inner[i]
        v = _hex_vertices(cx, cy, 1.0, ang)
        nrm = _hex_edge_normals(ang)
        verts.append(v)
        norms.append(nrm)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(verts[i], verts[j], norms[i], norms[j]) < -1e-9:
                return False
    return True

def _compute_outer(inner, outer_center, outer_angle):
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    return _outer_side_needed(all_pts, outer_angle)

def _make_hex_ring():
    """Classic hexagonal ring: 1 center + 6 around + 5 outer = 12."""
    pts = []
    s3 = np.sqrt(3)
    # Center
    pts.append([0.0, 0.0, 0.0])
    # Inner ring (6 hexagons at distance sqrt(3))
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([s3 * np.cos(ang), s3 * np.sin(ang), 0.0])
    # Outer ring (5 hexagons, skip one position)
    for k in range(5):
        ang = np.radians(60 * k + 30)
        d = 2 * s3
        pts.append([d * np.cos(ang), d * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts

def _make_hex_grid(rows, cols, rot=0.0, gap=0.05):
    pts = []
    sx = 2.0 + gap
    sy = np.sqrt(3) + gap * 0.5
    for r in range(rows):
        y = r * sy
        for c in range(cols):
            x = c * sx
            if r % 2 == 1:
                x += sx / 2.0
            pts.append([x, y, rot])
    pts = np.array(pts[:12])
    if len(pts) < 12:
        while len(pts) < 12:
            pts = np.vstack([pts, pts[-1] + [0.5, 0.5, 0.0]])
    pts = pts[:12]
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts

def _make_compact_cluster():
    """Tightly packed cluster."""
    pts = []
    s3 = np.sqrt(3)
    # 3-3-3-3 grid
    for r in range(4):
        for c in range(3):
            x = c * 2.0
            y = r * s3
            if r % 2 == 1:
                x += 1.0
            pts.append([x, y, 0.0])
    pts = np.array(pts)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts

def _physics_relax(inner_init, n_iter=500, dt=0.01):
    """Physics-based relaxation: repulsive forces between overlapping hexagons."""
    inner = inner_init.copy()
    best_inner = inner.copy()
    best_score = float('inf')
    
    for it in range(n_iter):
        if time.time() - _t0 > DEADLINE - 10:
            break
        
        # Compute forces
        forces = np.zeros_like(inner)
        
        for i in range(N):
            for j in range(i + 1, N):
                vi = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                vj = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                ni = _hex_edge_normals(inner[i][2])
                nj = _hex_edge_normals(inner[j][2])
                
                sep = _sat_overlap(vi, vj, ni, nj)
                if sep < 0:
                    # Overlapping: push apart
                    overlap = -sep
                    # Direction: from j to i
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9:
                        dx, dy = 0.01, 0.01
                        dist = math.sqrt(dx*dx + dy*dy)
                    fx = (dx / dist) * overlap * 10.0
                    fy = (dy / dist) * overlap * 10.0
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
        
        # Apply forces with damping
        damping = 0.95
        for i in range(N):
            inner[i][0] += forces[i][0] * dt
            inner[i][1] += forces[i][1] * dt
        
        # Score: max outer side needed
        all_pts = np.vstack([_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        all_pts[:, 0] -= all_pts[:, 0].mean()
        all_pts[:, 1] -= all_pts[:, 1].mean()
        L = _outer_side_needed(all_pts, 0.0)
        
        if L < best_score:
            best_score = L
            best_inner = inner.copy()
        
        # Dampen forces over time
        dt *= 0.999
    
    return best_inner

def _objective(params):
    """Penalized objective for optimization."""
    if time.time() - _t0 > DEADLINE - 5:
        return 1e6
    
    inner_params = params[:36].reshape(12, 3)
    ox, oy, oang = params[36], params[37], params[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner_params[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nrm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nrm)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                penalty += (-d) ** 2 * 50.0
    
    # Outer side
    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    L = _outer_side_needed(all_pts, oang)
    
    return L + penalty

def _polish(params, method='L-BFGS-B'):
    """Polish a configuration using gradient-based optimization."""
    try:
        res = minimize(
            _objective,
            params,
            method=method,
            options={
                'maxiter': 300,
                'ftol': 1e-8,
                'gtol': 1e-6,
                'maxls': 50
            }
        )
        return res.x
    except Exception:
        return params

def _valid_and_score(params):
    """Check validity and compute score."""
    inner_params = params[:36].reshape(12, 3)
    ox, oy, oang = params[36], params[37], params[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner_params[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nrm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nrm)
    
    # Check disjoint
    for i in range(12):
        for j in range(i + 1, 12):
            if _sat_overlap(verts_list[i], verts_list[j], norms_list[i], norms_list[j]) < -1e-8:
                return None
    
    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    L = _outer_side_needed(all_pts, oang)
    return L

def construct():
    global _t0
    _t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - _t0)
    
    best_L = float('inf')
    best_params = None
    
    # Generate initial configurations
    configs = []
    configs.append(_make_hex_ring())
    configs.append(_make_hex_grid(4, 3, rot=0.0))
    configs.append(_make_hex_grid(4, 3, rot=30.0))
    configs.append(_make_hex_grid(3, 4, rot=0.0))
    configs.append(_make_hex_grid(3, 4, rot=30.0))
    configs.append(_make_compact_cluster())
    configs.append(_make_compact_cluster() * 0.95)  # Slightly compressed
    
    # Random configs
    np.random.seed(42)
    for seed in range(8):
        np.random.seed(seed)
        pts = np.random.randn(12, 3) * 2.5
        pts[:, 2] = np.random.choice([0, 30, 60, 90, 120, 150, 210, 240, 270, 300, 330], size=12)
        pts[:, 0] -= pts[:, 0].mean()
        pts[:, 1] -= pts[:, 1].mean()
        configs.append(pts)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 22.5]
    
    # Phase 1: Physics relaxation on each config
    for cfg in configs:
        if time_left() < 20:
            break
        relaxed = _physics_relax(cfg, n_iter=300, dt=0.02)
        for orot in outer_rots[:2]:
            params = np.concatenate((relaxed.flatten(), [0.0, 0.0, orot]))
            score = _valid_and_score(params)
            if score is not None and score < best_L:
                best_L = score
                best_params = params.copy()
    
    # Phase 2: Differential Evolution (global search)
    if time_left() > 40:
        # Define bounds
        bounds = []
        for i in range(12):
            bounds.extend([(-8, 8), (-8, 8), (0, 360)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        # Use best config as a starting point influence
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 0.9),
                recombination=0.7,
                seed=42,
                polish=True,
                init='sobol'
            )
            if de_result.success or _objective(de_result.x) < best_L:
                score = _valid_and_score(de_result.x)
                if score is not None and score < best_L:
                    best_L = score
                    best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B polish on best and several good candidates
    if best_params is not None and time_left() > 15:
        polished = _polish(best_params, method='L-BFGS-B')
        score = _valid_and_score(polished)
        if score is not None and score < best_L:
            best_L = score
            best_params = polished.copy()
    
    # Phase 4: Try more local optimizations from different starts
    if time_left() > 25:
        for cfg in configs[:6]:
            if time_left() < 10:
                break
            for orot in outer_rots:
                params = np.concatenate((cfg.flatten(), [0.0, 0.0, orot]))
                # Quick L-BFGS-B
                try:
                    res = minimize(
                        _objective,
                        params,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-8}
                    )
                    score = _valid_and_score(res.x)
                    if score is not None and score < best_L:
                        best_L = score
                        best_params = res.x.copy()
                except Exception:
                    pass
    
    # Phase 5: Final Nelder-Mead polish
    if best_params is not None and time_left() > 10:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            score = _valid_and_score(res.x)
            if score is not None and score < best_L:
                best_L = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Fallback
    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        best_L = _objective(best_params)
    
    # Extract result
    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    # Validate
    if not _check_disjoint(inner):
        # Fallback to safe config
        cfg = _make_hex_grid(4, 3, gap=0.8)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        outer_side = _compute_outer(inner, outer_center, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    outer_side = _compute_outer(inner, outer_center, outer_angle)
    outer_side *= 1.0001  # Small safety margin
    
    return inner, outer_center, outer_side, outer_angle
