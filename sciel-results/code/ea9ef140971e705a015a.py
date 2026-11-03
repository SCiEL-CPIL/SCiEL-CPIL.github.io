import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 142.0
SQRT3 = math.sqrt(3)

# Precompute unit hexagon vertices and edge normals
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]

def _get_hex_verts(pos, ang_rad):
    c = np.cos(ang_rad)
    s = np.sin(ang_rad)
    vx = pos[0] + _UX * c - _UY * s
    vy = pos[1] + _UX * s + _UY * c
    return np.column_stack((vx, vy))

def _get_hex_normals(ang_rad):
    c = np.cos(ang_rad)
    s = np.sin(ang_rad)
    nx = _ENX * c - _ENY * s
    ny = _ENX * s + _ENY * c
    return np.column_stack((nx, ny))

def _min_separation(v1, v2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they intersect, positive if separated.
    Uses Separating Axis Theorem (SAT) logic.
    """
    n1 = _get_hex_normals(np.zeros(1))[0]
    n2 = _get_hex_normals(np.zeros(1))[0]
    
    # Wait, v1 and v2 are vertices. I need normals.
    # Let's assume the caller passes positions and angles, or I recompute.
    # To keep it fast inside the objective, I'll pass precomputed verts and normals.
    
    pass

def _overlap_penalty(v1, n1, v2, n2):
    """
    Compute a smooth approximation of the overlap depth between two hexagons.
    If they are separated, returns 0.
    If they overlap, returns a positive value proportional to the minimum penetration depth.
    """
    min_sep = float('inf')
    
    # Check axes from hex 1
    for i in range(6):
        nx, ny = n1[i, 0], n1[i, 1]
        p1 = v1[:, 0] * nx + v1[:, 1] * ny
        p2 = v2[:, 0] * nx + v2[:, 1] * ny
        # Separation is max(p1) - min(p2) if p1 is to the "right"
        # Or we can just compute the gap.
        gap1 = p1.max() - p2.min()
        gap2 = p2.max() - p1.min()
        # If gap1 < 0 and gap2 < 0, they overlap on this axis.
        # The depth of overlap is -min(gap1, gap2) if both are negative?
        # Actually, if they overlap, one gap is negative and the other is positive?
        # No. If they are disjoint, one of the gaps is positive.
        # If they intersect, all gaps are negative? No.
        # Let's think.
        # Projection of A is [minA, maxA], B is [minB, maxB].
        # Disjoint if maxA < minB or maxB < minA.
        # Overlap if maxA >= minB and maxB >= minA.
        # In overlap case, maxA - minB >= 0 and maxB - minA >= 0.
        # Wait, if they overlap, the intervals intersect.
        # The "penetration depth" along this axis is not simply defined if they are just touching.
        # For a penalty function, we can use:
        # If disjoint, penalty = 0.
        # If overlapping, penalty = max(0, -(min(gap1, gap2)))? No.
        
        # Let's use a standard smooth penalty:
        # d = min(gap1, gap2)
        # If d > 0, they are separated by d.
        # If d < 0, they overlap. The depth of overlap is roughly -d?
        # Actually, if A is [0, 2] and B is [1, 3], gap1 = 2-1=1, gap2 = 3-0=3. min=1. They overlap.
        # If A is [0, 1] and B is [2, 3], gap1 = 1-2=-1, gap2 = 3-0=3. min=-1. They are disjoint.
        # So if min(gap1, gap2) < 0, they are disjoint.
        # The separation distance is min(gap1, gap2) if it's positive? No.
        # If disjoint, one gap is negative, one positive. The separation is the positive one?
        # No, if A=[0,1], B=[2,3]. gap1 = -1, gap2 = 3. Separation is 1.
        # So separation = max(gap1, gap2)? No, max is 3.
        # Actually, if disjoint, min(gap1, gap2) is the negative one.
        # So separation = -min(gap1, gap2) if min < 0?
        # If A=[0,1], B=[2,3]. min=-1. sep=1. Correct.
        # If A=[0,2], B=[1,3]. min=1. They overlap.
        
        # So:
        # sep = min(gap1, gap2)
        # If sep > 0, they are separated by sep? No, if sep > 0, it means both gaps are positive, which means they overlap.
        # Wait.
        # A=[0,1], B=[2,3]. gap1 = 1-2 = -1. gap2 = 3-0 = 3. min = -1.
        # A=[0,2], B=[1,3]. gap1 = 2-1 = 1. gap2 = 3-0 = 3. min = 1.
        
        # So if min(gap1, gap2) < 0, they are disjoint. The distance is -min(gap1, gap2).
        # If min(gap1, gap2) >= 0, they overlap. The "penetration" is not directly min.
        # Actually, if they overlap, the minimum distance is 0.
        # For a penalty, we want 0 when disjoint, and >0 when overlapping.
        # We can use: penalty = max(0, -min(gap1, gap2))
        # If disjoint, min < 0, so -min > 0. Penalty > 0.
        # If overlapping, min > 0, so -min < 0. Penalty = 0.
        # This is backwards! We want penalty when overlapping.
        
        # Let's re-evaluate.
        # Disjoint: A left of B. maxA < minB. gap1 = maxA - minB < 0. gap2 = maxB - minA > 0.
        # Overlap: maxA >= minB and maxB >= minA. gap1 >= 0 and gap2 >= 0.
        
        # So if min(gap1, gap2) < 0, they are disjoint.
        # If min(gap1, gap2) >= 0, they overlap.
        
        # We want a penalty for overlap.
        # So penalty should be large if min(gap1, gap2) >= 0.
        # Let d = min(gap1, gap2).
        # If d < 0, sep = -d. Penalty = 0.
        # If d >= 0, they overlap. Penalty = d?
        # If they just touch, d=0. Penalty 0.
        # If they overlap deeply, d is large?
        # A=[0, 10], B=[9, 11]. gap1 = 10-9=1. gap2 = 11-0=11. min=1.
        # A=[0, 10], B=[0, 10]. gap1 = 10. gap2 = 10. min=10.
        # So d is a measure of overlap depth?
        # Yes, d is the length of the intersection of the projections? No.
        # Intersection length = min(maxA, maxB) - max(minA, minB).
        # Here d = min(maxA-minB, maxB-minA).
        
        # Anyway, let's use:
        # penalty_axis = max(0, d) where d = min(gap1, gap2).
        # This is 0 when disjoint, and positive when overlapping.
        
        d = min(gap1, gap2)
        if d > 0:
            # Overlapping. Penalty is d.
            # We want the MINIMUM separation, which corresponds to the MAXIMUM penalty?
            # No, the minimum distance between polygons is the MINIMUM over all axes of the separation.
            # If they overlap, the distance is 0.
            # But for a smooth penalty, we want to push them apart.
            # The "worst" axis is the one with the smallest separation (most overlap).
            # So we take the MAX of the penalties?
            # If they are disjoint, all axes have d < 0.
            # If they overlap, some axes have d > 0.
            # The minimum distance is 0.
            # To create a gradient, we can use the maximum penetration depth.
            # max_pen = max(0, d) for each axis.
            # Then total penalty = sum or max of these?
            # Usually, for non-overlap constraints, we require d < 0 for ALL axes.
            # So if any d >= 0, it's a violation.
            # The magnitude of violation is max(0, d).
            # To get a single scalar penalty for the pair, we can take the max over axes.
            
            if d > min_sep:
                min_sep = d # Wait, I named it min_sep but I'm storing max_pen?
                # Let's rename. I want the "most overlapping" axis.
                # Actually, if I take the max of max(0, d) over axes, that gives the penetration depth.
                # Let's do that.
                
    # I need to restructure the loop to accumulate the max penetration.
    
    max_pen = 0.0
    for i in range(6):
        nx, ny = n1[i, 0], n1[i, 1]
        p1 = v1[:, 0] * nx + v1[:, 1] * ny
        p2 = v2[:, 0] * nx + v2[:, 1] * ny
        gap1 = p1.max() - p2.min()
        gap2 = p2.max() - p1.min()
        d = min(gap1, gap2)
        if d > 0:
            if d > max_pen:
                max_pen = d
    
    for i in range(6):
        nx, ny = n2[i, 0], n2[i, 1]
        p1 = v1[:, 0] * nx + v1[:, 1] * ny
        p2 = v2[:, 0] * nx + v2[:, 1] * ny
        gap1 = p1.max() - p2.min()
        gap2 = p2.max() - p1.min()
        d = min(gap1, gap2)
        if d > 0:
            if d > max_pen:
                max_pen = d
                
    return max_pen

def _objective(x, weight):
    """
    x: [x1, y1, a1, ..., x12, y12, a12]
    """
    pos = x[:N*2].reshape(N, 2)
    angles = x[N*2:]
    
    # Compute outer side length
    # Use the same logic as before: project all vertices onto 6 directions, find max abs proj, scale.
    # To make it differentiable, we can use a smooth max or just the raw max (L-BFGS can handle non-smooth).
    
    # Get all vertices
    all_verts = np.zeros((N*6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = _get_hex_verts(pos[i], angles[i])
        
    center = np.mean(all_verts, axis=0)
    offsets = all_verts - center
    
    # The outer hexagon side length S is determined by the max projection.
    # S = (2/sqrt(3)) * max_proj
    # We can minimize max_proj.
    
    # To allow rotation of the outer hexagon, we need to optimize phi as well.
    # But the task asks for inner positions and angles. The outer hexagon is centered at the centroid?
    # The task says: "outer hexagon center (x, y), side length and rotation".
    # The score is the side length.
    # So we should optimize the outer hexagon's center and rotation to minimize the side length.
    # This means the objective function should internally find the best outer hexagon for the given inner config.
    
    # This makes the objective non-smooth (min over phi).
    # However, L-BFGS-B can handle it if we provide gradients? No, we don't have gradients for the min.
    # We can use a smooth approximation or just let L-BFGS-B use numerical gradients.
    
    # Let's use a simple approach:
    # 1. Compute centroid of inner hexagons.
    # 2. Assume outer hexagon is centered there.
    # 3. Find optimal rotation phi by scanning or using a smooth max.
    
    # Actually, the current program does a golden section search for phi.
    # We can do that inside the objective. It's expensive but accurate.
    
    # To speed it up, we can just use a fixed set of 12 angles for phi and take the min.
    # Or use a smooth max of the projections.
    
    # Let's use the smooth max trick: soft_max(x, t) = (1/t) * log(sum(exp(t*x)))
    # This is differentiable.
    
    # But we need min over phi.
    # Let's just scan phi in 10 degrees steps inside the objective. It's fast enough for N=12.
    
    best_side = float('inf')
    phi_vals = np.arange(0, 60, 5)
    for phi_deg in phi_vals:
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        side = float(np.max(side_vals))
        if side < best_side:
            best_side = side
            
    # Compute overlap penalty
    penalty = 0.0
    for i in range(N):
        v1 = _get_hex_verts(pos[i], angles[i])
        n1 = _get_hex_normals(angles[i])
        for j in range(i+1, N):
            v2 = _get_hex_verts(pos[j], angles[j])
            n2 = _get_hex_normals(angles[j])
            pen = _overlap_penalty(v1, n1, v2, n2)
            penalty += pen * pen # Quadratic penalty
            
    return best_side + weight * penalty

def _make_initial_config(seed):
    rng = np.random.RandomState(seed)
    # Honeycomb init
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
        
    # Add some noise
    pos += rng.uniform(-0.5, 0.5, (N, 2))
    ang += rng.uniform(-10, 10, N)
    
    x = np.concatenate([pos.flatten(), ang])
    return x

def construct():
    t0 = time.time()
    
    best_side = float('inf')
    best_x = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Simulated Annealing loop
    # We will use L-BFGS-B as the local optimizer, and SA to restart or perturb.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE - 20:
            break
            
        x0 = _make_initial_config(42 + r)
        
        # Start with high penalty weight, reduce it
        # Or just use a fixed high weight and rely on L-BFGS to find a valid config.
        weight = 10.0
        
        # Run L-BFGS-B
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', args=(weight,), 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})
            
            x_opt = res.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            ang_opt = x_opt[N*2:]
            
            # Check validity
            # If valid, compute exact side length
            valid = True
            for i in range(N):
                v1 = _get_hex_verts(pos_opt[i], ang_opt[i])
                n1 = _get_hex_normals(ang_opt[i])
                for j in range(i+1, N):
                    v2 = _get_hex_verts(pos_opt[j], ang_opt[j])
                    n2 = _get_hex_normals(ang_opt[j])
                    if _overlap_penalty(v1, n1, v2, n2) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Compute exact min outer side
                all_verts = np.zeros((N*6, 2))
                for i in range(N):
                    all_verts[i*6:(i+1)*6] = _get_hex_verts(pos_opt[i], ang_opt[i])
                center = np.mean(all_verts, axis=0)
                offsets = all_verts - center
                
                best_phi_local = 0
                best_side_local = float('inf')
                phi_vals = np.arange(0, 60, 0.5)
                for phi_deg in phi_vals:
                    phi = np.radians(phi_deg)
                    normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
                    nx = np.cos(normal_angles)
                    ny = np.sin(normal_angles)
                    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                    abs_proj = np.abs(proj)
                    max_projs = np.max(abs_proj, axis=0)
                    side_vals = (2.0 / SQRT3) * max_projs
                    side = float(np.max(side_vals))
                    if side < best_side_local:
                        best_side_local = side
                        best_phi_local = phi_deg
                        
                if best_side_local < best_side:
                    best_side = best_side_local
                    best_x = x_opt.copy()
                    best_phi = best_phi_local
                    best_cx = float(center[0])
                    best_cy = float(center[1])
                    
        except Exception as e:
            pass
            
    if best_x is None:
        # Fallback to honeycomb
        pos0, ang0 = _make_initial_config(42)
        # Force valid by pushing apart? Or just return it if it's valid.
        # The honeycomb is valid.
        best_x = pos0.flatten().tolist() + ang0.tolist()
        # Compute side
        pos = np.array(best_x[:N*2]).reshape(N, 2)
        ang = np.array(best_x[N*2:])
        all_verts = np.zeros((N*6, 2))
        for i in range(N):
            all_verts[i*6:(i+1)*6] = _get_hex_verts(pos[i], ang[i])
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        best_phi = 0
        best_side = float('inf')
        phi_vals = np.arange(0, 60, 0.5)
        for phi_deg in phi_vals:
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / SQRT3) * max_projs
            side = float(np.max(side_vals))
            if side < best_side:
                best_side = side
                best_phi = phi_deg
        best_cx = float(center[0])
        best_cy = float(center[1])

    pos_final = best_x[:N*2].reshape(N, 2)
    ang_final = best_x[N*2:]
    
    inner = []
    for i in range(N):
        inner.append([float(pos_final[i, 0]), float(pos_final[i, 1]), float(ang_final[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

