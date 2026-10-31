"""Structural change: vectorized objective + multi-start Nelder-Mead with
better hex-grid initializations and adaptive outer-rotation search."""
N = 11
import numpy as np
import math
import time
from scipy.optimize import minimize

# ---- geometry helpers (vectorized) ---------------------------------------

def _hex_verts_batch(positions, angles, s=1.0):
    """positions: (N,2), angles: (N,) radians -> (N,6,2) vertices"""
    n = len(positions)
    base_angles = np.pi/6 + np.arange(6)*np.pi/3  # (6,)
    rot = angles[:, None] + base_angles[None, :]   # (N,6)
    cx = positions[:, 0:1]
    cy = positions[:, 1:2]
    vx = cx + s * np.cos(rot)
    vy = cy + s * np.sin(rot)
    return np.stack([vx, vy], axis=-1)

def _required_side(positions, angles_rad, phi_rad, center=(0.0, 0.0)):
    """Exact required side length of outer hex with rotation phi centered at center."""
    verts = _hex_verts_batch(np.array(positions), np.array(angles_rad)) - np.array(center)
    thetas = phi_rad + np.array([(2*k+1)*np.pi/6 for k in range(6)])  # (6,)
    cos_t = np.cos(thetas)  # (6,)
    sin_t = np.sin(thetas)  # (6,)
    # proj[k, i] = verts[i,0]*cos_t[k] + verts[i,1]*sin_t[k]
    proj = verts[:, 0] * cos_t[None, :] + verts[:, 1] * sin_t[None, :]  # (N*6, 6)
    max_proj = np.max(proj, axis=0)  # (6,)
    return np.max(max_proj) / (math.sqrt(3)/2)

def _smooth_side(positions, angles_rad, phi_rad, center=(0.0, 0.0), beta=30.0):
    """Smooth surrogate using log-sum-exp for both max over vertices and max over directions."""
    verts = _hex_verts_batch(np.array(positions), np.array(angles_rad)) - np.array(center)
    thetas = phi_rad + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    proj = verts[:, 0] * cos_t[None, :] + verts[:, 1] * sin_t[None, :]  # (N*6, 6)
    # log-sum-exp over vertices for each direction
    m1 = np.max(proj, axis=0)  # (6,)
    lse_dir = m1 + (1.0/beta) * np.log(np.sum(np.exp(proj - m1[None, :]), axis=0))
    # log-sum-exp over directions
    m2 = np.max(lse_dir)
    lse_final = m2 + (1.0/beta) * math.log(math.exp(beta * (lse_dir - m2)).sum())
    return lse_final / (math.sqrt(3)/2)

# ---- SAT overlap check (pairwise, using vertex/edge normals) --------------

def _sat_gap_fast(p1, a1_rad, p2, a2_rad, s=1.0):
    """Return minimum gap (negative = overlap) between two unit hexagons."""
    v1 = p1[0] + s * np.cos(a1_rad + np.pi/6 + np.arange(6)*np.pi/3)
    v1y = p1[1] + s * np.sin(a1_rad + np.pi/6 + np.arange(6)*np.pi/3)
    v2 = p2[0] + s * np.cos(a2_rad + np.pi/6 + np.arange(6)*np.pi/3)
    v2y = p2[1] + s * np.sin(a2_rad + np.pi/6 + np.arange(6)*np.pi/3)
    
    min_gap = float('inf')
    for i in range(6):
        for (vx, vy) in [(v1, v1y), (v2, v2y)]:
            j = (i+1) % 6
            ex = vx[j] - vx[i]
            ey = vy[j] - vy[i]
            nrm = math.sqrt(ex*ex + ey*ey)
            if nrm < 1e-12:
                continue
            nx = -ey / nrm
            ny = ex / nrm
            proj1x = v1 * nx + v1y * ny
            proj2x = v2 * nx + v2y * ny
            gap = min(proj1x.max() - proj2x.min(), proj2x.max() - proj1x.min())
            if gap < min_gap:
                min_gap = gap
    return min_gap

def _check_disjoint(positions, angles_rad, tol=1e-9):
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if _sat_gap_fast(positions[i], angles_rad[i], positions[j], angles_rad[j]) < -tol:
                return False
    return True

# ---- Objective for optimizer ---------------------------------------------

def _objective(x, beta=30.0):
    """
    x = [x0, y0, a0_deg, x1, y1, a1_deg, ..., x10, y10, a10_deg, cx, cy, phi_deg]
    """
    positions = np.array([(x[3*i], x[3*i+1]) for i in range(N)])
    angles_deg = np.array([x[3*i+2] for i in range(N)])
    angles_rad = np.deg2rad(angles_deg)
    cx, cy = x[3*N], x[3*N+1]
    phi_deg = x[3*N+2]
    phi_rad = math.radians(phi_deg)
    
    side = _smooth_side(positions, angles_rad, phi_rad, (cx, cy), beta)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            gap = _sat_gap_fast(positions[i], angles_rad[i], positions[j], angles_rad[j])
            if gap < 0:
                penalty += gap * gap * 50.0
    return side + penalty

# ---- Initial configurations ----------------------------------------------

def _axial_to_xy(q, r):
    s3 = math.sqrt(3)
    return (1.5 * q, s3 * (r + q / 2.0))

def _build_init(coords):
    x = np.zeros(3*N + 3)
    for i, (q, r) in enumerate(coords):
        px, py = _axial_to_xy(q, r)
        x[3*i] = px
        x[3*i+1] = py
        x[3*i+2] = 0.0
    x[3*N] = 0.0
    x[3*N+1] = 0.0
    x[3*N+2] = 0.0
    return x

def _get_inits():
    inits = []
    # 3-4-3 cluster (center + ring1 + 3 ring2)
    c343 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
            (2,0),(1,1),(-1,2)]
    inits.append(_build_init(c343))
    
    # Alternative ring-2 selections
    c_alt1 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
              (2,0),(0,2),(-2,2)]
    inits.append(_build_init(c_alt1))
    
    c_alt2 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
              (2,0),(-2,0),(0,-2)]
    inits.append(_build_init(c_alt2))
    
    c_alt3 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
              (1,1),(-1,2),(1,-2)]
    inits.append(_build_init(c_alt3))
    
    # Random perturbations
    rng = np.random.RandomState(12345)
    base = inits[0].copy()
    for _ in range(5):
        x = base.copy()
        x[:3*N] += rng.randn(3*N) * 0.25
        inits.append(x)
    
    # Linear chain as fallback-ish init
    x_lin = np.zeros(3*N + 3)
    s3 = math.sqrt(3)
    for i in range(N):
        x_lin[3*i] = 2.0 * (i - N//2)
        x_lin[3*i+1] = 0.0
        x_lin[3*i+2] = 0.0
    inits.append(x_lin)
    
    return inits

# ---- Main construct -------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin
    
    best_side = float('inf')
    best_x = None
    
    inits = _get_inits()
    
    for x0 in inits:
        if time.time() > deadline - 15.0:
            break
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7,
                                   'adaptive': True})
        except Exception:
            continue
        
        x_opt = res.x
        positions = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
        angles_deg = [x_opt[3*i+2] for i in range(N)]
        angles_rad = [math.radians(a) for a in angles_deg]
        cx, cy = x_opt[3*N], x_opt[3*N+1]
        phi_deg = x_opt[3*N+2]
        phi_rad = math.radians(phi_deg)
        
        # Check validity
        if not _check_disjoint(positions, angles_rad, tol=1e-7):
            continue
        
        side = _required_side(positions, angles_rad, phi_rad, (cx, cy))
        
        if side < best_side:
            best_side = side
            best_x = x_opt
    
    # If no valid result from optimization, use the 3-4-3 grid directly
    if best_x is None:
        x0 = inits[0]
        positions = [(x0[3*i], x0[3*i+1]) for i in range(N)]
        angles_deg = [0.0]*N
        angles_rad = [0.0]*N
        best_side = _required_side(positions, angles_rad, 0.0, (0.0, 0.0))
        best_x = x0
        phi_deg = 0.0
    
    # Extract final result
    positions = [(best_x[3*i], best_x[3*i+1]) for i in range(N)]
    angles_deg = [best_x[3*i+2] for i in range(N)]
    angles_rad = [math.radians(a) for a in angles_deg]
    cx, cy = best_x[3*N], best_x[3*N+1]
    phi_deg = best_x[3*N+2]
    phi_rad = math.radians(phi_deg)
    
    # Shift so outer center is at origin
    positions = [(p[0]-cx, p[1]-cy) for p in positions]
    
    # Recompute side in final frame
    final_side = _required_side(positions, angles_rad, phi_rad, (0.0, 0.0))
    
    # Final verification
    if not _check_disjoint(positions, angles_rad, tol=1e-7):
        # Fallback to linear arrangement
        positions = [(2.0*k, 0.0) for k in range(-(N//2), N-N//2)]
        angles_deg = [0.0]*N
        angles_rad = [0.0]*N
        phi_rad = 0.0
        phi_deg = 0.0
        final_side = _required_side(positions, angles_rad, phi_rad, (0.0, 0.0))
    
    final_side *= (1.0 + 1e-9)
    
    inner = [[positions[i][0], positions[i][1], angles_deg[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, phi_deg
