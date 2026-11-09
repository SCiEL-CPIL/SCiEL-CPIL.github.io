"""Packing 12 unit hexagons in a minimal regular hexagon: structured lattice + local optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)
APO = SQRT3 / 2.0  # apothem of unit hexagon

# Unit flat-top hexagon vertices and outward unit edge-normals
_HEX_V = np.array([
    [1.0, 0.0], [0.5, APO], [-0.5, APO],
    [-1.0, 0.0], [-0.5, -APO], [0.5, -APO],
])
_HEX_N = np.array([
    [SQRT3/2, 0.5], [0.0, 1.0], [-SQRT3/2, 0.5],
    [-SQRT3/2, -0.5], [0.0, -1.0], [SQRT3/2, -0.5],
])
_OUTER_N = _HEX_N  # outer hexagon (flat-top) unit normals

_DEADLINE = 140.0
_T0 = [0.0]

def _elapsed():
    return time.time() - _T0[0]

def _compute(x, y, a, lam):
    """Return (S, overlap_penalty). x,y,a are (N,) arrays.
    S is the outer hexagon side length needed (with center at origin, rotation 0).
    """
    ca = np.cos(a); sa = np.sin(a)
    vx = _HEX_V[:, 0]; vy = _HEX_V[:, 1]
    # all vertices (N,6,2)
    all_vx = ca[:, None] * vx[None, :] - sa[:, None] * vy[None, :] + x[:, None]
    all_vy = sa[:, None] * vx[None, :] + ca[:, None] * vy[None, :] + y[:, None]
    # S: outer center origin, outer rotation 0
    on0 = _OUTER_N[:, 0]; on1 = _OUTER_N[:, 1]
    dots = all_vx[:, :, None] * on0[None, None, :] + all_vy[:, :, None] * on1[None, None, :]
    S = float(np.max(dots)) / APO
    if S < 0.0:
        S = 0.0
    if lam == 0.0:
        return S, 0.0
    # normals (N,6,2)
    nx = _HEX_N[:, 0]; ny = _HEX_N[:, 1]
    all_nx = ca[:, None] * nx[None, :] - sa[:, None] * ny[None, :]
    all_ny = sa[:, None] * nx[None, :] + ca[:, None] * ny[None, :]
    pen = 0.0
    for i in range(N):
        vi_x = all_vx[i]; vi_y = all_vy[i]
        ni_x = all_nx[i]; ni_y = all_ny[i]
        for j in range(i + 1, N):
            ax = np.empty(12); ay = np.empty(12)
            ax[:6] = ni_x; ay[:6] = ni_y
            ax[6:] = all_nx[j]; ay[6:] = all_ny[j]
            pi = vi_x[:, None] * ax[None, :] + vi_y[:, None] * ay[None, :]  # (6,12)
            pj = all_vx[j, :, None] * ax[None, :] + all_vy[j, :, None] * ay[None, :]
            min_i = pi.min(axis=0); max_i = pi.max(axis=0)
            min_j = pj.min(axis=0); max_j = pj.max(axis=0)
            sep = np.maximum(min_i - max_j, min_j - max_i).max()
            if sep < 0.0:
                d = -sep
                pen += d * d
    return S, lam * pen

def _objective(v, lam):
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]
    S, pen = _compute(x, y, a, lam)
    return S + pen

def _is_valid(x, y, a, tol=1e-6):
    S, raw_pen = _compute(x, y, a, 1.0)
    return raw_pen < tol, S

def _fallback():
    # guaranteed-valid: 2 rows x 6, wide spacing
    inner = []
    for r in range(2):
        for c in range(6):
            inner.append([3.2 * (c - 2.5), 2.9 * r, 0.0])
    return inner, (0.0, 0.0), 22.0, 0.0

def _make_hexagonal_lattice():
    """Create a hexagonal lattice arrangement of 12 hexagons."""
    # Hexagonal lattice: rows offset by half spacing
    # For flat-top hexagons, horizontal spacing is 2*sqrt(3)/2 * side = sqrt(3) for touching
    # Vertical spacing is 1.5 * side for touching
    # Let's use a more compact arrangement
    positions = []
    # 3 rows: 4, 4, 4
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * SQRT3 * 1.05  # slight expansion
            y = (r - 1) * 1.5 * 1.05
            positions.append([x, y, 0.0])
    return positions

def _make_grid_4x3():
    """4 columns x 3 rows grid."""
    positions = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 2.0
            y = (r - 1) * SQRT3
            positions.append([x, y, 0.0])
    return positions

def _make_circular():
    """Circular arrangement: 1 center + 5 inner + 6 outer."""
    positions = []
    # Center
    positions.append([0.0, 0.0, 0.0])
    # Inner ring: 5 hexagons at radius ~2
    for i in range(5):
        ang = 2 * np.pi * i / 5
        r = 2.2
        positions.append([r * np.cos(ang), r * np.sin(ang), 0.0])
    # Outer ring: 6 hexagons at radius ~4
    for i in range(6):
        ang = 2 * np.pi * i / 6
        r = 4.2
        positions.append([r * np.cos(ang), r * np.sin(ang), 0.0])
    return positions

def _make_two_rows():
    """2 rows of 6 hexagons."""
    positions = []
    for r in range(2):
        for c in range(6):
            x = (c - 2.5) * SQRT3 * 1.1
            y = (r - 0.5) * 1.5 * 1.1
            positions.append([x, y, 0.0])
    return positions

def construct():
    _T0[0] = time.time()
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    init_configs = [
        _make_hexagonal_lattice(),
        _make_grid_4x3(),
        _make_circular(),
        _make_two_rows(),
    ]
    
    # Add random perturbations to some configs
    for _ in range(3):
        base = rng.choice(init_configs)
        perturbed = []
        for p in base:
            perturbed.append([
                p[0] + rng.normal(0, 0.3),
                p[1] + rng.normal(0, 0.3),
                rng.uniform(-np.pi/6, np.pi/6)
            ])
        init_configs.append(perturbed)
    
    best = None
    best_f = np.inf
    best_S = np.inf
    
    # Local optimization from each initial config
    for idx, config in enumerate(init_configs):
        if _elapsed() > _DEADLINE - 5:
            break
        
        v0 = np.array([p[0] for p in config] + [p[1] for p in config] + [p[2] for p in config])
        
        try:
            # First pass: moderate penalty
            r1 = minimize(lambda v: _objective(v, 5.0), v0, method='Powell',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if r1.x is not None:
                v0 = r1.x.copy()
                f1 = r1.fun
                if f1 < best_f:
                    best_f = f1
                    best = v0.copy()
                    x = v0[0:N]; y = v0[N:2*N]; a = v0[2*N:3*N]
                    ok, S = _is_valid(x, y, a)
                    if ok and S < best_S:
                        best_S = S
            
            # Second pass: stronger penalty for refinement
            if _elapsed() > _DEADLINE - 3:
                break
            r2 = minimize(lambda v: _objective(v, 20.0), v0, method='Powell',
                          options={'maxiter': 80, 'xatol': 1e-7, 'fatol': 1e-9})
            if r2.x is not None:
                v0 = r2.x.copy()
                f2 = r2.fun
                if f2 < best_f:
                    best_f = f2
                    best = v0.copy()
                    x = v0[0:N]; y = v0[N:2*N]; a = v0[2*N:3*N]
                    ok, S = _is_valid(x, y, a)
                    if ok and S < best_S:
                        best_S = S
                        
        except Exception:
            pass
    
    if best is None:
        return _fallback()
    
    x = best[0:N]; y = best[N:2*N]; a = best[2*N:3*N]
    ok, S = _is_valid(x, y, a)
    
    if not ok:
        # Try one more high-penalty polish
        try:
            r3 = minimize(lambda v: _objective(v, 100.0), best, method='Powell',
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            if r3.x is not None:
                x = r3.x[0:N]; y = r3.x[N:2*N]; a = r3.x[2*N:3*N]
                ok, S = _is_valid(x, y, a)
        except Exception:
            pass
    
    if not ok:
        return _fallback()
    
    inner = [[float(x[i]), float(y[i]), float(np.degrees(a[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
