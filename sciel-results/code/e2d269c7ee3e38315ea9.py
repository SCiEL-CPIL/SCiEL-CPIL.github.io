import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

# Precompute unit hexagon data
_HEX_ANGLES = np.arange(6) * np.pi / 3
_UNIT_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_NORMAL_ANGLES = _HEX_ANGLES + np.pi / 6
_UNIT_NORMALS = np.column_stack((np.cos(_NORMAL_ANGLES), np.sin(_NORMAL_ANGLES)))


def _rotate(verts, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return verts @ R.T


def _hex_verts(x, y, theta):
    return _rotate(_UNIT_VERTS, theta) + np.array([x, y])


def _hex_normals(theta):
    return _rotate(_UNIT_NORMALS, theta)


def _sat_disjoint_pair(v1, n1, v2, n2):
    """Check if two hexagons are disjoint using SAT. Returns True if disjoint."""
    all_n = np.vstack((n1, n2))
    proj1 = v1 @ all_n.T
    proj2 = v2 @ all_n.T
    min1, max1 = proj1.min(axis=0), proj1.max(axis=0)
    min2, max2 = proj2.min(axis=0), proj2.max(axis=0)
    return np.any((max1 < min2) | (max2 < min1))


def _compute_side(params):
    """Compute minimum outer side length for given inner hexagons (outer fixed at origin, fixed orientation)."""
    max_extent = 0.0
    for i in range(N):
        x, y, theta = params[3*i], params[3*i+1], params[3*i+2]
        verts = _hex_verts(x, y, theta)
        proj = verts @ _UNIT_NORMALS.T
        ext = np.max(np.abs(proj), axis=0)
        local_max = ext.max()
        if local_max > max_extent:
            max_extent = local_max
    return 2.0 * max_extent / SQRT3


def _all_disjoint(params):
    """Check all pairs are disjoint. Returns True if all disjoint."""
    verts_list = []
    normals_list = []
    for i in range(N):
        x, y, theta = params[3*i], params[3*i+1], params[3*i+2]
        verts_list.append(_hex_verts(x, y, theta))
        normals_list.append(_hex_normals(theta))
    for i in range(N):
        for j in range(i+1, N):
            if not _sat_disjoint_pair(verts_list[i], normals_list[i],
                                      verts_list[j], normals_list[j]):
                return False
    return True


def _objective(params):
    """Objective: outer side length if valid, else large penalty."""
    # Quick distance check for speed
    for i in range(N):
        xi, yi = params[3*i], params[3*i+1]
        for j in range(i+1, N):
            xj, yj = params[3*j], params[3*j+1]
            dx, dy = xi - xj, yi - yj
            if dx*dx + dy*dy < 2.5:  # Too close, likely overlap
                # Do full SAT check
                vi = _hex_verts(xi, yi, params[3*i+2])
                ni = _hex_normals(params[3*i+2])
                vj = _hex_verts(xj, yj, params[3*j+2])
                nj = _hex_normals(params[3*j+2])
                if not _sat_disjoint_pair(vi, ni, vj, nj):
                    return 1e6
    return _compute_side(params)


def _flower_initial():
    """Flower pattern: 1 center + 6 ring1 + 5 ring2."""
    params = np.zeros(3*N)
    params[0:3] = [0, 0, 0]
    for i in range(6):
        theta = i * np.pi / 3
        d = 2.05
        idx = 1 + i
        params[3*idx:3*idx+3] = [d*np.cos(theta), d*np.sin(theta), 0]
    for i in range(5):
        theta = i * 2*np.pi/5 + np.pi/5
        d = 4.05
        idx = 7 + i
        params[3*idx:3*idx+3] = [d*np.cos(theta), d*np.sin(theta), 0]
    return params


def _ring_initial():
    """Two concentric rings: 6 inner + 6 outer."""
    params = np.zeros(3*N)
    for i in range(6):
        theta = i * np.pi / 3
        d = 2.05
        params[3*i:3*i+3] = [d*np.cos(theta), d*np.sin(theta), 0]
    for i in range(6):
        theta = (i + 0.5) * np.pi / 3
        d = 4.1
        idx = 6 + i
        params[3*idx:3*idx+3] = [d*np.cos(theta), d*np.sin(theta), 0]
    return params


def _grid_initial():
    """3x4 grid."""
    params = np.zeros(3*N)
    for i in range(4):
        for j in range(3):
            idx = i*3 + j
            x = (i - 1.5) * 2.1
            y = (j - 1) * 1.9
            params[3*idx:3*idx+3] = [x, y, 0]
    return params


def _random_initial():
    """Random positions, zero angles."""
    params = np.random.rand(3*N) * 6 - 3
    params[2::3] = 0
    return params


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_params = None
    best_score = np.inf

    # Generate diverse initial guesses
    initial_guesses = [
        _flower_initial(),
        _ring_initial(),
        _grid_initial(),
        _random_initial(),
        _random_initial(),
        _random_initial(),
    ]

    # Add perturbed versions of the first few
    for base in initial_guesses[:3]:
        initial_guesses.append(base + np.random.normal(0, 0.3, 3*N))

    for guess in initial_guesses:
        if time.time() > deadline:
            break

        # Nelder-Mead optimization
        try:
            res = minimize(_objective, guess, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

        # If we found a good solution, do local polish
        if best_params is not None and best_score < 10 and time.time() < deadline - 15:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                perturbed = best_params + np.random.normal(0, 0.15, 3*N)
                try:
                    res2 = minimize(_objective, perturbed, method='Nelder-Mead',
                                   options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-5})
                    if res2.fun < best_score:
                        best_score = res2.fun
                        best_params = res2.x.copy()
                except Exception:
                    pass

    if best_params is None:
        # Fallback
        best_params = _flower_initial()
        best_score = _objective(best_params)

    # Final validation and cleanup
    # Ensure angles are in reasonable range
    for i in range(N):
        theta = best_params[3*i+2]
        # Normalize to [-pi, pi]
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        best_params[3*i+2] = theta

    # Extract inner hexagons
    inner = []
    for i in range(N):
        x, y, theta = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
        inner.append([float(x), float(y), float(np.degrees(theta))])

    center = (0.0, 0.0)
    side = float(best_score)
    angle = 0.0

    return inner, center, side, angle
