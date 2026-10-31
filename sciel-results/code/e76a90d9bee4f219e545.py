"""
Strategy: Rigid-body parameterized FCC truncated octahedron with 6D optimization.
Rationale: Instead of searching the full 114-dimensional atomic coordinate space via SA or basin-hopping,
we embed the known global minimum topology (FCC truncated octahedron) as a rigid body defined by
its symmetry operations. We then optimize only a small number of global parameters (scale, and possibly
small anisotropic distortions) using gradient descent. This drastically reduces the search space and
avoids the icosahedral funnel entirely by construction.

The FCC truncated octahedron for LJ38 has a specific shell structure. We generate the 38 points
from the FCC lattice, centered, and then optimize:
1. A global scale factor (1 param)
2. Optional: small anisotropic scale factors (3 params) to allow slight distortion from perfect FCC
3. Optional: a shear parameter (2 params) to allow slight distortion

This is a 6-dimensional optimization problem, which is trivially solvable with L-BFGS in milliseconds.
The key insight is that the global minimum is very close to perfect FCC, so the small parameter space
is sufficient to capture the exact minimum without needing to explore the full energy landscape.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0
START_TIME = time.time()


def lj_energy_vectorized(positions):
    """Compute LJ energy efficiently."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy


def check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)


def make_fcc_truncated_octahedron_raw():
    """
    Generate the 38-atom FCC truncated octahedron points (uncentered, unscaled).
    These are the integer FCC lattice points closest to the origin.
    """
    points = []
    max_coord = 5
    scale = 1.0 / math.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i * i + j * j + k * k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    
    return coords


def apply_rigid_transform(raw_coords, params):
    """
    Apply rigid-body transformation to raw FCC coords.
    params: [scale, sx, sy, sz, shear_xy, shear_xz]
    - scale: global scale
    - sx, sy, sz: anisotropic scale factors (1.0 = no distortion)
    - shear_xy, shear_xz: shear parameters (0.0 = no shear)
    """
    scale = params[0]
    sx = params[1]
    sy = params[2]
    sz = params[3]
    shear_xy = params[4]
    shear_xz = params[5]
    
    # Build transformation matrix
    # Diagonal scaling
    diag_scale = np.diag([sx, sy, sz])
    
    # Shear: add shear to off-diagonal
    shear_matrix = np.zeros((3, 3))
    shear_matrix[0, 1] = shear_xy
    shear_matrix[0, 2] = shear_xz
    
    transform = scale * (diag_scale + shear_matrix)
    
    transformed = raw_coords @ transform.T
    
    # Recenter
    transformed -= np.mean(transformed, axis=0)
    
    return transformed


def energy_from_params(params, raw_coords):
    """Compute energy given rigid-body parameters."""
    try:
        positions = apply_rigid_transform(raw_coords, params)
        if not check_validity(positions):
            return 1e10
        return lj_energy_vectorized(positions)
    except Exception:
        return 1e10


def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Generate raw FCC truncated octahedron
    try:
        raw_coords = make_fcc_truncated_octahedron_raw()
    except Exception:
        # Fallback: random
        rng = np.random.default_rng(0)
        raw_coords = rng.uniform(-2, 2, size=(N, 3))
        raw_coords -= np.mean(raw_coords, axis=0)
    
    # Initial parameters: scale=1.0, no distortion
    initial_params = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    
    # Optimize the 6 parameters using L-BFGS
    best_params = initial_params.copy()
    best_energy = energy_from_params(initial_params, raw_coords)
    
    try:
        if time.time() - START_TIME < DEADLINE - 20:
            result = minimize(
                energy_from_params,
                initial_params,
                args=(raw_coords,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            if result.fun < best_energy:
                best_params = result.x
                best_energy = result.fun
    except Exception:
        pass
    
    # Convert best parameters to positions
    try:
        final_pos = apply_rigid_transform(raw_coords, best_params)
        if not check_validity(final_pos):
            # Fallback: use initial params
            final_pos = apply_rigid_transform(raw_coords, initial_params)
    except Exception:
        final_pos = raw_coords
    
    # Now do a full 114D L-BFGS polish on the rigid-body result
    # This captures any residual strain not captured by the 6D parameterization
    try:
        if time.time() - START_TIME < DEADLINE - 10:
            def energy_func(pos):
                pos = pos.reshape(N, 3)
                return lj_energy_vectorized(pos)
            
            result = minimize(
                energy_func,
                final_pos.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            polished = result.x.reshape(N, 3)
            if check_validity(polished):
                e_polished = lj_energy_vectorized(polished)
                e_current = lj_energy_vectorized(final_pos)
                if e_polished < e_current:
                    final_pos = polished
    except Exception:
        pass
    
    # One more polish pass if time remains
    if time.time() - START_TIME < DEADLINE - 5:
        try:
            def energy_func(pos):
                pos = pos.reshape(N, 3)
                return lj_energy_vectorized(pos)
            
            result = minimize(
                energy_func,
                final_pos.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-16}
            )
            polished = result.x.reshape(N, 3)
            if check_validity(polished):
                e_polished = lj_energy_vectorized(polished)
                e_current = lj_energy_vectorized(final_pos)
                if e_polished < e_current:
                    final_pos = polished
        except Exception:
            pass
    
    # Final validity check
    if not check_validity(final_pos):
        final_pos = raw_coords
        if not check_validity(final_pos):
            rng = np.random.default_rng(123)
            pts = []
            for _ in range(N):
                while True:
                    p = rng.uniform(-2, 2, size=3)
                    if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                        pts.append(p)
                        break
            final_pos = np.array(pts)
    
    return final_pos


# EVOLVE-BLOCK-END
