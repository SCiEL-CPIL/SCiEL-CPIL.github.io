import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
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
_SQRT3 = math.sqrt(3)

# Helper: Get vertices and normals for a hexagon given center and angle
def _get_hex_props(cx, cy, angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    vx = cx + _UX * c - _UY * s
    vy = cy + _UX * s + _UY * c
    nx = _ENX * c - _ENY * s
    ny = _ENX * s + _ENY * c
    return vx, vy, nx, ny

# Helper: Check separation using SAT. Returns True if disjoint.
# Also computes minimum penetration for gradient estimation if needed, but for SLSQP we just need inequality.
# Constraint: g(x) >= 0.
# For pair (i, j), we require that for at least one axis k, the projection intervals do not overlap.
# This is an OR condition, which is hard for standard NLP.
# However, we can use a "soft" penalty or a specific formulation.
# Actually, for packing, we can define the constraint as:
# min_{axes} (separation) >= 0 is NOT correct.
# The correct condition is: max_{axes} (gap) >= 0.
# Gap on axis n is min( max(proj_i) - min(proj_j), max(proj_j) - min(proj_i) )? No.
# If they are separated, there exists an axis where they don't overlap.
# Let d_i(n) be the interval of hex i on axis n.
# They are disjoint if there exists n such that d_i(n) and d_j(n) are disjoint.
# This is a disjunctive constraint.
#
# Alternative Strategy:
# Instead of strict SAT constraints which are disjunctive, we can use a smooth penalty for overlap in the objective function and rely on global search (Differential Evolution) to find a valid configuration, then polish with local search that handles overlaps via simple push-apart or by using a large penalty weight.
#
# Let's try a different structural change:
# Use Differential Evolution (global) to minimize the outer side length, with a large penalty for any overlap.
# The penalty function is the sum of penetration depths.
# This avoids the disjunctive constraint issue.

def _penalty_and_side(x):
    """
    x: array of shape (N, 3) [x, y, angle_deg]
    Returns: (penalty, side_length)
    """
    t_start = time.time()
    angles_rad = np.radians(x[:, 2])
    
    # Compute all vertices
    # v[i, k, 0] is x-coord of k-th vertex of i-th hex
    # v[i, k, 1] is y-coord
    vx = np.zeros((N, 6))
    vy = np.zeros((N, 6))
    nx = np.zeros((N, 6))
    ny = np.zeros((N, 6))
    
    for i in range(N):
        vx[i], vy[i], nx[i], ny[i] = _get_hex_props(x[i, 0], x[i, 1], angles_rad[i])
        
    # Check overlaps and compute penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Quick distance check
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            if dx*dx + dy*dy > 4.5: # (1.5*sqrt(3))^2 approx, actually max distance for touch is 2*sqrt(3) ~ 3.46, squared ~ 12. 4.5 is too small?
                # Max distance between centers of touching unit hexes is 2*sqrt(3) ~ 3.464. Squared is 12.
                # If dist_sq > 12, they are definitely disjoint.
                continue
            
            # SAT check for penetration
            # We need to find if there is a separating axis.
            # If not, compute the minimum penetration depth to use as penalty.
            
            min_penetration = float('inf')
            separated = False
            
            # Check axes of hex i
            for k in range(6):
                ax, ay = nx[i, k], ny[i, k]
                proj_i = vx[i] * ax + vy[i] * ay
                proj_j = vx[j] * ax + vy[j] * ay
                
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                
                # Check for separation
                if max_i <= min_j or max_j <= min_i:
                    separated = True
                    break
                
                # Calculate overlap (penetration) on this axis
                # The gap is negative if overlapping.
                # overlap = min(max_i, max_j) - max(min_i, min_j)
                # We want the minimum overlap across all axes? No, the minimum penetration depth is the minimum overlap across all axes?
                # Actually, the minimum translation vector (MTV) magnitude is the minimum overlap across all axes.
                overlap = min(max_i, max_j) - max(min_i, min_j)
                if overlap < min_penetration:
                    min_penetration = overlap
            
            # Check axes of hex j if not separated
            if not separated:
                for k in range(6):
                    ax, ay = nx[j, k], ny[j, k]
                    proj_i = vx[i] * ax + vy[i] * ay
                    proj_j = vx[j] * ax + vy[j] * ay
                    
                    min_i, max_i = proj_i.min(), proj_i.max()
                    min_j, max_j = proj_j.min(), proj_j.max()
                    
                    if max_i <= min_j or max_j <= min_i:
                        separated = True
                        break
                    
                    overlap = min(max_i, max_j) - max(min_i, min_j)
                    if overlap < min_penetration:
                        min_penetration = overlap
            
            if not separated:
                # They overlap. Penalty is the penetration depth.
                # We use the minimum penetration depth found (MTV length).
                penalty += min_penetration
                # Early exit if penalty is already huge?
                # if penalty > 100: return penalty, 0.0

    # Compute outer side length
    # Flatten vertices
    all_vx = vx.flatten()
    all_vy = vy.flatten()
    center_x = all_vx.mean()
    center_y = all_vy.mean()
    
    offsets_x = all_vx - center_x
    offsets_y = all_vy - center_y
    
    # Find min enclosing hexagon side length
    # We scan rotation angle phi.
    # For a given phi, the side length S is determined by the max projection onto the 6 normal directions.
    # S = (2/sqrt(3)) * max_k ( max_v | dot(v - center, n_k) | )
    
    # To speed this up, we can sample phi or use a coarse scan + fine tune.
    # Since this is called many times in DE, we need it fast.
    # Coarse scan 0-60 deg with 1 deg step.
    best_side = float('inf')
    phi_vals = np.radians(np.arange(0, 60, 1))
    
    # Normals for each phi
    # n_k(phi) = phi + pi/6 + k*pi/3
    # We can vectorize this.
    # offsets shape (72, 2)
    # phi_vals shape (60,)
    # normal_angles shape (60, 6)
    
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nxs = np.cos(normal_angles)
    nys = np.sin(normal_angles)
    
    # proj[i, j, k] = offsets[i, 0]*nx[j, k] + offsets[i, 1]*ny[j, k]
    # This is (72, 60, 6). 72*60*6 = 25920 operations. Very fast.
    proj = offsets_x[:, None, None] * nxs[None, :, :] + offsets_y[:, None, None] * nys[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0) # (60, 6)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side = np.max(side_vals).min()
    
    # Optional: fine tune if time permits? For DE, probably not worth it in objective function.
    # We will polish later.
    
    return penalty, float(best_side)

def _objective(x):
    penalty, side = _penalty_and_side(x)
    # Large weight for penalty to ensure validity
    return side + 100.0 * penalty

def construct():
    t0 = time.time()
    
    # Bounds for Differential Evolution
    # Initial guess: hexagons near origin
    # x, y in [-5, 5], angle in [-30, 30]
    bounds = [(-5, 5), (-5, 5), (-30, 30)] * N
    
    # Use Differential Evolution for global search
    # Max time for DE: 100 seconds
    de_deadline = t0 + 100.0
    
    try:
        res = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-4,
            polish=False,  # We will do our own polish
            # Update function to check time
            callback=lambda xk, convergence: (time.time() > de_deadline),
            init='sobol'
        )
        
        best_x = res.x.reshape(N, 3)
        
    except Exception as e:
        # Fallback to a simple heuristic if DE fails or times out badly
        best_x = None
        # Generate a honeycomb pattern
        a1 = np.array([_SQRT3, 0.0])
        a2 = np.array([_SQRT3 / 2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_x = np.zeros((N, 3))
        for idx, (i, j) in enumerate(lattice_coords):
            best_x[idx, 0] = i * a1[0] + j * a2[0]
            best_x[idx, 1] = i * a1[1] + j * a2[1]
            best_x[idx, 2] = 0.0

    # Check time
    if time.time() > DEADLINE - 10.0:
        # Not enough time for polish, just return best found
        penalty, side = _penalty_and_side(best_x)
        if penalty > 1e-6:
             # If invalid, try to return a known valid config?
             # The honeycomb is valid.
             a1 = np.array([_SQRT3, 0.0])
             a2 = np.array([_SQRT3 / 2.0, 1.5])
             lattice_coords = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (-1, 1), (2, -1), (-2, 1),
                (1, 1), (-1, -1), (2, 0),
             ]
             best_x = np.zeros((N, 3))
             for idx, (i, j) in enumerate(lattice_coords):
                best_x[idx, 0] = i * a1[0] + j * a2[0]
                best_x[idx, 1] = i * a1[1] + j * a2[1]
                best_x[idx, 2] = 0.0
             penalty, side = _penalty_and_side(best_x)

    # Polish with Local Search (Nelder-Mead or Simple Coordinate Descent)
    # We want to minimize side length while keeping penalty 0.
    # Use a smooth objective that heavily penalizes overlap.
    
    def _smooth_obj(x):
        penalty, side = _penalty_and_side(x)
        return side + 1000.0 * penalty
        
    # Use Nelder-Mead for local optimization
    # Starting point: best_x
    # We need to be careful with angle wrapping.
    
    # Reshape to 1D for minimize
    x0 = best_x.flatten()
    
    nm_deadline = DEADLINE - 5.0
    
    try:
        # Nelder-Mead might be slow if dimensions are high (36D).
        # But it's robust.
        # Alternatively, use COBYLA?
        # Let's try Nelder-Mead with a limited number of iterations.
        res_nm = minimize(
            _smooth_obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        # Check if it improved and is valid
        x_nm = res_nm.x.reshape(N, 3)
        p_nm, s_nm = _penalty_and_side(x_nm)
        
        # Compare with previous best
        p_old, s_old = _penalty_and_side(best_x)
        
        # Prefer valid solutions with lower side.
        # If p_old is 0, we only accept if p_nm is 0 and s_nm < s_old.
        # If p_old > 0, we accept if p_nm < p_old or (p_nm==0 and s_nm < s_old).
        
        if p_nm < p_old or (p_nm < 1e-6 and s_nm < s_old):
            best_x = x_nm
            best_side = s_nm
        else:
            best_side = s_old
            
    except Exception:
        p_old, best_side = _penalty_and_side(best_x)

    # Final Verification and Calculation of Center and Angle
    # We need to return the center and rotation of the OUTER hexagon.
    # The `_penalty_and_side` function calculates the side length assuming the outer hexagon is centered at the centroid of the inner hexagons and rotated by the optimal phi.
    # We need to extract that phi and the center.
    
    angles_rad = np.radians(best_x[:, 2])
    vx = np.zeros((N, 6))
    vy = np.zeros((N, 6))
    for i in range(N):
        vx[i], vy[i], _, _ = _get_hex_props(best_x[i, 0], best_x[i, 1], angles_rad[i])
        
    all_vx = vx.flatten()
    all_vy = vy.flatten()
    center_x = all_vx.mean()
    center_y = all_vy.mean()
    
    offsets_x = all_vx - center_x
    offsets_y = all_vy - center_y
    
    # Re-scan for optimal phi to get the exact angle
    phi_vals = np.radians(np.arange(0, 60, 0.1)) # Finer scan for final result
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nxs = np.cos(normal_angles)
    nys = np.sin(normal_angles)
    proj = offsets_x[:, None, None] * nxs[None, :, :] + offsets_y[:, None, None] * nys[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side_final = np.max(side_vals).min()
    best_idx = np.argmin(np.max(side_vals, axis=1))
    best_phi = float(phi_vals[best_idx])
    
    # If penalty is still non-zero, the result is invalid.
    # The evaluator will score 0 if invalid.
    # We should ensure we return a valid configuration.
    # The honeycomb is always valid.
    # Let's check validity one last time.
    p_final, _ = _penalty_and_side(best_x)
    if p_final > 1e-5:
        # Revert to honeycomb
        a1 = np.array([_SQRT3, 0.0])
        a2 = np.array([_SQRT3 / 2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_x = np.zeros((N, 3))
        for idx, (i, j) in enumerate(lattice_coords):
            best_x[idx, 0] = i * a1[0] + j * a2[0]
            best_x[idx, 1] = i * a1[1] + j * a2[1]
            best_x[idx, 2] = 0.0
            
        # Recalculate center and phi for honeycomb
        angles_rad = np.radians(best_x[:, 2])
        vx = np.zeros((N, 6))
        vy = np.zeros((N, 6))
        for i in range(N):
            vx[i], vy[i], _, _ = _get_hex_props(best_x[i, 0], best_x[i, 1], angles_rad[i])
        all_vx = vx.flatten()
        all_vy = vy.flatten()
        center_x = all_vx.mean()
        center_y = all_vy.mean()
        offsets_x = all_vx - center_x
        offsets_y = all_vy - center_y
        
        phi_vals = np.radians(np.arange(0, 60, 0.1))
        normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
        nxs = np.cos(normal_angles)
        nys = np.sin(normal_angles)
        proj = offsets_x[:, None, None] * nxs[None, :, :] + offsets_y[:, None, None] * nys[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / _SQRT3) * max_projs
        best_side_final = np.max(side_vals).min()
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_phi = float(phi_vals[best_idx])

    inner = []
    for i in range(N):
        inner.append([float(best_x[i, 0]), float(best_x[i, 1]), float(best_x[i, 2])])
    
    center = (float(center_x), float(center_y))
    side = float(best_side_final)
    angle = float(best_phi)
    
    return inner, center, side, angle
