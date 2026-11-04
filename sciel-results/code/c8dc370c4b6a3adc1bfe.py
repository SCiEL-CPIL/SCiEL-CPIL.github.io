"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    best_result = _construct_inner()
    return best_result

def _hexagon_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    angles = angle + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _enclosing_hexagon_side(positions, angles_deg, outer_center, outer_angle_deg):
    """
    Compute minimal side length of a regular hexagon with given center and rotation
    that contains all inner hexagons.
    """
    # All vertices of all inner hexagons
    all_verts = []
    for (x, y), ang in zip(positions, angles_deg):
        verts = _hexagon_vertices(x, y, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (66, 2)
    
    # Outer hexagon normals: 6 directions
    outer_angle = np.radians(outer_angle_deg)
    normals = np.column_stack([
        np.cos(outer_angle + np.arange(6) * np.pi / 3.0),
        np.sin(outer_angle + np.arange(6) * np.pi / 3.0)
    ])  # (6, 2)
    
    # For each normal direction, compute max projection of (vertex - center)
    rel = all_verts - np.array(outer_center)  # (66, 2)
    projections = rel @ normals.T  # (66, 6)
    max_proj = np.max(projections, axis=0)  # (6,)
    
    # Apothem = s * sqrt(3)/2, so s = max_proj / (sqrt(3)/2)
    s = np.max(max_proj) / (np.sqrt(3) / 2.0)
    return s

def _cost(params, outer_center, outer_angle_deg):
    """Cost function for optimization."""
    n = len(params) // 3
    positions = []
    angles = []
    for i in range(n):
        x, y, ang = params[3*i:3*i+3]
        positions.append((x, y))
        angles.append(ang)
    return _enclosing_hexagon_side(positions, angles, outer_center, outer_angle_deg)

def _construct_inner():
    """Main construction with optimization."""
    deadline = START_TIME + TIME_LIMIT
    best_s = float('inf')
    best_params = None
    
    # Initial guess: compact honeycomb-like arrangement
    # Place hexagons in a hexagonal pattern
    def make_initial_params(seed_offset=0):
        np.random.seed(42 + seed_offset)
        params = []
        # Center hexagon
        params.extend([0.0, 0.0, 0.0])
        # 6 surrounding
        for k in range(6):
            angle = k * np.pi / 3.0
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            params.extend([x, y, 0.0])
        # 4 more in second ring
        for k in range(4):
            angle = k * np.pi / 3.0 + np.pi / 6.0
            x = 2 * np.sqrt(3) * np.cos(angle)
            y = 2 * np.sqrt(3) * np.sin(angle)
            params.extend([x, y, 0.0])
        # Add small random perturbation
        params = np.array(params)
        params += np.random.randn(len(params)) * 0.1
        return params
    
    # Try multiple initial configurations
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        
        params0 = make_initial_params(trial)
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        # Use differential evolution for global search
        bounds = [(-10, 10)] * (N * 3)
        try:
            result = differential_evolution(
                _cost, bounds, args=(outer_center, outer_angle),
                seed=trial, maxiter=50, tol=1e-6, polish=False,
                init='sobol'
            )
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
        except Exception:
            pass
        
        # Polish with local optimization
        if best_params is not None and time.time() < deadline - 5.0:
            try:
                result2 = minimize(
                    _cost, best_params, args=(outer_center, outer_angle),
                    method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result2.fun < best_s:
                    best_s = result2.fun
                    best_params = result2.x
            except Exception:
                pass
    
    # If no optimization succeeded, fall back to seed-like result
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract final positions and angles
    inner = []
    for i in range(N):
        x, y, ang = best_params[3*i:3*i+3]
        inner.append([float(x), float(y), float(ang)])
    
    # The outer hexagon is centered at origin with rotation 0
    return inner, (0.0, 0.0), float(best_s), 0.0
# EVOLVE-BLOCK-END
