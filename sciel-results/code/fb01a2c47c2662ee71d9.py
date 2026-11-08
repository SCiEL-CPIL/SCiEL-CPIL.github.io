import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _support_unit_hex(theta):
    """Support function of a unit regular hexagon (circumradius 1) in direction theta."""
    angles = np.arange(6) * np.pi / 3
    return np.max(np.cos(theta - angles))

def _check_fit(inner, outer_center, side, outer_angle):
    """Check if all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    apothem = side * np.sqrt(3) / 2
    normal_angles = outer_angle + np.arange(6) * np.pi / 3
    for (xi, yi, beta) in inner:
        for na in normal_angles:
            proj_center = (xi - cx) * np.cos(na) + (yi - cy) * np.sin(na)
            support = _support_unit_hex(na - beta)
            if proj_center + support > apothem + 1e-9:
                return False
    return True

def _objective(params, inner):
    """Objective: minimize side length, with penalty for infeasibility."""
    cx, cy, side, outer_angle = params
    if side <= 0:
        return 1e6
    apothem = side * np.sqrt(3) / 2
    normal_angles = outer_angle + np.arange(6) * np.pi / 3
    violations = 0.0
    for (xi, yi, beta) in inner:
        for na in normal_angles:
            proj_center = (xi - cx) * np.cos(na) + (yi - cy) * np.sin(na)
            support = _support_unit_hex(na - beta)
            violation = proj_center + support - apothem
            if violation > 0:
                violations += violation * violation
    return side + 100.0 * violations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    sqrt3 = np.sqrt(3)
    
    # Generate hexagonal lattice positions for N unit hexagons
    axial_positions = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            dist = max(abs(q), abs(r), abs(q + r))
            if dist <= 2:
                axial_positions.append((q, r))
    
    def hex_dist(q, r):
        return max(abs(q), abs(r), abs(q + r))
    
    axial_positions.sort(key=lambda p: (hex_dist(p[0], p[1]), p[0], p[1]))
    axial_selected = axial_positions[:N]
    
    # Convert to Cartesian (flat-top hexagon tiling)
    positions = []
    for q, r in axial_selected:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
    
    # Inner hexagons: all with angle 0
    inner = [[x, y, 0.0] for (x, y) in positions]
    
    # Estimate initial side length
    max_x = max(abs(p[0]) for p in positions)
    max_y = max(abs(p[1]) for p in positions)
    initial_side = 2 * max(max_x, max_y) + 2
    
    best_params = [0.0, 0.0, initial_side, 0.0]
    best_side = initial_side
    
    try:
        # Nelder-Mead optimization
        result = minimize(
            _objective,
            best_params,
            args=(inner,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        if result.x[2] > 0 and result.fun < best_side:
            best_params = result.x
            best_side = result.x[2]
        
        # Try different initial angles
        if time.time() < deadline - 30:
            for angle_offset in [np.pi/6, np.pi/3, np.pi/12]:
                if time.time() > deadline - 30:
                    break
                test_params = [0.0, 0.0, initial_side, angle_offset]
                result2 = minimize(
                    _objective,
                    test_params,
                    args=(inner,),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if result2.x[2] > 0 and result2.fun < best_side:
                    best_params = result2.x
                    best_side = result2.x[2]
    except Exception:
        pass
    
    # Verify and adjust side length via binary search
    cx, cy, side, outer_angle = best_params
    
    if not _check_fit(inner, (cx, cy), side, outer_angle):
        lo = side * 0.5
        hi = side * 2.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if _check_fit(inner, (cx, cy), mid, outer_angle):
                hi = mid
            else:
                lo = mid
        side = hi
    
    return inner, (cx, cy), side, outer_angle
# EVOLVE-BLOCK-END
