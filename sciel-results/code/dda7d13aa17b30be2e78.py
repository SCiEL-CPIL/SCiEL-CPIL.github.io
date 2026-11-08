"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([(cx + side * np.cos(angle_rad + i * np.pi / 3),
                      cy + side * np.sin(angle_rad + i * np.pi / 3)) for i in range(6)])

def _min_enclosing_hex_side(centers, sides, angles, outer_center, outer_angle):
    max_extent = 0.0
    outer_angle_rad = np.radians(outer_angle)
    for cx, cy, ang in zip(centers, sides, angles):
        v = _hex_vertices(cx, cy, sides[0] if len(sides) == 1 else sides, ang)
        for px, py in v:
            dx = px - outer_center[0]
            dy = py - outer_center[1]
            x = dx * np.cos(-outer_angle_rad) - dy * np.sin(-outer_angle_rad)
            y = dx * np.sin(-outer_angle_rad) + dy * np.cos(-outer_angle_rad)
            for k in range(6):
                theta = k * np.pi / 3
                val = x * np.cos(theta) + y * np.sin(theta)
                needed = abs(val) * 2 / np.sqrt(3)
                if needed > max_extent:
                    max_extent = needed
    return max_extent

def _score(outer_center, outer_angle, inner_positions):
    centers = [p[0] for p in inner_positions]
    sides = [1.0] * len(inner_positions)
    angles = [p[2] for p in inner_positions]
    return _min_enclosing_hex_side(centers, sides, angles, outer_center, outer_angle)

def construct():
    sqrt3 = np.sqrt(3)
    
    # Hexagonal cluster: 1 + 6 + 4
    positions = [(0.0, 0.0, 0.0)]
    for i in range(6):
        angle = i * np.pi / 3
        positions.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle), 0.0))
    
    # 4 more in second ring
    for angle_deg in [30, 150, 210, 330]:
        angle = np.radians(angle_deg)
        # Second ring at distance sqrt(3) * sqrt(3) = 3? No, in hex grid the second ring is at distance 2*sqrt(3) in main dirs
        # Let me place them at the "corners" of the hexagonal cluster
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        positions.append((x, y, 0.0))
    
    def objective(params):
        ocx, ocy, oang = params
        return _score((ocx, ocy), oang, positions)
    
    best = float('inf')
    best_p = (0.0, 0.0, 0.0)
    
    for oang0 in [0, 30, 60, 90, 120, 150]:
        res = minimize(objective, [0.0, 0.0, oang0], method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.fun < best:
            best = res.fun
            best_p = res.x
    
    outer_center = (best_p[0], best_p[1])
    outer_angle = best_p[2]
    side = best
    
    inner = [[p[0], p[1], p[2]] for p in positions]
    return inner, outer_center, side, outer_angle
# EVOLVE-BLOCK-END

