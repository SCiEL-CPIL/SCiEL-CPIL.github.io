"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice arrangement of 11 unit hexagons
    # Using axial coordinates, convert to Cartesian for pointy-top hexagons
    # x = sqrt(3) * (q + r/2), y = 3/2 * r
    
    # Compact arrangement: 7 in hex cluster + 4 extending
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2),
    ]
    
    positions = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
    
    # Center of mass
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    
    inner = []
    for x, y in positions:
        inner.append([x - cx, y - cy, 0.0])
    
    # Compute minimum outer hexagon side length
    # For a regular hexagon with side S centered at origin, a point (x,y) is inside
    # iff for all 6 directions at 60° intervals, the projection <= S * sqrt(3) / 2
    
    def outer_hexagon_apothem(S):
        return S * sqrt3 / 2.0
    
    def point_in_outer_hex(x, y, S):
        apothem = outer_hexagon_apothem(S)
        for k in range(6):
            theta = np.radians(60 * k)
            proj = x * np.cos(theta) + y * np.sin(theta)
            if proj > apothem + 1e-10:
                return False
        return True
    
    def hexagon_fully_inside(px, py, ang_deg, S):
        """Check if unit hexagon at (px,py) with rotation ang_deg is inside outer hex of side S at origin"""
        for k in range(6):
            theta = np.radians(60 * k + ang_deg)
            vx = px + np.cos(theta)
            vy = py + np.sin(theta)
            if not point_in_outer_hex(vx, vy, S):
                return False
        return True
    
    def all_inside(S):
        for px, py, ang in inner:
            if not hexagon_fully_inside(px, py, ang, S):
                return False
        return True
    
    # Binary search for minimum S
    lo, hi = 1.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if all_inside(mid):
            hi = mid
        else:
            lo = mid
    
    S = hi
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
