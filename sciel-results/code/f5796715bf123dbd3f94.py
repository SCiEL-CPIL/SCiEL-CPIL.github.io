"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_contains(x, y, S, angle_deg):
    """Check if point (x,y) is inside regular hexagon with side S, center origin, rotation angle_deg."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Rotate point by -angle to align hexagon with axes
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return (np.abs(xr) <= S and 
            np.abs(0.5 * xr + (np.sqrt(3)/2) * yr) <= S and 
            np.abs(-0.5 * xr + (np.sqrt(3)/2) * yr) <= S)

def _hex_contains_vec(X, Y, S, angle_deg):
    """Vectorized containment check."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    xr = ca * X + sa * Y
    yr = -sa * X + ca * Y
    c1 = np.abs(xr) <= S
    c2 = np.abs(0.5 * xr + (np.sqrt(3)/2) * yr) <= S
    c3 = np.abs(-0.5 * xr + (np.sqrt(3)/2) * yr) <= S
    return c1 & c2 & c3

def _min_side_for_config(positions, angle_deg, outer_angle_deg, tol=1e-9):
    """
    Given inner hexagon centers (N x 2), their rotation angles (N,), 
    and outer hexagon rotation, find minimal outer side length S.
    Each inner hexagon has 6 vertices at distance 1 from center.
    S must be large enough that all vertices of all inner hexagons are inside outer hexagon.
    """
    N_pts = len(positions)
    # Generate all vertices of all inner hexagons
    all_x = []
    all_y = []
    for i in range(N_pts):
        cx, cy = positions[i]
        ang_i = np.radians(angle_deg[i])
        for j in range(6):
            va = ang_i + j * np.pi / 3
            all_x.append(cx + np.cos(va))
            all_y.append(cy + np.sin(va))
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Binary search on S
    lo, hi = 1.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _hex_contains_vec(all_x, all_y, mid, outer_angle_deg).all():
            hi = mid
        else:
            lo = mid
    return hi

def _check_valid(positions, angle_deg, tol=1e-6):
    """Check if inner hexagons are pairwise non-overlapping."""
    N_pts = len(positions)
    for i in range(N_pts):
        for j in range(i+1, N_pts):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            # For two unit hexagons with rotations a1, a2, the minimum distance 
            # to avoid overlap depends on relative rotation.
            # Conservative bound: if both at 0 deg, need dist >= 2.
            # For general rotations, the minimum separation is at least 
            # 2 * cos(max relative angle effect). Safest: use dist >= 2.0 
            # when both are axis-aligned, but for rotated ones it can be less.
            # Actually, two unit regular hexagons (circumradius 1) have inradius sqrt(3)/2.
            # The minimum distance between centers for non-overlap:
            # If both axis-aligned: 2.0
            # If one rotated 30 deg relative: the "width" changes.
            # General formula: min distance = 2 * max over directions of support function.
            # Simplified: use 2.0 as safe lower bound (may be slightly conservative).
            # Better: compute actual minimum distance.
            a1 = np.radians(angle_deg[i])
            a2 = np.radians(angle_deg[j])
            # The Minkowski difference of two hexagons is a 12-gon.
            # The minimum distance is the minimum distance from origin to boundary of Minkowski diff.
            # For simplicity, use: dist >= 2.0 * cos(0) = 2.0 as safe bound.
            # More accurately, for two hexagons with relative rotation delta:
            # min_dist = 2 * max(cos(delta/2), sqrt(3)/2 * sin(delta/2) * 2 / 2) ... 
            # Let's just use 2.0 as a safe conservative bound.
            if dist < 2.0 - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s safety margin
    
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = float('inf')
    best_angle = 0.0
    
    # Strategy: try various configurations
    # 1. Center + 6 ring + 4 extra
    # 2. Two rings with optimization
    # 3. Grid-like arrangements
    
    # Configuration 1: 1 center + 6 ring + 4 more
    # Ring 1: 6 hexagons at distance d1, angles 0,60,120,180,240,300
    # Ring 2: 4 hexagons at distance d2, angles 30,90,150,210,270,330 (choose 4)
    
    # Try outer rotation angles
    outer_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    
    for outer_ang in outer_angles:
        if time.time() > deadline - 10:
            break
            
        # Try various ring distances
        for d1 in np.linspace(2.0, 3.0, 21):
            if time.time() > deadline - 10:
                break
            for d2 in np.linspace(2.5, 4.0, 31):
                if time.time() > deadline - 10:
                    break
                    
                # Build configuration: center + 6 ring1 + 4 ring2
                positions = [(0.0, 0.0)]
                angle_deg = [0.0]
                
                # Ring 1: 6 hexagons
                for k in range(6):
                    a = np.radians(k * 60.0)
                    positions.append((d1 * np.cos(a), d1 * np.sin(a)))
                    angle_deg.append(0.0)
                
                # Ring 2: choose 4 from 6 possible positions (offset by 30 deg)
                # Try different subsets of 4
                ring2_angles = [30, 90, 150, 210, 270, 330]
                # Try all combinations of 4 from 6
                from itertools import combinations
                for subset in combinations(range(6), 4):
                    if time.time() > deadline - 10:
                        break
                    test_pos = list(positions)
                    test_ang = list(angle_deg)
                    for idx in subset:
                        a = np.radians(ring2_angles[idx])
                        test_pos.append((d2 * np.cos(a), d2 * np.sin(a)))
                        test_ang.append(0.0)
                    
                    if len(test_pos) != N:
                        continue
                    
                    # Check validity
                    if not _check_valid(test_pos, test_ang):
                        continue
                    
                    # Compute min outer side
                    S = _min_side_for_config(test_pos, test_ang, outer_ang)
                    if S < best_S:
                        best_S = S
                        best_inner = [[p[0], p[1], a] for p, a in zip(test_pos, test_ang)]
                        best_center = (0.0, 0.0)
                        best_side = S
                        best_angle = outer_ang
    
    # Also try: center + 10 in a more compact arrangement
    # Try a 3-4-3 or similar pattern
    
    # Try hexagonal lattice-like packing
    # 3 in a row, 4 in middle, 3 in bottom
    for outer_ang in [0.0, 30.0, 60.0]:
        if time.time() > deadline - 5:
            break
        # Hex grid spacing
        for dx in np.linspace(2.0, 2.5, 11):
            if time.time() > deadline - 5:
                break
            dy = dx * np.sqrt(3) / 2
            # Row 1: 3 hexagons
            # Row 2: 4 hexagons (offset)
            # Row 3: 3 hexagons
            positions = []
            angle_deg = []
            # Center at origin
            # Row 2 (middle): 4 hexagons at y=0
            for i in range(4):
                x = (i - 1.5) * dx
                positions.append((x, 0.0))
                angle_deg.append(0.0)
            # Row 1: 3 hexagons at y=dy
            for i in range(3):
                x = (i - 1.0) * dx
                positions.append((x, dy))
                angle_deg.append(0.0)
            # Row 3: 3 hexagons at y=-dy
            for i in range(3):
                x = (i - 1.0) * dx
                positions.append((x, -dy))
                angle_deg.append(0.0)
            
            if len(positions) != N:
                continue
            
            if not _check_valid(positions, angle_deg):
                continue
            
            S = _min_side_for_config(positions, angle_deg, outer_ang)
            if S < best_S:
                best_S = S
                best_inner = [[p[0], p[1], a] for p, a in zip(positions, angle_deg)]
                best_center = (0.0, 0.0)
                best_side = S
                best_angle = outer_ang
    
    # Fallback: if no valid config found, use a simple valid one
    if best_inner is None:
        # Simple: 11 hexagons in a line (will be large but valid)
        positions = [(2.0 * k, 0.0) for k in range(N)]
        angle_deg = [0.0] * N
        S = _min_side_for_config(positions, angle_deg, 0.0)
        best_inner = [[p[0], p[1], a] for p, a in zip(positions, angle_deg)]
        best_center = (0.0, 0.0)
        best_side = S
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
