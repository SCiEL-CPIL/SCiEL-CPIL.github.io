"""Hexagon packing: place N unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3.0)
HF = SQRT3 / 2.0  # half-width across flats
FW = SQRT3        # full width across flats

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, circumradius=1.0):
    """Return 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.array([cx + circumradius * np.cos(angles),
                     cy + circumradius * np.sin(angles)]).T


def _outer_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of outer regular hexagon (circumradius = side)."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.array([cx + side * np.cos(angles),
                     cy + side * np.sin(angles)]).T


def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point is inside regular hexagon."""
    dx, dy = px - cx, py - cy
    a = -np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # Regular hexagon: intersection of 6 half-planes
    # |lx| <= side, |ly| <= SQRT3/2 * side, |ly| <= SQRT3 * (side - |lx|)
    if abs(lx) > side + 1e-9 or abs(ly) > SQRT3 / 2 * side + 1e-9:
        return False
    if abs(ly) > SQRT3 * (side - abs(lx)) + 1e-9:
        return False
    return True


def _closest_on_segment(px, py, q1x, q1y, q2x, q2y):
    """Closest point on segment (q1,q2) to point (px,py)."""
    dx, dy = q2x - q1x, q2y - q1y
    t = ((px - q1x) * dx + (py - q1y) * dy) / (dx * dx + dy * dy + 1e-15)
    t = max(0.0, min(1.0, t))
    return q1x + t * dx, q1y + t * dy


def _min_sep(p1, p2):
    """Minimum separation between two hexagons (negative if overlapping).
    p1, p2 are (6,2) vertex arrays. Returns (dist, push_x, push_y) where
    push is direction to move p1 away from p2."""
    min_d = 1e18
    push = np.array([0.0, 0.0])
    for i in range(6):
        for j in range(6):
            # Edge-edge
            p1a = p1[i]; p1b = p1[(i+1) % 6]
            p2a = p2[j]; p2b = p2[(j+1) % 6]
            d1 = p1b - p1a
            d2 = p2b - p2a
            r = p1a - p2a
            a = np.dot(d1, d1)
            b = np.dot(d1, d2)
            c = np.dot(d2, d2)
            d = np.dot(d1, r)
            e = np.dot(d2, r)
            denom = a * c - b * b
            if abs(denom) < 1e-12:
                # Parallel: check vertex-edge distances
                for v in [p1a, p1b]:
                    cx, cy = _closest_on_segment(v[0], v[1], p2a[0], p2a[1], p2b[0], p2b[1])
                    diff = np.array([v[0] - cx, v[1] - cy])
                    dist = np.linalg.norm(diff)
                    if dist < min_d:
                        min_d = dist
                        push = diff
                for v in [p2a, p2b]:
                    cx, cy = _closest_on_segment(v[0], v[1], p1a[0], p1a[1], p1b[0], p1b[1])
                    diff = np.array([v[0] - cx, v[1] - cy])
                    dist = np.linalg.norm(diff)
                    if dist < min_d:
                        min_d = dist
                        push = -diff
            else:
                s = (b * e - c * d) / denom
                t = (a * e - b * d) / denom
                s = max(0, min(1, s))
                t = max(0, min(1, t))
                cp = p1a + s * d1
                cq = p2a + t * d2
                diff = cp - cq
                dist = np.linalg.norm(diff)
                if dist < min_d:
                    min_d = dist
                    push = diff
    # Also vertex-vertex
    for i in range(6):
        for j in range(6):
            diff = p1[i] - p2[j]
            dist = np.linalg.norm(diff)
            if dist < min_d:
                min_d = dist
                push = diff
    return min_d, push[0], push[1]


def _boundary_violation(poly, outer_cx, outer_cy, outer_side, outer_angle):
    """Return (min_margin, push_x, push_y) where min_margin < 0 means outside."""
    ovx, ovy = _outer_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    min_margin = 1e18
    push = np.array([0.0, 0.0])
    for i in range(6):
        px, py = poly[i]
        if _point_in_hexagon(px, py, outer_cx, outer_cy, outer_side, outer_angle):
            continue
        # Find closest point on outer boundary
        best_d = 1e18
        best_diff = np.array([0.0, 0.0])
        for j in range(6):
            q1 = ovx[j]; q2 = ovx[(j+1) % 6]
            cx, cy = _closest_on_segment(px, py, q1[0], q1[1], q2[0], q2[1])
            diff = np.array([px - cx, py - cy])
            d = np.linalg.norm(diff)
            if d < best_d:
                best_d = d
                best_diff = diff
        margin = -best_d
        if margin < min_margin:
            min_margin = margin
            push = -best_diff  # push inward
    return min_margin, push[0], push[1]


def _check_all(positions, outer_cx, outer_cy, outer_side, outer_angle):
    """Check validity. Returns (valid, max_violation)."""
    max_viol = 0.0
    polys = [_hex_vertices(positions[i, 0], positions[i, 1], positions[i, 2]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d, _, _ = _min_sep(polys[i], polys[j])
            if d < 0:
                max_viol = max(max_viol, -d)
        m, _, _ = _boundary_violation(polys[i], outer_cx, outer_cy, outer_side, outer_angle)
        if m < 0:
            max_viol = max(max_viol, -m)
    return max_viol < 1e-6, max_viol


def _required_outer_side(positions, outer_angle=0.0):
    """Compute minimum outer side to contain all hexagons."""
    # For each hexagon vertex, compute the maximum "hex-norm" distance from center
    # For a flat-top hexagon centered at origin with side s:
    # A point (x,y) is inside iff the hex-norm <= s
    # Hex-norm for flat-top: max(|x|, (|x| + sqrt(3)*|y|)/2) ... let me use the half-plane form
    # Actually for circumradius R (flat-top), the hexagon contains (x,y) iff:
    # |x| <= R and |y| <= R*sqrt(3)/2 and |y| <= sqrt(3)*(R - |x|)
    # So R >= max over vertices of: |x|, |y|/sqrt(3), (|y| + |x|)/sqrt(3) ... 
    # Let me just compute: for each vertex, find the minimum R such that it's inside.
    # Point (px, py) relative to center (0,0): 
    # R must satisfy: R >= |px|, R >= |py|/sqrt(3), R >= (|py| + |px|)/sqrt(3)
    # Actually: |py| <= sqrt(3)*(R - |px|) => R >= |px| + |py|/sqrt(3)
    # and |py| <= sqrt(3)*R/2 => R >= |py|/sqrt(3)*... wait
    # |py| <= sqrt(3)*R/2 means R >= 2|py|/sqrt(3)
    # |py| <= sqrt(3)*(R - |px|) means R >= |px| + |py|/sqrt(3)
    # |px| <= R means R >= |px|
    
    max_r = 0.0
    for i in range(N):
        vx, vy = _hex_vertices(positions[i, 0], positions[i, 1], positions[i, 2])
        for k in range(6):
            dx = vx[k] - outer_cx
            dy = vy[k] - outer_cy
            # Rotate to align with outer hexagon
            a = -np.deg2rad(outer_angle)
            ca, sa = np.cos(a), np.sin(a)
            lx = ca * dx + sa * dy
            ly = -sa * dx + ca * dy
            r1 = abs(lx)
            r2 = abs(ly) / SQRT3
            r3 = abs(lx) + abs(ly) / SQRT3
            r4 = 2.0 * abs(ly) / SQRT3
            max_r = max(max_r, r1, r2, r3, r4)
    return max_r


def _initial_honeycomb():
    """Initial positions: honeycomb-like arrangement."""
    positions = np.zeros((N, 3))
    # Center
    positions[0] = [0.0, 0.0, 0.0]
    # 6 around center at distance sqrt(3) (touching flats)
    for k in range(6):
        angle = k * np.pi / 3
        r = SQRT3
        positions[k + 1, 0] = r * np.cos(angle)
        positions[k + 1, 1] = r * np.sin(angle)
        positions[k + 1, 2] = 0.0
    # 4 more: placed in gaps
    # The gaps between adjacent ring hexagons are at angles 30, 90, 150, 210, 270, 330
    # Distance to ring hexagon center is sqrt(3). The gap center is at distance sqrt(3)*sqrt(3) = 3
    # Let's place at distance ~2.8 from center
    gap_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6]
    for k in range(4):
        angle = gap_angles[k]
        r = 2.9
        positions[k + 7, 0] = r * np.cos(angle)
        positions[k + 7, 1] = r * np.sin(angle)
        positions[k + 7, 2] = 0.0
    return positions


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_positions = None
    best_side = 1e18
    
    # Start with honeycomb
    positions = _initial_honeycomb()
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Compute initial required side
    required = _required_outer_side(positions, outer_angle)
    outer_side = required + 0.3
    
    # Validate initial
    valid, viol = _check_all(positions, outer_cx, outer_cy, outer_side, outer_angle)
    if valid:
        best_positions = positions.copy()
        best_side = outer_side
    
    # Simulated annealing / local search
    np.random.seed(42)
    step_size = 0.3
    lr = 0.05
    
    for gen in range(5000):
        if time.time() > deadline:
            break
        
        # Perturb a random hexagon
        idx = np.random.randint(N)
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        dangle = np.random.normal(0, 0.1)
        
        old_x, old_y, old_a = positions[idx]
        positions[idx, 0] += dx
        positions[idx, 1] += dy
        positions[idx, 2] += dangle
        
        # Recompute required outer side
        required = _required_outer_side(positions, outer_angle)
        new_side = required + 0.05
        
        # Quick check: just the moved hexagon against others and boundary
        poly = _hex_vertices(positions[idx, 0], positions[idx, 1], positions[idx, 2])
        local_viol = 0.0
        for j in range(N):
            if j == idx:
                continue
            poly_j = _hex_vertices(positions[j, 0], positions[j, 1], positions[j, 2])
            d, _, _ = _min_sep(poly, poly_j)
            if d < 0:
                local_viol = max(local_viol, -d)
        m, _, _ = _boundary_violation(poly, outer_cx, outer_cy, new_side, outer_angle)
        if m < 0:
            local_viol = max(local_viol, -m)
        
        if local_viol < 1e-6:
            # Valid move - update
            old_side = outer_side
            outer_side = new_side
            if outer_side < best_side:
                # Full validation
                valid, viol = _check_all(positions, outer_cx, outer_cy, outer_side, outer_angle)
                if valid:
                    best_positions = positions.copy()
                    best_side = outer_side
        else:
            # Reject
            positions[idx, 0] = old_x
            positions[idx, 1] = old_y
            positions[idx, 2] = old_a
        
        # Also try shrinking outer side slightly if we have margin
        if gen % 50 == 0 and best_positions is not None:
            test_side = best_side - 0.01
            valid, _ = _check_all(best_positions, outer_cx, outer_cy, test_side, outer_angle)
            if valid:
                best_side = test_side
        
        # Anneal: decrease step size
        step_size *= 0.999
        if step_size < 0.01:
            step_size = 0.01
    
    # Final: try to minimize outer side via binary search
    if best_positions is not None:
        lo = 0.0
        hi = best_side
        for _ in range(40):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2
            valid, _ = _check_all(best_positions, outer_cx, outer_cy, mid, outer_angle)
            if valid:
                hi = mid
            else:
                lo = mid
        best_side = hi
        if best_side < 1e10:
            inner = [[float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_positions[i, 2])] for i in range(N)]
            return inner, (float(outer_cx), float(outer_cy)), float(best_side), float(outer_angle)
    
    # Fallback: honeycomb with generous outer side
    positions = _initial_honeycomb()
    required = _required_outer_side(positions, 0.0)
    outer_side = required + 0.5
    inner = [[float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])] for i in range(N)]
    return inner, (0.0, 0.0), float(outer_side), 0.0

# EVOLVE-BLOCK-END
