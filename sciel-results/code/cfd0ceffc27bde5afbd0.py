"""Hexagon cluster packing: 12 unit hexes in concentric rings, optimize rotation."""
import math
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, theta):
    """Vertices of regular hexagon, center (cx,cy), circumradius r, rotation theta (rad)."""
    return [(cx + r * math.cos(theta + i * math.pi / 3.0),
             cy + r * math.sin(theta + i * math.pi / 3.0)) for i in range(6)]

def _hexagon_support(cx, cy, r, theta, ux, uy):
    """Support function of a hexagon in direction (ux,uy)."""
    best = -1e18
    for i in range(6):
        a = theta + i * math.pi / 3.0
        vx = cx + r * math.cos(a)
        vy = cy + r * math.sin(a)
        s = vx * ux + vy * uy
        if s > best:
            best = s
    return best

def _min_side_length(cx, cy, r, theta):
    """Min side length of a regular hexagon (any orientation) enclosing the given hexagon."""
    best = 1e18
    for k in range(6):
        phi = k * math.pi / 6.0
        # Support in 6 directions of the enclosing hexagon
        sup = 0.0
        for i in range(6):
            a = phi + i * math.pi / 3.0
            ux, uy = math.cos(a), math.sin(a)
            s = _hexagon_support(cx, cy, r, theta, ux, uy)
            if s > sup:
                sup = s
        # For enclosing hexagon with circumradius R, support in its vertex dir = R
        # side = R, so R >= sup
        if sup < best:
            best = sup
    return best

def _enclosing_side(ring1, ring2, global_rot):
    """Compute min enclosing hexagon side for the cluster given ring radii and global rotation.
    ring1 = radius of inner ring (5 hexes), ring2 = radius of outer ring (6 hexes).
    global_rot = rotation of the whole cluster.
    Returns (side_length, positions) where positions is list of (cx, cy, theta).
    """
    r = 1.0  # circumradius of unit hexagon
    positions = []
    # Center hex
    positions.append((0.0, 0.0, global_rot))
    # Inner ring: 5 hexes at 72-degree intervals, offset by 36 deg from global
    for i in range(5):
        a = global_rot + math.pi / 5.0 + i * 2 * math.pi / 5.0
        cx = ring1 * math.cos(a)
        cy = ring1 * math.sin(a)
        positions.append((cx, cy, global_rot))
    # Outer ring: 6 hexes at 60-degree intervals, aligned with global
    for i in range(6):
        a = global_rot + i * math.pi / 3.0
        cx = ring2 * math.cos(a)
        cy = ring2 * math.sin(a)
        positions.append((cx, cy, global_rot))

    # Compute enclosing hexagon side
    # We fix the orientation of the enclosing hexagon to have vertices at 0, pi/3, ...
    # This corresponds to side normals at pi/6, pi/2, ...
    # The minimum side length S is determined by the maximum projection of the cluster
    # onto the side normals of the enclosing hexagon.
    # Inradius R_in = S * sqrt(3) / 2  =>  S = R_in * 2 / sqrt(3)
    
    max_support = 0.0
    for k in range(6):
        # Normal direction for side k
        phi_n = k * math.pi / 3.0 + math.pi / 6.0
        ux = math.cos(phi_n)
        uy = math.sin(phi_n)
        sup = 0.0
        for (cx, cy, th) in positions:
            s = _hexagon_support(cx, cy, r, th, ux, uy)
            if s > sup:
                sup = s
        if sup > max_support:
            max_support = sup
    
    side = max_support * 2.0 / math.sqrt(3.0)
    return side, positions

def _overlap_check(ring1, ring2, global_rot):
    """Check if any two hexagons overlap (distance between centers < 2*r for same orientation).
    For same-orientation unit hexagons, non-overlap requires center distance >= 2.0.
    """
    r = 1.0
    positions = []
    positions.append((0.0, 0.0))
    for i in range(5):
        a = global_rot + math.pi / 5.0 + i * 2 * math.pi / 5.0
        positions.append((ring1 * math.cos(a), ring1 * math.sin(a)))
    for i in range(6):
        a = global_rot + i * math.pi / 3.0
        positions.append((ring2 * math.cos(a), ring2 * math.sin(a)))

    # Check all pairs
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < 2.0 - 1e-9:
                return False
    return True

def _optimize_rotation(ring1, ring2):
    """Find optimal global rotation via golden-section search over [0, pi/6]."""
    a0 = 0.0
    a1 = math.pi / 6.0
    gr = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate

    def f(rot):
        # Penalty for overlap
        if not _overlap_check(ring1, ring2, rot):
            return 1e10
        side, _ = _enclosing_side(ring1, ring2, rot)
        return side

    fa = f(a0)
    fb = f(a1)
    best_rot = a0 if fa < fb else a1
    best_val = min(fa, fb)

    x1 = a1 - gr * (a1 - a0)
    x2 = a0 + gr * (a1 - a0)
    f1 = f(x1)
    f2 = f(x2)

    for _ in range(60):
        if f1 < f2:
            a1 = x2
            x2 = x1
            f2 = f1
            x1 = a1 - gr * (a1 - a0)
            f1 = f(x1)
        else:
            a0 = x1
            x1 = x2
            f1 = f2
            x2 = a0 + gr * (a1 - a0)
            f2 = f(x2)
        if f1 < best_val:
            best_val = f1
            best_rot = x1
        if f2 < best_val:
            best_val = f2
            best_rot = x2
        if abs(a1 - a0) < 1e-10:
            break

    return best_rot, best_val

def construct():
    t0 = time.time()
    deadline = t0 + 120.0

    # Optimal ring radii for hexagonal cluster:
    # Inner ring: 5 hexes around center, center-to-center = 2.0 (touching)
    # But 5 hexes at radius 2.0 with 72-degree spacing: adjacent distance = 2*2*sin(36 deg) = 4*0.588 = 2.35 > 2.0, OK
    # Outer ring: 6 hexes, center-to-center = 2*ring2*sin(30 deg) = ring2, need >= 2.0
    # Also outer ring hexes must not overlap inner ring hexes
    # For ring2: adjacent outer hexes at 60 deg: dist = 2*ring2*sin(30) = ring2, need >= 2.0
    # Inner-outer: need to check

    # Let's try ring1 = 2.0 (touching center), ring2 = 2.0 + sqrt(3) ~ 3.732 (touching inner ring)
    # Actually for hexagonal packing, the natural spacing:
    # Center to inner: 2.0
    # Inner to outer: 2.0 (in the direction of the gap)
    # Let me just use ring1=2.0 and sweep ring2

    best_side = 1e18
    best_params = (2.0, 4.0, 0.0)

    # First: sweep ring2 for ring1=2.0
    r1 = 2.0
    for r2 in [3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.2, 4.4, 4.5, 4.6, 4.8, 5.0]:
        if time.time() > deadline - 5.0:
            break
        rot, val = _optimize_rotation(r1, r2)
        if val < best_side and val < 1e9:
            best_side = val
            best_params = (r1, r2, rot)

    # Also try different ring1 values
    for r1 in [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]:
        if time.time() > deadline - 5.0:
            break
        for r2 in [3.0, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 5.0]:
            if time.time() > deadline - 5.0:
                break
            rot, val = _optimize_rotation(r1, r2)
            if val < best_side and val < 1e9:
                best_side = val
                best_params = (r1, r2, rot)

    # Final fine-tune: local search around best
    if time.time() < deadline - 10.0:
        r1_best, r2_best, rot_best = best_params
        # Fine sweep
        for dr1 in [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]:
            for dr2 in [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]:
                if time.time() > deadline - 5.0:
                    break
                r1_t = r1_best + dr1
                r2_t = r2_best + dr2
                if r1_t < 1.5 or r2_t < 2.5:
                    continue
                rot_t, val_t = _optimize_rotation(r1_t, r2_t)
                if val_t < best_side and val_t < 1e9:
                    best_side = val_t
                    best_params = (r1_t, r2_t, rot_t)

    r1_f, r2_f, rot_f = best_params

    # Build final positions
    r = 1.0
    inner = []
    # Center
    inner.append([0.0, 0.0, math.degrees(rot_f)])
    # Inner ring: 5 hexes
    for i in range(5):
        a = rot_f + math.pi / 5.0 + i * 2 * math.pi / 5.0
        cx = r1_f * math.cos(a)
        cy = r1_f * math.sin(a)
        inner.append([cx, cy, math.degrees(rot_f)])
    # Outer ring: 6 hexes
    for i in range(6):
        a = rot_f + i * math.pi / 3.0
        cx = r2_f * math.cos(a)
        cy = r2_f * math.sin(a)
        inner.append([cx, cy, math.degrees(rot_f)])

    # Recalculate the exact side length and orientation for the final configuration
    # We used a fixed orientation for the enclosing hexagon in _enclosing_side (vertices at 0, 60, ...)
    # So outer_rot is 0 degrees.
    # However, we should verify the side length with the exact final positions to avoid floating point drift
    # or inconsistencies if best_params were updated.
    
    # Re-run the enclosing side calculation for the final parameters
    side, _ = _enclosing_side(r1_f, r2_f, rot_f)
    
    # The enclosing hexagon used in _enclosing_side has vertices at angles 0, pi/3, ...
    # So the rotation angle is 0 degrees.
    outer_rot_deg = 0.0
    
    return inner, (0.0, 0.0), side, outer_rot_deg

# EVOLVE-BLOCK-END
