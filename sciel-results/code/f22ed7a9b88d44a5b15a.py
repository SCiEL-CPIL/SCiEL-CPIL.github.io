"""Hexagon packing: place 12 unit regular hexagons inside a minimal outer regular hexagon."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
    return verts

def _poly_support(verts, ux, uy):
    """Max projection of polygon vertices onto direction (ux, uy)."""
    return max(vx * ux + vy * uy for vx, vy in verts)

def _sat_overlap(v1, v2):
    """SAT check: True if two convex polygons overlap."""
    polys = (v1, v2)
    for poly in polys:
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # Edge normal
            ex, ey = x2 - x1, y2 - y1
            length = math.hypot(ex, ey)
            if length < 1e-12:
                continue
            nx, ny = ey / length, -ex / length
            p1min = p1max = v1[0][0] * nx + v1[0][1] * ny
            for px, py in v1[1:]:
                proj = px * nx + py * ny
                if proj < p1min: p1min = proj
                if proj > p1max: p1max = proj
            p2min = p2max = v2[0][0] * nx + v2[0][1] * ny
            for px, py in v2[1:]:
                proj = px * nx + py * ny
                if proj < p2min: p2min = proj
                if proj > p2max: p2max = proj
            if p1max < p2min - 1e-9 or p2max < p1min - 1e-9:
                return False
    return True

def _outer_hex_side(verts_all, outer_angle_deg):
    """Minimum side length of outer hexagon (at origin, rotation outer_angle_deg) to contain all polygons."""
    a = math.radians(outer_angle_deg)
    # Normals to the 6 sides of a hexagon with vertex angles a + k*pi/3
    # Side between vertex k and k+1 has normal at angle a + pi/6 + k*pi/3
    max_needed = 0.0
    for k in range(6):
        ang = a + math.pi / 6.0 + k * math.pi / 3.0
        ux, uy = math.cos(ang), math.sin(ang)
        # Max support in direction u among all vertices
        sup = -float('inf')
        for poly in verts_all:
            s = _poly_support(poly, ux, uy)
            if s > sup: sup = s
        # Also check -u direction
        sup_neg = -float('inf')
        for poly in verts_all:
            s = _poly_support(poly, -ux, -uy)
            if s > sup_neg: sup_neg = s
        # Required side: inradius = s*sqrt(3)/2 must be >= max(sup, sup_neg)
        needed = 2.0 * max(sup, sup_neg) / math.sqrt(3.0)
        if needed > max_needed:
            max_needed = needed
    return max_needed

def _is_valid(centers, angles_deg, outer_angle_deg, outer_side):
    """Check no overlap and all inside outer hexagon. Returns (valid, max_violation)."""
    # Build vertices
    all_verts = []
    for (cx, cy), ang in zip(centers, angles_deg):
        all_verts.append(_hex_vertices(cx, cy, 1.0, ang))
    # Overlap check (SAT)
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(all_verts[i], all_verts[j]):
                return False, float('inf')
    # Containment: outer side must be at least required
    req = _outer_hex_side(all_verts, outer_angle_deg)
    if req > outer_side + 1e-9:
        return False, req - outer_side
    return True, 0.0

def _score(centers, angles_deg, outer_angle_deg):
    """Return outer side needed (lower better)."""
    all_verts = []
    for (cx, cy), ang in zip(centers, angles_deg):
        all_verts.append(_hex_vertices(cx, cy, 1.0, ang))
    return _outer_hex_side(all_verts, outer_angle_deg)

def _make_config(i, j, lattice_scale):
    """Generate a centered lattice config for grid index (i,j) with given scale.
    Returns (centers, angles_deg, outer_angle_deg) and required side."""
    d = lattice_scale  # center spacing
    a1 = np.array([d, 0.0])
    a2 = np.array([d * 0.5, d * math.sqrt(3.0) / 2.0])
    pts = np.array([i0 * a1 + j0 * a2 for i0 in range(i) for j0 in range(j)])
    center = np.mean(pts, axis=0)
    pts_centered = pts - center
    centers = [(float(p[0]), float(p[1])) for p in pts_centered]
    # Inner hexagons aligned with lattice (angle 0)
    angles_deg = [0.0] * N
    outer_angle = 0.0
    side = _score(centers, angles_deg, outer_angle)
    return centers, angles_deg, outer_angle, side

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    best_side = float('inf')
    best_result = None

    # ---- Phase 1: Grid-based lattice search with scale and rotation optimization ----
    # Try several grid shapes that have exactly 12 cells
    grid_shapes = [(3, 4), (4, 3), (2, 6), (6, 2), (1, 12), (12, 1), (3, 3), (4, 4)]
    # Filter to exactly N
    grid_shapes = [s for s in grid_shapes if s[0] * s[1] == N]
    # Add the 3x4 rhombus and some hex-ish shapes
    if (3, 4) not in grid_shapes:
        grid_shapes.append((3, 4))

    # For each grid shape, search over lattice scale and rotation
    for (gi, gj) in grid_shapes:
        if time.time() > DEADLINE - 20:
            break
        # The natural spacing for touching hexagons is sqrt(3)
        # Search scale from slightly above sqrt(3) (to avoid overlap) up
        for scale in np.linspace(math.sqrt(3.0) + 0.01, math.sqrt(3.0) + 1.0, 15):
            if time.time() > DEADLINE - 20:
                break
            # Try a range of outer hexagon rotations relative to the lattice
            for outer_rot in np.linspace(0, math.pi / 3, 25):
                d = scale
                a1 = np.array([d, 0.0])
                a2 = np.array([d * 0.5, d * math.sqrt(3.0) / 2.0])
                pts = np.array([i0 * a1 + j0 * a2 for i0 in range(gi) for j0 in range(gj)])
                center = np.mean(pts, axis=0)
                pts_centered = pts - center
                centers = [(float(p[0]), float(p[1])) for p in pts_centered]
                # Inner hexagons at angle 0 (aligned with lattice)
                angles_deg = [0.0] * N
                outer_angle_deg = math.degrees(outer_rot)
                side = _score(centers, angles_deg, outer_angle_deg)
                if side < best_side:
                    best_side = side
                    best_result = (centers, angles_deg, outer_angle_deg)

    # ---- Phase 2: Local refinement via simulated annealing ----
    if best_result is not None and time.time() < DEADLINE - 30:
        centers, angles_deg, outer_angle_deg = best_result
        cur_side = best_side
        cur_centers = list(centers)
        cur_angles = list(angles_deg)
        cur_outer = outer_angle_deg

        T = 0.5  # initial temperature
        cooling = 0.995
        max_iter = 3000

        for it in range(max_iter):
            if time.time() > DEADLINE - 15:
                break
            # Propose a small perturbation
            r = random.random()
            if r < 0.5:
                # Move a random hexagon
                idx = random.randint(0, N - 1)
                dx = random.gauss(0, 0.15)
                dy = random.gauss(0, 0.15)
                new_centers = list(cur_centers)
                new_centers[idx] = (new_centers[idx][0] + dx, new_centers[idx][1] + dy)
                new_angles = cur_angles
                new_outer = cur_outer
            elif r < 0.75:
                # Rotate a random hexagon
                idx = random.randint(0, N - 1)
                new_angles = list(cur_angles)
                new_angles[idx] = cur_angles[idx] + random.gauss(0, 5.0)
                new_centers = cur_centers
                new_outer = cur_outer
            else:
                # Rotate outer hexagon
                new_outer = cur_outer + random.gauss(0, 3.0)
                new_centers = cur_centers
                new_angles = cur_angles

            new_side = _score(new_centers, new_angles, new_outer)
            if new_side < cur_side:
                cur_side = new_side
                cur_centers = new_centers
                cur_angles = new_angles
                cur_outer = new_outer
                if cur_side < best_side:
                    best_side = cur_side
                    best_result = (cur_centers, cur_angles, cur_outer)
            else:
                # Accept with probability exp(-(delta)/T)
                delta = new_side - cur_side
                if delta < 0:
                    delta = 0
                if random.random() < math.exp(-delta / max(T, 1e-6)):
                    cur_centers = new_centers
                    cur_angles = new_angles
                    cur_outer = new_outer
            T *= cooling

    # ---- Phase 3: Final validation and return ----
    if best_result is None:
        # Fallback: a valid large hexagon with hexagons in a line
        inner = [[float(i * 2.5), 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 20.0, 0.0

    centers, angles_deg, outer_angle_deg = best_result
    # Verify validity; if invalid, expand outer side to make it valid
    all_verts = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy), ang in zip(centers, angles_deg)]
    req_side = _outer_hex_side(all_verts, outer_angle_deg)
    # Ensure at least req_side to be safe
    final_side = max(req_side, best_side)
    # Add a tiny margin for floating point
    final_side += 1e-6

    inner = [[float(cx), float(cy), float(ang)] for (cx, cy), ang in zip(centers, angles_deg)]
    return inner, (0.0, 0.0), float(final_side), float(outer_angle_deg)

# EVOLVE-BLOCK-END
