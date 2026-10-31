"""Unit hexagons in a hexagon: minimize outer side length."""
import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hex_verts(cx, cy, angle_deg, s=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for k in range(6):
        t = math.radians(60.0 * k)
        x = s * math.cos(t)
        y = s * math.sin(t)
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        pts.append((cx + xr, cy + yr))
    return pts

def _poly_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5

def _separating_axis_overlap(a, b):
    # a,b are lists of (x,y) polygons; returns True if overlap (strict interior)
    polys = [a, b]
    # Build edge normals
    for p in polys:
        n = len(p)
        for i in range(n):
            x1, y1 = p[i]
            x2, y2 = p[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            L = math.hypot(ex, ey)
            if L < 1e-12:
                continue
            nx, ny = -ey / L, ex / L
            min_a, max_a = 1e18, -1e18
            for (x, y) in a:
                d = x * nx + y * ny
                if d < min_a: min_a = d
                if d > max_a: max_a = d
            min_b, max_b = 1e18, -1e18
            for (x, y) in b:
                d = x * nx + y * ny
                if d < min_b: min_b = d
                if d > max_b: max_b = d
            # Overlap if intervals intersect
            # Strict: gap > 0 means no overlap; small epsilon tolerance
            if max_a < min_b - 1e-12 or max_b < min_a - 1e-12:
                return False
    # Check if one polygon is fully inside the other (same orientation convex)
    # Since all convex, if no axis separates, they overlap
    return True

def _no_overlap(pts_list, eps=1e-9):
    n = len(pts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _separating_axis_overlap(pts_list[i], pts_list[j]):
                return False
    return True

def _outer_contains(inner_pts_list, outer_pts):
    # Check each inner vertex inside outer (convex)
    for ip in inner_pts_list:
        for (x, y) in ip:
            # point in convex polygon (outer_pts CCW)
            n = len(outer_pts)
            inside = True
            for k in range(n):
                x1, y1 = outer_pts[k]
                x2, y2 = outer_pts[(k + 1) % n]
                cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                if cross < -1e-9:
                    inside = False
                    break
            if not inside:
                return False
    return True

def _outer_side_needed(inner_pts_list, outer_center, outer_angle_deg):
    # Find minimal outer side length s (outer centered at outer_center, rotated outer_angle_deg)
    # such that all inner vertices are inside outer hexagon of side s.
    # Outer hexagon with side s has vertices at distance R = s from center at angles 60k + outer_angle.
    # For a given direction, the boundary of the hexagon is at distance r(theta) = s * sqrt(3) / (2 * max_k cos(theta - (60k+outer_angle)))... 
    # Actually, for a regular hexagon with circumradius R=s, the support function: distance from center to edge along normal at angle phi is apothem = s*sqrt(3)/2.
    # The hexagon is intersection of 6 half-planes. A point is inside iff for all 6 edges, dot(point-center, normal_i) <= apothem.
    # normals at angles 60k + outer_angle + 30 (edge normals point outward perpendicular to edges)
    # Edge normals are at angles 60k + outer_angle + 30? For hexagon with vertices at 60k+outer_angle, edges are between consecutive vertices; the outward normal of edge k is at angle 60k + outer_angle + 30.
    # apothem = R * cos(30) = s * sqrt(3)/2.
    a = math.radians(outer_angle_deg)
    apothem_base = math.sqrt(3) / 2.0  # for s=1
    # For each inner point, compute required apothem = max over normals of dot(p-c, n_k)
    # Then s = apothem / apothem_base
    max_needed = 0.0
    for ip in inner_pts_list:
        for (x, y) in ip:
            px, py = x - outer_center[0], y - outer_center[1]
            for k in range(6):
                nangle = a + math.radians(30.0 + 60.0 * k)
                nx, ny = math.cos(nangle), math.sin(nangle)
                d = px * nx + py * ny
                if d > max_needed:
                    max_needed = d
    s = max_needed / apothem_base
    if s < 1.0:
        s = 1.0
    return s

def _score_config(positions, angles, outer_center, outer_angle_deg):
    inner_pts = [_hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
    # Check overlaps
    if not _no_overlap(inner_pts):
        return float('inf'), inner_pts
    s = _outer_side_needed(inner_pts, outer_center, outer_angle_deg)
    return s, inner_pts

def _lattice_initial():
    # Hexagonal packing lattice: unit hexagons, spacing to be non-overlapping.
    # Adjacent hex centers distance for non-overlap: at least 2*apothem? Actually two unit hexagons (side 1) 
    # minimum center distance when edges touch = 2 * apothem = sqrt(3) for aligned hexagons.
    # But we can rotate to pack better. Start with a flat-top lattice.
    # Use spacing sqrt(3) for axis-aligned.
    pos = []
    ang = []
    # Centered hex lattice of 11 points: rings
    # Ring 0: 1, ring 1: 6, ring 2: 12 -> but we need 11. Use ring0 + ring1 + part of ring2.
    # Simpler: 11 in a compact cluster.
    # Let's do a triangular arrangement.
    import random
    rng = random.Random(42)
    # Start with hex lattice positions
    sp = math.sqrt(3)  # spacing for aligned
    # Generate a hex grid cluster
    pts = []
    # axial coordinates for small hex cluster
    # We'll just place in rows
    # Row 0: 3, Row 1: 4, Row 2: 4 -> 11
    rows = [3, 4, 4]
    r = 0
    for cnt in rows:
        for i in range(cnt):
            x = (i - (cnt - 1) / 2.0) * sp
            y = r * 1.5  # vertical spacing for pointy? For flat-top, vertical spacing 1.5*s? 
            # For flat-top hexes with side 1, horizontal spacing sqrt(3), vertical spacing 1.5
            pts.append([x, y])
        r += 1
    pos = [tuple(p) for p in pts]
    ang = [0.0] * N
    return pos, ang

def construct():
    t0 = time.time()
    deadline = 120.0 - 15.0  # safety margin
    
    pos, ang = _lattice_initial()
    
    # Evaluate initial
    best_score, _ = _score_config(pos, ang, (0.0, 0.0), 0.0)
    best_pos = [list(p) for p in pos]
    best_ang = list(ang)
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try to improve outer angle for current positions
    for oa in range(0, 60, 5):
        s, _ = _score_config(pos, ang, (0.0, 0.0), float(oa))
        if s < best_score:
            best_score = s
            best_outer_angle = float(oa)
    
    # Simple local search: random perturbations of positions and angles, accept if better
    import random
    rng = random.Random(123)
    
    step = 0.3
    iters = 0
    while time.time() < deadline:
        # Random move: perturb one or more hexagons
        npert = rng.randint(1, 3)
        new_pos = [list(p) for p in best_pos]
        new_ang = list(best_ang)
        for _ in range(npert):
            idx = rng.randint(0, N - 1)
            new_pos[idx] = [new_pos[idx][0] + rng.uniform(-step, step),
                            new_pos[idx][1] + rng.uniform(-step, step)]
            new_ang[idx] = (new_ang[idx] + rng.uniform(-15.0, 15.0)) % 360.0
        
        s, _ = _score_config(new_pos, new_ang, best_outer_center, best_outer_angle)
        if s < best_score - 1e-12:
            best_score = s
            best_pos = new_pos
            best_ang = new_ang
            # Occasionally re-optimize outer angle
            if iters % 20 == 0:
                for oa in range(0, 60, 5):
                    s2, _ = _score_config(best_pos, best_ang, best_outer_center, float(oa))
                    if s2 < best_score - 1e-12:
                        best_score = s2
                        best_outer_angle = float(oa)
        else:
            # Slowly reduce step
            step *= 0.995
            if step < 0.01:
                step = 0.01
        iters += 1
    
    # Final: ensure validity
    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], best_ang[i]])
    
    return inner, best_outer_center, best_score, best_outer_angle
# EVOLVE-BLOCK-END
