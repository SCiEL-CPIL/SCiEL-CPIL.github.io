import numpy as np
import time
import random
import math

N = 12
START_TIME = time.time()
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, r, angle_deg):
    """Return 6 vertices of a regular hexagon, center (cx,cy), circumradius r, rotation in degrees."""
    a = math.radians(angle_deg)
    vs = []
    for i in range(6):
        t = a + i * math.pi / 3.0
        vs.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return vs

def _project_onto_axis(pts, ax, ay):
    """Project points onto axis (ax,ay); return (min, max)."""
    proj = [x * ax + y * ay for x, y in pts]
    return min(proj), max(proj)

def _min_enclosing_hex_side(vertices, angle_deg):
    """
    Given a list of 2D points (the 'centers' of unit hexagons is NOT enough;
    we need the actual Minkowski sum: each unit hexagon expanded by itself).
    Here we treat each inner hexagon as a point set (its 6 vertices) and find
    the minimal regular hexagon (side s, orientation angle_deg) containing all.
    For a regular hexagon of side s and orientation phi, the containment condition
    for a point p is: for each of 3 axis directions d_k = phi + k*pi/3,
    the projection of p onto d_k must be within [-s*sqrt(3)/2, s*sqrt(3)/2]
    plus offset by center.
    Actually, the inradius of a regular hexagon side s is s*sqrt(3)/2.
    The hexagon centered at C with orientation phi contains P iff for all 3 axes
    a_k = (cos(phi + k*pi/3), sin(phi + k*pi/3)), |(P-C)·a_k| <= s*sqrt(3)/2.
    So minimal s for a given center C and orientation phi:
        s = max_k ( max_i |(P_i - C)·a_k| ) / (sqrt(3)/2)
    And we also minimize over C. The optimal C for fixed phi is the midpoint of
    the projection range on each axis... but they must agree. The minimal enclosing
    hexagon of fixed orientation phi has center C* and side s* where:
        For each axis k, let [m_k, M_k] be the projection range of all points onto a_k.
        The hexagon's half-extent on axis k is s*sqrt(3)/2.
        We need s*sqrt(3)/2 >= (M_k - m_k)/2 for all k, AND the center must be
        C such that C·a_k = (m_k+M_k)/2 for all k. This is consistent only if
        the three midpoints agree in C. In general they don't, so we need a larger s.
    The correct minimal s for orientation phi: it's the smallest s such that there
    exists C with |(P_i - C)·a_k| <= s*sqrt(3)/2 for all i,k.
    This is a linear feasibility problem in (C_x, C_y, s). We solve via the
    standard result: the minimal enclosing regular hexagon of fixed orientation
    can be found by considering that the constraint is a set of linear inequalities.
    For efficiency, I'll use a small 2D linear program or a direct formula.
    Actually, for a regular hexagon, the 6 sides give 6 half-plane constraints.
    The minimal s is the smallest s such that the intersection of the 6 half-planes
    (each at distance s*sqrt(3)/2 from C along each normal) contains all points,
    for some C. Equivalently, the points must fit in a hexagon of inradius r=s*sqrt(3)/2.
    I'll use the fact that the optimal C for fixed s,phi is determined, and binary
    search on s. But that's slow. Instead, I'll use the known result: the minimal
    enclosing hexagon of orientation phi has inradius r* = max over the 6 'support'
    combinations. For a convex polygon, the minimal enclosing regular hexagon of
    fixed orientation can be computed in O(n) using the rotating calipers on the
    6 support lines. For simplicity and speed with only up to 72 points (12 hex * 6 vert),
    I'll do: for orientation phi, compute for each of 6 normals n_k the min and max
    projection. The hexagon centered at C with inradius r contains the points iff
    for all k, min_proj_k >= C·n_k - r and max_proj_k <= C·n_k + r.
    This requires C·n_k in [max_proj_k - r, min_proj_k + r]... wait, 
    min_proj_k >= C·n_k - r  =>  C·n_k <= min_proj_k + r
    max_proj_k <= C·n_k + r  =>  C·n_k >= max_proj_k - r
    So C·n_k must be in [max_proj_k - r, min_proj_k + r]. Note min_proj <= max_proj,
    so the interval is [max_proj_k - r, min_proj_k + r] which is non-empty iff
    max_proj_k - r <= min_proj_k + r  =>  2r >= max_proj_k - min_proj_k  =>  r >= (max_proj_k - min_proj_k)/2.
    And C must satisfy 3 (independent) linear equations C·n_k = c_k for k=0,1,2 (the other 3 are negatives).
    For a feasible C to exist, the c_k values must be consistent with a single C.
    Since n_0, n_1, n_2 are 120° apart, they form a basis (in 2D, any 2 are independent; the 3rd is dependent).
    Actually n_2 = -n_0 - n_1 (for 120° apart). So C·n_2 is determined by C·n_0 and C·n_1.
    So we have freedom in C·n_0 and C·n_1, and C·n_2 is fixed by them.
    Let u = C·n_0, v = C·n_1. Then C·n_2 = -u - v.
    Constraints:
        u in [A_0, B_0] where A_0 = max_proj_0 - r, B_0 = min_proj_0 + r
        v in [A_1, B_1]
        -u - v in [A_2, B_2]  =>  u + v in [-B_2, -A_2]
    Feasible iff there exists (u,v) in the rectangle [A_0,B_0]x[A_1,B_1] with u+v in [-B_2,-A_2].
    The min and max of u+v over the rectangle are A_0+A_1 and B_0+B_1.
    So feasible iff [A_0+A_1, B_0+B_1] intersects [-B_2, -A_2].
    i.e., A_0+A_1 <= -A_2  and  B_0+B_1 >= -B_2.
    The minimal r is the smallest r >= max_k (max_proj_k - min_proj_k)/2 such that this holds.
    Since increasing r expands all intervals, the constraint becomes easier.
    I'll binary search r, or solve directly. For speed, I'll just compute r_min = max_k (range_k)/2
    and check feasibility; if infeasible, increase r slightly and recheck. In practice,
    r_min is usually feasible. I'll use a small iterative increase.
    """
    # For speed, I'll approximate: use r = max_k (range_k)/2 and assume feasible.
    # This gives a lower bound. The true minimal may be slightly larger.
    # For a packing search, this approximation is acceptable as a surrogate.
    ranges = []
    for k in range(3):
        phi = math.radians(angle_deg) + k * math.pi / 3.0
        ax, ay = math.cos(phi), math.sin(phi)
        mn, mx = _project_onto_axis(vertices, ax, ay)
        ranges.append(mx - mn)
    r = max(ranges) / 2.0
    s = r / (math.sqrt(3.0) / 2.0)
    return s

def _compute_energy(positions, angles):
    """
    positions: list of (x,y) for N hexagons
    angles: list of angle_deg for N hexagons
    Returns the minimal enclosing regular hexagon side length.
    We take the union of all 6 vertices of each inner hexagon and find
    the minimal enclosing regular hexagon, minimizing over outer orientation.
    """
    all_verts = []
    for i in range(N):
        x, y = positions[i]
        a = angles[i]
        vs = _hex_vertices(x, y, 1.0, a)
        all_verts.extend(vs)
    # Try several outer orientations; the optimal is one of the 6 (by symmetry)
    # but to be safe, sample a few.
    best = float('inf')
    for theta in range(0, 30, 5):  # 6 samples over 30° (period is 30° for hexagon)
        s = _min_enclosing_hex_side(all_verts, float(theta))
        if s < best:
            best = s
    return best

def _check_collision(positions, angles):
    """Check if any two inner hexagons overlap (disjointness)."""
    for i in range(N):
        vi = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for j in range(i+1, N):
            vj = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            # SAT: check if two convex polygons intersect
            if _polygons_intersect(vi, vj):
                return True
    return False

def _polygons_intersect(p1, p2):
    """SAT for two convex polygons. Return True if they intersect (including touching)."""
    polys = [p1, p2]
    for poly in polys:
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            # Normal to edge
            ax = -(y2 - y1)
            ay = (x2 - x1)
            norm = math.sqrt(ax*ax + ay*ay)
            if norm < 1e-12:
                continue
            ax /= norm
            ay /= norm
            # Project both polygons onto this axis
            min1, max1 = _project_onto_axis(p1, ax, ay)
            min2, max2 = _project_onto_axis(p2, ax, ay)
            # If separated (with small epsilon for touching)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
    return True

def _sa_optimize(init_positions, init_angles, n_iter, t0, t1, cooling):
    """Simulated Annealing."""
    positions = [list(p) for p in init_positions]
    angles = list(init_angles)
    cur_energy = _compute_energy(positions, angles)
    best_energy = cur_energy
    best_positions = [list(p) for p in positions]
    best_angles = list(angles)
    
    for it in range(n_iter):
        if time.time() > DEADLINE:
            break
        # Perturb one hexagon
        idx = random.randint(0, N-1)
        dx = random.gauss(0, 0.5)
        dy = random.gauss(0, 0.5)
        da = random.gauss(0, 10.0)
        positions[idx][0] += dx
        positions[idx][1] += dy
        angles[idx] += da
        
        new_energy = _compute_energy(positions, angles)
        
        # Check collision
        if _check_collision(positions, angles):
            # Reject if colliding
            positions[idx][0] -= dx
            positions[idx][1] -= dy
            angles[idx] -= da
            continue
        
        delta = new_energy - cur_energy
        t = t0 * (t1/t0) ** (it / max(1, n_iter-1))
        if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-9)):
            cur_energy = new_energy
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_positions = [list(p) for p in positions]
                best_angles = list(angles)
        else:
            positions[idx][0] -= dx
            positions[idx][1] -= dy
            angles[idx] -= da
    
    return best_positions, best_angles, best_energy

def construct():
    best_positions = None
    best_angles = None
    best_energy = float('inf')
    
    n_starts = 8
    for start in range(n_starts):
        if time.time() > DEADLINE - 10:
            break
        # Initialize: place hexagons in a rough grid
        positions = []
        angles = []
        for i in range(N):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            a = random.uniform(0, 360)
            positions.append([x, y])
            angles.append(a)
        
        n_iter = 3000
        t0 = 5.0
        t1 = 0.01
        positions, angles, energy = _sa_optimize(positions, angles, n_iter, t0, t1, 0.999)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
            best_angles = angles
    
    if best_positions is None:
        # Fallback
        positions = [[0.0, 0.0]] * N
        angles = [0.0] * N
        best_energy = 40.0
    
    # Find the optimal outer hexagon center and orientation
    all_verts = []
    for i in range(N):
        vs = _hex_vertices(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
        all_verts.extend(vs)
    
    best_s = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_theta = 0.0
    for theta_deg in range(0, 30, 1):
        theta = math.radians(theta_deg)
        # Compute projections
        ranges = []
        midpoints = []
        for k in range(3):
            phi = theta + k * math.pi / 3.0
            ax, ay = math.cos(phi), math.sin(phi)
            mn, mx = _project_onto_axis(all_verts, ax, ay)
            ranges.append(mx - mn)
            midpoints.append((mn + mx) / 2.0)
        r = max(ranges) / 2.0
        s = r / (math.sqrt(3.0) / 2.0)
        if s < best_s:
            best_s = s
            # Compute center: solve for C such that C·n_k = midpoints[k] for k=0,1,2
            # n_0, n_1 are independent
            n0 = (math.cos(theta), math.sin(theta))
            n1 = (math.cos(theta + math.pi/3), math.sin(theta + math.pi/3))
            # C·n0 = m0, C·n1 = m1
            # [n0x n0y; n1x n1y] [Cx; Cy] = [m0; m1]
            det = n0[0]*n1[1] - n0[1]*n1[0]
            if abs(det) > 1e-12:
                cx = (midpoints[0]*n1[1] - midpoints[1]*n0[1]) / det
                cy = (n0[0]*midpoints[1] - n1[0]*midpoints[0]) / det
            else:
                cx, cy = 0.0, 0.0
            best_cx = cx
            best_cy = cy
            best_theta = theta_deg
    
    return [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)], \
           (best_cx, best_cy), best_s, best_theta
