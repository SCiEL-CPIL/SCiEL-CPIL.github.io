"""Simulated annealing packing of 12 unit hexagons in a regular hexagon."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 138.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices(theta):
    # Regular hexagon, side 1, circumradius 1, rotated by theta
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _norms(theta):
    # 6 edge-normal directions of a regular hexagon oriented by theta
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _bounding_side_fast(pts, theta):
    # side length of smallest regular hexagon (orientation theta, centered at origin)
    # containing the set of points pts.
    nrm = _norms(theta)
    P = pts @ nrm.T
    req = P.max(axis=0)
    return 2.0 * float(np.max(req)) / math.sqrt(3.0)

def _bounding_side_centers(centers, angles, outer_theta):
    """Compute bounding side length from centers and angles directly."""
    norms = _norms(outer_theta)
    min_proj = np.full(6, np.inf)
    max_proj = np.full(6, -np.inf)
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
    widths = max_proj - min_proj
    return float(np.max(widths) / math.sqrt(3.0))

def _compute_center(pts, theta):
    # Compute optimal center for bounding hexagon with orientation theta
    nrm = _norms(theta)
    P = pts @ nrm.T
    c = (P.max(axis=0) + P.min(axis=0)) / 2.0
    c_world = nrm.T @ c
    return c_world

def _sat_overlap(A, B):
    """Fast SAT check for two convex hexagons. Returns True if they overlap."""
    for poly in (A, B):
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            projA = A @ normal
            projB = B @ normal
            if projA.max() < projB.min() - 1e-9 or projB.max() < projA.min() - 1e-9:
                return False
    return True

def _is_valid(centers, angles, outer_side=None, outer_theta=None, outer_center=None):
    """Check if all hexagons are non-overlapping and (optionally) contained."""
    verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
    
    # Check pairwise overlaps via SAT
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                return False
    
    # Check containment if outer params given
    if outer_side is not None and outer_theta is not None and outer_center is not None:
        nrm = _norms(outer_theta)
        c = np.array(outer_center) @ nrm.T
        for k in range(N):
            P = verts[k] @ nrm.T
            for face in range(6):
                if (P[:, face] - c[face]).max() > outer_side * math.sqrt(3.0) / 2.0 + 1e-9:
                    return False
    
    return True

# ---------------------------------------------------------------------------
# Seed generators
# ---------------------------------------------------------------------------
def _seed_lattice():
    """Triangular lattice seed, centered at origin."""
    d = 2.0 * math.sqrt(3.0) + 0.001
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    pts = []
    for i in range(-1, 3):
        for j in range(-1, 2):
            pts.append(i * a1 + j * a2)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

def _seed_hex_cluster():
    """Hexagonal cluster: center + 6 neighbors + 5 more."""
    d = 2.0 * math.sqrt(3.0) + 0.001
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
              (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
    pts = []
    for (i, j) in coords:
        pts.append(i * a1 + j * a2)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

def _seed_ring():
    """Ring of 6 around center + 5 more on outer ring."""
    d = 2.0 * math.sqrt(3.0) + 0.001
    pts = [[0.0, 0.0]]
    for k in range(6):
        ang = k * math.pi / 3.0
        pts.append([d * math.cos(ang), d * math.sin(ang)])
    for k in range(5):
        ang = k * 2 * math.pi / 5.0 + 0.3
        r = 2 * d
        pts.append([r * math.cos(ang), r * math.sin(ang)])
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

def _seed_random():
    """Random positions in a disk of radius 5."""
    pts = []
    for _ in range(N):
        r = random.uniform(0, 4.0)
        theta = random.uniform(0, 2 * math.pi)
        pts.append([r * math.cos(theta), r * math.sin(theta)])
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

# ---------------------------------------------------------------------------
# Physics relaxation with rotation
# ---------------------------------------------------------------------------
def _physics_relax(centers, angles, outer_side, outer_theta, outer_center, steps=300, dt=0.05):
    """Physics-based relaxation with rotation to resolve overlaps and compact."""
    centers = centers.copy()
    angles = angles.copy()
    vel = np.zeros((N, 2))
    omega = np.zeros(N)
    
    for s in range(steps):
        if time.time() > DEADLINE:
            break
        verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
        
        # Damping
        vel *= 0.9
        omega *= 0.9
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                A = verts[i]
                B = verts[j]
                # Check overlap and find minimum translation
                overlapping = True
                min_pen = float('inf')
                best_axis = None
                for poly in (A, B):
                    for k in range(6):
                        p1 = poly[k]
                        p2 = poly[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        nl = math.hypot(normal[0], normal[1])
                        if nl < 1e-9:
                            continue
                        normal = normal / nl
                        projA = A @ normal
                        projB = B @ normal
                        sep = min(projA.max() - projB.min(), projB.max() - projA.min())
                        if sep >= -1e-9:
                            overlapping = False
                            break
                        if abs(sep) < abs(min_pen):
                            min_pen = sep
                            best_axis = normal
                    if not overlapping:
                        break
                if overlapping and best_axis is not None:
                    f = -min_pen * 0.5
                    vel[i] += f * best_axis
                    vel[j] -= f * best_axis
                    # Small rotation torque to help escape
                    omega[i] += random.uniform(-0.1, 0.1)
                    omega[j] += random.uniform(-0.1, 0.1)
        
        # Containment force
        nrm = _norms(outer_theta)
        c = np.array(outer_center) @ nrm.T
        for i in range(N):
            P = verts[i] @ nrm.T
            for face in range(6):
                exceed = (P[:, face] - c[face]).max() - outer_side * math.sqrt(3.0) / 2.0
                if exceed > 0:
                    vel[i] -= 0.5 * exceed * nrm[face]
        
        # Mild attraction to center
        for i in range(N):
            vel[i] += -0.01 * centers[i]
        
        # Integrate
        centers += vel * dt
        angles += omega * dt
    
    return centers, angles

# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------
def _simulated_annealing(centers, angles, outer_theta, max_iters=20000, T_start=0.5, T_end=0.005, alpha=0.999):
    """Simulated annealing to minimize bounding hexagon side length."""
    centers = centers.copy()
    angles = angles.copy()
    
    # Initial evaluation
    side = _bounding_side_centers(centers, angles, outer_theta)
    if not _is_valid(centers, angles):
        # Try physics fix
        centers, angles = _physics_relax(centers, angles, side, outer_theta, (0.0, 0.0), steps=100)
        side = _bounding_side_centers(centers, angles, outer_theta)
        if not _is_valid(centers, angles):
            return None, None, float('inf')
    
    best_side = side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    T = T_start
    step_size = 0.5
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Propose move
        i = random.randint(0, N - 1)
        move_type = random.random()
        
        old_cx, old_cy, old_a = centers[i][0], centers[i][1], angles[i]
        
        if move_type < 0.7:
            # Move center
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            centers[i][0] += dx
            centers[i][1] += dy
        else:
            # Rotate
            da = random.gauss(0, step_size * 0.5)
            angles[i] += da
        
        # Evaluate
        new_side = _bounding_side_centers(centers, angles, outer_theta)
        
        # Check validity (non-overlapping)
        if not _is_valid(centers, angles):
            # Revert
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
            T *= alpha
            step_size *= 0.9995
            continue
        
        # Accept/reject
        if new_side < side or random.random() < math.exp(-(new_side - side) / T):
            side = new_side
            if side < best_side:
                best_side = side
                best_centers = centers.copy()
                best_angles = angles.copy()
        else:
            # Revert
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
        
        T = max(T * alpha, T_end)
        step_size *= 0.9995
    
    return best_centers, best_angles, best_side

# ---------------------------------------------------------------------------
# Outer angle/center refinement
# ---------------------------------------------------------------------------
def _refine_outer(centers, angles):
    """Sweep outer angle and compute optimal center; return best (side, center, angle)."""
    all_pts = np.vstack([_hex_vertices(angles[k]) + centers[k] for k in range(N)])
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for ang in np.linspace(0.0, math.pi / 3.0, 181):
        nrm = _norms(ang)
        P = all_pts @ nrm.T
        c_normal = (P.max(axis=0) + P.min(axis=0)) / 2.0
        c_world = nrm.T @ c_normal
        side = 2.0 * float(np.max(P.max(axis=0) - c_normal)) / math.sqrt(3.0)
        if side < best_side - 1e-12:
            best_side = side
            best_center = (float(c_world[0]), float(c_world[1]))
            best_angle = float(ang)
    
    return best_side, best_center, best_angle

# ---------------------------------------------------------------------------
# Local refinement: nudge individual hexagons to reduce bounding side
# ---------------------------------------------------------------------------
def _local_refine(centers, angles, outer_theta, iterations=200):
    """Local search: try small moves to reduce bounding side."""
    centers = centers.copy()
    angles = angles.copy()
    side = _bounding_side_centers(centers, angles, outer_theta)
    best_side = side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    for it in range(iterations):
        if time.time() > DEADLINE:
            break
        i = random.randint(0, N - 1)
        old_cx, old_cy, old_a = centers[i][0], centers[i][1], angles[i]
        
        # Small random move
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        da = random.gauss(0, 0.05)
        centers[i][0] += dx
        centers[i][1] += dy
        angles[i] += da
        
        new_side = _bounding_side_centers(centers, angles, outer_theta)
        
        if not _is_valid(centers, angles) or new_side > side:
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
        else:
            side = new_side
            if side < best_side:
                best_side = side
                best_centers = centers.copy()
                best_angles = angles.copy()
    
    return best_centers, best_angles, best_side

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    random.seed(42)
    np.random.seed(42)

    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def _consider(centers, angles, side, center, angle):
        nonlocal best_side, best_centers, best_angles, best_center, best_angle
        if side < best_side:
            best_side = side
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_center = center
            best_angle = angle

    # Try multiple seeds and outer angles
    seed_funcs = [_seed_lattice, _seed_hex_cluster, _seed_ring, _seed_random]
    outer_angles = [0.0, math.pi / 6.0, math.pi / 3.0 * 0.5, math.pi / 3.0 * 0.25]

    for seed_func in seed_funcs:
        for oa in outer_angles:
            if time.time() > DEADLINE:
                break
            
            centers = seed_func()
            angles = np.zeros(N)
            
            # Physics relaxation to resolve initial overlaps
            centers, angles = _physics_relax(centers, angles, 8.0, oa, (0.0, 0.0), steps=200)
            
            # Simulated annealing
            bc, ba, bs = _simulated_annealing(centers, angles, oa, max_iters=15000)
            
            if bc is not None:
                # Local refinement
                lc, la, ls = _local_refine(bc, ba, oa, iterations=100)
                
                # Refine outer
                side, center, angle = _refine_outer(lc, la)
                
                _consider(lc, la, side, center, angle)
    
    # Final fine sweep on best configuration
    if best_centers is not None:
        side, center, angle = _refine_outer(best_centers, best_angles)
        _consider(best_centers, best_angles, side, center, angle)

    # Fallback
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner = []
    for k in range(N):
        inner.append([float(best_centers[k][0]), float(best_centers[k][1]), float(math.degrees(best_angles[k]))])
    
    return inner, best_center, best_side, math.degrees(best_angle)
