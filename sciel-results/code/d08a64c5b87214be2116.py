"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 145.0  # Safety margin before the 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(theta):
    """
    Returns vertices of a regular hexagon with side length 1, circumradius 1,
    centered at origin, rotated by theta.
    """
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _sat_overlap(A, B):
    """
    Fast SAT check for two convex polygons A and B (arrays of (x,y)).
    Returns True if they overlap.
    """
    # Check axes from A
    for i in range(6):
        p1 = A[i]
        p2 = A[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        projA = A @ normal
        projB = B @ normal
        if projA.max() < projB.min() or projB.max() < projA.min():
            return False
    # Check axes from B
    for i in range(6):
        p1 = B[i]
        p2 = B[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        projA = A @ normal
        projB = B @ normal
        if projA.max() < projB.min() or projB.max() < projA.min():
            return False
    return True

def _bounding_side_fast(centers, angles, outer_theta):
    """
    Calculates the minimum side length of a regular hexagon (orientation outer_theta)
    that contains all inner hexagons defined by centers and angles.
    """
    # Normals for the outer hexagon
    norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    
    min_proj = np.inf
    max_proj = -np.inf
    
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        # Project vertices onto the 6 normal directions
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
        
    widths = max_proj - min_proj
    # Side length s relates to width w by w = s * sqrt(3)
    return float(np.max(widths) / math.sqrt(3.0))

def _is_valid(centers, angles, tol=1e-9):
    """
    Checks if the configuration has no overlaps.
    """
    verts = [_hex_vertices(angles[i]) + centers[i] for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                return False
    return True

def _compute_optimal_center(centers, angles, outer_theta, side):
    """
    Compute the center of the outer hexagon that best contains the inner hexagons.
    """
    norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    
    min_proj = np.inf
    max_proj = -np.inf
    
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
    
    # The center in normal space is the midpoint of min and max projections
    center_normal = (max_proj + min_proj) / 2.0
    
    # Convert back to Cartesian using the inverse of the normals matrix
    # For a regular hexagon, the normals span 6 directions at 60 degrees apart
    # We can solve for the center by finding a point c such that n_k . c = center_normal[k]
    # This is an overdetermined system, so we use least squares
    try:
        c, _, _, _ = np.linalg.lstsq(norms, center_normal, rcond=None)
        return tuple(float(x) for x in c)
    except:
        # Fallback to centroid
        all_verts = []
        for i in range(N):
            all_verts.append(_hex_vertices(angles[i]) + centers[i])
        all_verts = np.vstack(all_verts)
        centroid = all_verts.mean(axis=0)
        return tuple(float(x) for x in centroid)

def _physics_relax(centers, angles, side, outer_theta, steps=200, dt=0.05):
    """
    Simple physics simulation to resolve overlaps and compact the pack.
    Returns updated centers and angles.
    """
    centers = np.array(centers, dtype=float)
    angles = np.array(angles, dtype=float)
    vel = np.zeros((N, 2))
    omega = np.zeros(N)
    
    for s in range(steps):
        # Forces
        for i in range(N):
            # Attraction to center
            vel[i] += -0.01 * centers[i]
            # Damping
            vel[i] *= 0.9
            omega[i] *= 0.9
            
        # Collision resolution
        for i in range(N):
            vi = _hex_vertices(angles[i]) + centers[i]
            for j in range(i + 1, N):
                vj = _hex_vertices(angles[j]) + centers[j]
                if _sat_overlap(vi, vj):
                    # Find minimum translation vector
                    min_pen = np.inf
                    min_axis = None
                    for k in range(6):
                        p1 = vi[k]
                        p2 = vi[(k + 1) % 6]
                        e = p2 - p1
                        n = np.array([-e[1], e[0]])
                        nl = math.hypot(n[0], n[1])
                        if nl < 1e-9: continue
                        n = n / nl
                        pa = vi @ n
                        pb = vj @ n
                        # Separation
                        sep = min(pa.max() - pb.min(), pb.max() - pa.min())
                        if sep < 0 and abs(sep) < abs(min_pen):
                            min_pen = sep
                            min_axis = n
                    if min_axis is not None:
                        f = -min_pen * 0.5
                        vel[i] += f * min_axis
                        vel[j] -= f * min_axis
                        # Small rotation torque to help escape
                        omega[i] += random.uniform(-0.1, 0.1)
                        omega[j] += random.uniform(-0.1, 0.1)

        # Integrate
        centers += vel * dt
        angles += omega * dt

    return centers, angles

def _simulated_annealing(initial_centers, initial_angles, outer_theta, T_start=0.5, T_end=0.01, alpha=0.999):
    """
    Simulated Annealing to minimize the bounding side length.
    """
    centers = np.array(initial_centers, dtype=float)
    angles = np.array(initial_angles, dtype=float)
    
    current_side = _bounding_side_fast(centers, angles, outer_theta)
    if not _is_valid(centers, angles):
        # If initial is invalid, try to fix with physics first
        c, a = _physics_relax(centers, angles, current_side, outer_theta, steps=100)
        centers, angles = c, a
        current_side = _bounding_side_fast(centers, angles, outer_theta)
        if not _is_valid(centers, angles):
            return None, None, float('inf')

    best_side = current_side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    T = T_start
    step_size = 0.5
    
    while T > T_end:
        # Pick a random hexagon to move
        i = random.randrange(N)
        
        # Random perturbation
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, step_size * 0.5)
        
        old_cx, old_cy, old_a = centers[i][0], centers[i][1], angles[i]
        
        centers[i][0] += dx
        centers[i][1] += dy
        angles[i] += da
        
        # Check validity
        if not _is_valid(centers, angles):
            # Revert
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
            T *= alpha
            continue
        
        new_side = _bounding_side_fast(centers, angles, outer_theta)
        
        # Acceptance criterion
        if new_side < current_side or random.random() < math.exp(-(new_side - current_side) / T):
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_centers = centers.copy()
                best_angles = angles.copy()
        else:
            # Revert
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
            
        T *= alpha
        # Reduce step size slightly
        step_size *= 0.9995
        
    return best_centers, best_angles, best_side

def _rigid_body_dynamics(centers, angles, outer_theta, steps=300, dt=0.02, 
                         repulsion_strength=1.5, attraction_strength=0.03):
    """
    Rigid-body dynamics simulation using symplectic Euler integration.
    Repulsive forces between overlapping hexagons and attractive forces toward
    the outer boundary's incenter to naturally resolve collisions and settle
    into a dense equilibrium.
    """
    centers = np.array(centers, dtype=float)
    angles = np.array(angles, dtype=float)
    vel = np.zeros((N, 2))
    omega = np.zeros(N)
    
    for step in range(steps):
        # Reset forces
        forces = np.zeros((N, 2))
        torques = np.zeros(N)
        
        # Repulsive forces between overlapping hexagons
        for i in range(N):
            vi = _hex_vertices(angles[i]) + centers[i]
            for j in range(i + 1, N):
                vj = _hex_vertices(angles[j]) + centers[j]
                if _sat_overlap(vi, vj):
                    # Find minimum translation vector
                    min_pen = np.inf
                    min_axis = None
                    for k in range(6):
                        p1 = vi[k]
                        p2 = vi[(k + 1) % 6]
                        e = p2 - p1
                        n = np.array([-e[1], e[0]])
                        nl = math.hypot(n[0], n[1])
                        if nl < 1e-9: continue
                        n = n / nl
                        pa = vi @ n
                        pb = vj @ n
                        sep = min(pa.max() - pb.min(), pb.max() - pa.min())
                        if sep < 0 and abs(sep) < abs(min_pen):
                            min_pen = sep
                            min_axis = n
                    
                    if min_axis is not None:
                        f = -min_pen * repulsion_strength
                        forces[i] += f * min_axis
                        forces[j] -= f * min_axis
                        # Small torque to help escape
                        torques[i] += random.uniform(-0.05, 0.05)
                        torques[j] += random.uniform(-0.05, 0.05)
        
        # Attractive forces toward the outer boundary's incenter (origin)
        for i in range(N):
            forces[i] += -attraction_strength * centers[i]
        
        # Damping
        vel *= 0.95
        omega *= 0.95
        
        # Symplectic Euler integration
        vel += forces * dt
        omega += torques * dt
        centers += vel * dt
        angles += omega * dt
    
    return centers, angles

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_inner = None
    best_outer = (0.0, 0.0)
    best_theta = 0.0

    # Define initializers
    def init_lattice():
        d = 2.0 * math.sqrt(3.0)
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        pts = []
        # 4x3 block
        for i in range(-1, 3):
            for j in range(-1, 2):
                p = i * a1 + j * a2
                pts.append(p)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        return [[float(p[0]), float(p[1])] for p in pts], [0.0] * N

    def init_ring():
        pts = [[0.0, 0.0]]
        r = 2.0 * math.sqrt(3.0)
        for i in range(11):
            angle = 2 * math.pi * i / 11
            pts.append([r * math.cos(angle), r * math.sin(angle)])
        return pts, [0.0] * N

    def init_random():
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-2, 2), random.uniform(-2, 2)])
        return pts, [random.uniform(0, math.pi/3) for _ in range(N)]

    def init_compact_cluster():
        """Compact cluster initializer with hexagonal symmetry."""
        pts = [[0.0, 0.0]]
        d = 2.0 * math.sqrt(3.0)
        # First ring: 6 hexagons
        for k in range(6):
            angle = k * math.pi / 3.0
            pts.append([d * math.cos(angle), d * math.sin(angle)])
        # Second ring: 5 more hexagons (partial)
        d2 = 2 * d
        for k in range(5):
            angle = k * 2 * math.pi / 5 + math.pi / 5
            pts.append([d2 * math.cos(angle), d2 * math.sin(angle)])
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        return [[float(p[0]), float(p[1])] for p in pts], [0.0] * N

    initializers = [init_lattice, init_ring, init_random, init_compact_cluster, init_random]

    for init_func in initializers:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        centers, angles = init_func()
        
        # Stage 1: Rigid-body dynamics to resolve overlaps and compact
        c1, a1 = _rigid_body_dynamics(centers, angles, 0.0, steps=300, dt=0.02)
        
        # Stage 2: Simulated Annealing for fine tuning
        c2, a2, side_sa = _simulated_annealing(c1, a1, 0.0, T_start=0.2, T_end=0.005)
        
        if c2 is None:
            c2, a2 = c1, a1
            side_sa = _bounding_side_fast(c1, a1, 0.0)
        
        # Stage 3: Try multiple outer orientations
        for outer_theta in np.linspace(0, math.pi/3, 13):
            s = _bounding_side_fast(c2, a2, outer_theta)
            if s < best_side:
                # Compute optimal center
                center = _compute_optimal_center(c2, a2, outer_theta, s)
                
                # Verify the configuration with the computed center
                # Shift all centers so the outer hexagon is centered at the computed center
                shifted_centers = c2 - np.array(center)
                
                # Re-check validity after shifting (shouldn't change, but verify)
                if _is_valid(shifted_centers, a2):
                    best_inner = [[float(shifted_centers[i][0]), float(shifted_centers[i][1]), 
                                   float(math.degrees(a2[i]))] for i in range(N)]
                    best_outer = center
                    best_side = s
                    best_theta = outer_theta

    if best_inner is None:
        # Fallback
        centers, angles = init_lattice()
        c_rel, a_rel = _rigid_body_dynamics(centers, angles, 0.0, steps=500, dt=0.02)
        best_inner = [[float(c_rel[i][0]), float(c_rel[i][1]), 0.0] for i in range(N)]
        best_side = _bounding_side_fast(c_rel, a_rel, 0.0)
        best_outer = (0.0, 0.0)
        best_theta = 0.0

    return best_inner, best_outer, best_side, math.degrees(best_theta)

# EVOLVE-BLOCK-END
