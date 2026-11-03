"""
Packing 12 unit regular hexagons into a regular hexagon.
Strategy:
1. Fast SAT overlap check.
2. Physics-based relaxation to find a valid, compact configuration.
3. Nelder-Mead local optimization to minimize the outer hexagon side length.
4. Final verification and center/orientation optimization.
"""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before the 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(theta, center=(0.0, 0.0), side=1.0):
    """
    Returns vertices of a regular hexagon with given side length,
    centered at 'center', rotated by 'theta'.
    """
    cx, cy = center
    # Vertices of a unit hexagon (side 1) centered at origin, rotated by theta
    # The distance from center to vertex is 'side' for a regular hexagon? 
    # No, for a regular hexagon of side s, circumradius R = s.
    # Apothem (distance to side) = s * sqrt(3) / 2.
    # The problem states "side length 1". So R=1.
    verts = np.array([
        [math.cos(theta + k * math.pi / 3.0), math.sin(theta + k * math.pi / 3.0)] 
        for k in range(6)
    ])
    return verts + np.array([cx, cy])

def _sat_overlap_depth(A, B):
    """
    Checks overlap between two convex polygons A and B (arrays of (x,y)).
    Returns (overlap_depth, axis) if overlapping, else (0.0, None).
    overlap_depth is the minimum translation distance to separate them.
    """
    min_pen = np.inf
    min_axis = None
    
    # Check axes from A
    for i in range(len(A)):
        p1 = A[i]
        p2 = A[(i + 1) % len(A)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = math.hypot(normal[0], normal[1])
        if nl < 1e-9: continue
        normal = normal / nl
        
        projA = A @ normal
        projB = B @ normal
        
        maxA = projA.max()
        minA = projA.min()
        maxB = projB.max()
        minB = projB.min()
        
        # If separated
        if maxA < minB or maxB < minA:
            return 0.0, None
        
        # Calculate penetration depth in this direction
        # We want the minimum shift to separate.
        # If A is to the left of B (maxA < minB is false, so they overlap)
        # The overlap is min(maxA, maxB) - max(minA, minB)
        # But we need the direction of separation.
        # If maxA < maxB and minA < minB, A is more to the left.
        # Separation distance is maxA - minB (if A is left) or maxB - minA (if B is left).
        # Actually, the penetration depth is min(maxA, maxB) - max(minA, minB) is the *overlap* length.
        # To separate, we need to move by that amount?
        # Let's check:
        # Interval A: [minA, maxA]
        # Interval B: [minB, maxB]
        # Overlap: [max(minA, minB), min(maxA, maxB)]
        # Depth = min(maxA, maxB) - max(minA, minB)
        
        depth = min(maxA, maxB) - max(minA, minB)
        if depth < min_pen:
            min_pen = depth
            # Determine direction: from A to B or B to A?
            # We just need the magnitude for penalty.
            # For physics, we need the vector.
            # If maxA < maxB, A is more to the left (assuming normal points right-ish).
            # Let's define the axis as pointing from A to B.
            # Center A vs Center B projection.
            cA = A.mean(axis=0) @ normal
            cB = B.mean(axis=0) @ normal
            if cB > cA:
                min_axis = normal
            else:
                min_axis = -normal

    # Check axes from B
    for i in range(len(B)):
        p1 = B[i]
        p2 = B[(i + 1) % len(B)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = math.hypot(normal[0], normal[1])
        if nl < 1e-9: continue
        normal = normal / nl
        
        projA = A @ normal
        projB = B @ normal
        
        maxA = projA.max()
        minA = projA.min()
        maxB = projB.max()
        minB = projB.min()
        
        if maxA < minB or maxB < minA:
            return 0.0, None
        
        depth = min(maxA, maxB) - max(minA, minB)
        if depth < min_pen:
            min_pen = depth
            cA = A.mean(axis=0) @ normal
            cB = B.mean(axis=0) @ normal
            if cB > cA:
                min_axis = normal
            else:
                min_axis = -normal

    if min_pen == np.inf:
        return 0.0, None
    return min_pen, min_axis

def _outer_hex_side(centers, angles, outer_theta=0.0):
    """
    Calculates the minimum side length of a regular hexagon (orientation outer_theta)
    that contains all inner hexagons.
    """
    # Normals for the outer hexagon sides
    # The sides are at angles outer_theta + pi/6 + k*pi/3
    # The normal to the side is perpendicular to the edge.
    # Edge direction is outer_theta + k*pi/3.
    # Normal direction is outer_theta + k*pi/3 + pi/2.
    # Wait, for a regular hexagon centered at origin:
    # Vertices at angles theta + k*pi/3.
    # Sides are between vertices.
    # The normal to the side between vertex k and k+1 is at angle theta + (k+0.5)*pi/3.
    # Which is theta + pi/6 + k*pi/3.
    
    norms = np.array([
        [math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] 
        for k in range(6)
    ])
    
    min_proj = np.inf * np.ones(6)
    max_proj = -np.inf * np.ones(6)
    
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
        
    widths = max_proj - min_proj
    # Width w = s * sqrt(3)
    return float(np.max(widths) / math.sqrt(3.0))

def _is_valid(centers, angles, tol=1e-9):
    """
    Checks if the configuration has no overlaps.
    """
    verts = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            depth, _ = _sat_overlap_depth(verts[i], verts[j])
            if depth > tol:
                return False
    return True

def _cost_function(params, outer_theta=0.0, penalty_weight=100.0):
    """
    Cost function for optimization.
    params: array of shape (3*N,) -> [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = params[0::3].reshape(-1, 2)
    angles = params[1::3]
    
    side = _outer_hex_side(centers, angles, outer_theta)
    
    # Calculate overlap penalty
    verts = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth, _ = _sat_overlap_depth(verts[i], verts[j])
            if depth > 0:
                overlap_penalty += depth ** 2
    
    return side + penalty_weight * overlap_penalty

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_theta = 0.0

    # Initializer: Hexagonal lattice
    def init_lattice():
        d = 2.0 * math.sqrt(3.0) # Distance between centers of touching hexagons
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        pts = []
        # 4x3 block is 12 points
        for i in range(-1, 3):
            for j in range(-1, 2):
                p = i * a1 + j * a2
                pts.append(p)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        centers = pts.tolist()
        angles = [0.0] * N
        return centers, angles

    def init_random():
        centers = []
        angles = []
        for _ in range(N):
            centers.append([random.uniform(-2, 2), random.uniform(-2, 2)])
            angles.append(random.uniform(0, math.pi/3))
        return centers, angles

    # Physics Relaxation
    def physics_relax(centers, angles, steps=500, dt=0.01):
        centers = np.array(centers, dtype=float)
        angles = np.array(angles, dtype=float)
        vel = np.zeros((N, 2))
        omega = np.zeros(N)
        
        for s in range(steps):
            # Calculate forces
            forces = np.zeros((N, 2))
            torques = np.zeros(N)
            
            # Attraction to center
            for i in range(N):
                forces[i] -= 0.05 * centers[i]
                
            # Overlap repulsion
            verts = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
            for i in range(N):
                for j in range(i + 1, N):
                    depth, axis = _sat_overlap_depth(verts[i], verts[j])
                    if depth > 0 and axis is not None:
                        # Force is proportional to depth
                        f = 5.0 * depth
                        forces[i] -= f * axis
                        forces[j] += f * axis
                        # Small torque to help escape local minima
                        torques[i] += random.uniform(-0.1, 0.1) * depth
                        torques[j] += random.uniform(-0.1, 0.1) * depth
            
            # Integrate (Verlet-like)
            vel += forces * dt
            vel *= 0.9 # Damping
            omega += torques * dt
            omega *= 0.9
            
            centers += vel * dt
            angles += omega * dt
            
        return centers, angles

    # Optimization Loop
    initializers = [init_lattice, init_random, init_random, init_random]
    
    for init_func in initializers:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        centers, angles = init_func()
        
        # 1. Physics Relaxation
        c_rel, a_rel = physics_relax(centers, angles, steps=300)
        
        # 2. Nelder-Mead Optimization
        # Flatten parameters
        params = np.zeros(3 * N)
        for i in range(N):
            params[3*i] = c_rel[i][0]
            params[3*i+1] = c_rel[i][1]
            params[3*i+2] = a_rel[i]
            
        # Try a few outer thetas
        for outer_theta in [0.0, math.pi/6, math.pi/3]:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            try:
                res = minimize(
                    _cost_function, 
                    params, 
                    method='Nelder-Mead', 
                    args=(outer_theta,),
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                
                # Extract results
                p = res.x
                c_opt = p[0::3].reshape(-1, 2)
                a_opt = p[1::3]
                
                # Check validity
                if _is_valid(c_opt, a_opt):
                    side = _outer_hex_side(c_opt, a_opt, outer_theta)
                    if side < best_side:
                        best_side = side
                        best_params = p.copy()
                        best_outer_theta = outer_theta
                        
                        # Calculate optimal center for this configuration
                        # The bounding box in normal space
                        norms = np.array([
                            [math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                             math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] 
                            for k in range(6)
                        ])
                        min_proj = np.inf * np.ones(6)
                        max_proj = -np.inf * np.ones(6)
                        for i in range(N):
                            verts = _hex_vertices(a_opt[i], c_opt[i])
                            projs = verts @ norms.T
                            min_proj = np.minimum(min_proj, projs.min(axis=0))
                            max_proj = np.maximum(max_proj, projs.max(axis=0))
                        
                        # The center of the bounding hexagon is the midpoint of the projections
                        # This is an approximation. The exact center minimizes the max distance.
                        # For a regular hexagon, the center is the average of the 6 support points?
                        # Let's use the centroid of the vertices as a robust estimate.
                        all_verts = []
                        for i in range(N):
                            all_verts.append(_hex_vertices(a_opt[i], c_opt[i]))
                        all_verts = np.vstack(all_verts)
                        centroid = all_verts.mean(axis=0)
                        best_outer_center = tuple(float(x) for x in centroid)
                        
            except Exception as e:
                pass

    if best_params is None:
        # Fallback
        centers, angles = init_lattice()
        best_params = np.zeros(3 * N)
        for i in range(N):
            best_params[3*i] = centers[i][0]
            best_params[3*i+1] = centers[i][1]
            best_params[3*i+2] = angles[i]
        best_side = _outer_hex_side(np.array(centers), np.array(angles), 0.0)
        best_outer_theta = 0.0
        best_outer_center = (0.0, 0.0)

    # Finalize
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        a_deg = float(math.degrees(best_params[3*i+2]))
        inner.append([x, y, a_deg])
        
    return inner, best_outer_center, best_side, float(math.degrees(best_outer_theta))

# EVOLVE-BLOCK-END
