import time
import math
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # --- Hexagon geometry helpers ---
    def hex_support(theta, phi):
        """
        Support function of a unit regular hexagon with rotation theta
        in direction phi.
        For a unit regular hexagon (side 1), the vertices are at distance
        1 from center at angles theta + k*pi/3.
        The support in direction phi is max_k cos(phi - (theta + k*pi/3)).
        This equals cos(min_k |phi - (theta + k*pi/3)| mod symmetry).
        Due to 6-fold symmetry, we can reduce phi - theta to [-pi/6, pi/6].
        """
        delta = (phi - theta) % (math.pi / 3)
        # Map to [-pi/6, pi/6]
        if delta > math.pi / 6:
            delta -= math.pi / 3
        return math.cos(delta)
    
    def max_side_length(positions, angles):
        """
        Given positions (list of (x,y)) and angles (list of radians),
        compute the minimum side length of a regular hexagon (centered at origin,
        rotatable) that contains all inner hexagons.
        
        We minimize over the outer hexagon's rotation angle alpha.
        For a given alpha, the outer hexagon (side S, rotation alpha) contains
        a point p if for all 6 outward normals (at angles alpha + k*pi/3),
        p·n <= S * (apothem of outer hex) = S * cos(pi/6) = S * sqrt(3)/2.
        
        Actually, the distance from center to a side (apothem) of a regular
        hexagon with side S is S * sqrt(3)/2.
        So we need: for each inner hex i, for each of its vertices v,
        v·n_k <= S * sqrt(3)/2 for all k=0..5.
        Equivalently, the max over all inner vertices and all normals of
        v·n_k must be <= S * sqrt(3)/2.
        So S >= (2/sqrt(3)) * max_{i, vertex j, k} v_{i,j} · n_k.
        
        We can compute this for a given alpha.
        """
        outer_normals = [alpha + k * math.pi / 3 for k in range(6)]
        max_dot = 0.0
        for pos, ang in zip(positions, angles):
            cx, cy = pos
            # Vertices of inner hex
            for k in range(6):
                v_angle = ang + k * math.pi / 3
                vx = cx + math.cos(v_angle)
                vy = cy + math.sin(v_angle)
                for n_angle in outer_normals:
                    dot = vx * math.cos(n_angle) + vy * math.sin(n_angle)
                    if dot > max_dot:
                        max_dot = dot
        return 2.0 * max_dot / math.sqrt(3)
    
    def objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, alpha]
        where a_i are inner rotations (radians), alpha is outer rotation.
        """
        n_inner = N
        positions = []
        angles = []
        for i in range(n_inner):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            positions.append((x, y))
            angles.append(a)
        alpha = params[3 * n_inner]
        return max_side_length(positions, angles)
    
    # --- Initial guess: hexagonal cluster ---
    # Center 1 hex at origin, 6 around it, 4 more in a second ring.
    # A compact arrangement of 11 hexagons.
    # Let's use a known compact packing: center + 6 around + 4 more.
    # Actually, for N=11, a good starting point is a hexagonal number pattern
    # with one missing from the second ring, or a linear chain if cluster is too big.
    
    # Try a hexagonal cluster: positions for a "hexagon of hexagons" with side 2
    # has 7 hexagons. For 11, we need a bit more.
    # Let's just use a grid or a known compact set.
    
    # Simpler initial guess: a linear row is bad (score 1.0). 
    # Let's try a 2x2x2x... arrangement.
    # Hexagonal packing: center at (0,0), then 6 at distance 2 (touching), etc.
    
    # Let's generate a compact set of 11 points in hex grid.
    # Hex grid neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Distance between centers of touching hexagons is 2 (side=1).
    
    # Coordinates for a hexagonal cluster of 11:
    # Use axial coordinates (q, r)
    # Center (0,0)
    # Ring 1: 6 points
    # Ring 2: we need 4 more.
    
    # Let's just start with a random cluster and let optimization fix it.
    # But a good initial guess helps.
    
    # Initial guess: 
    # A row of 11 is bad. 
    # A 3x3 grid of hexagons has 9. Add 2 more.
    # Let's try a compact shape.
    
    # Axial coordinates for 11 hexagons:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  [6 neighbors]
    # (2,-1), (1,1), (-2,1), (-2,0) [4 more]
    
    hex_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-2, 0)
    ]
    
    # Convert axial to cartesian
    # x = 2*q + r
    # y = sqrt(3)*r
    # (This is for pointy-top. For flat-top, swap or rotate.)
    # Let's use flat-top:
    # x = 3/2 * q
    # y = sqrt(3)/2 * q + sqrt(3) * r
    
    init_positions = []
    for q, r in hex_coords:
        x = 1.5 * q
        y = math.sqrt(3) / 2 * q + math.sqrt(3) * r
        init_positions.append((x, y))
    
    init_angles = [0.0] * N
    init_alpha = 0.0
    
    x0 = []
    for p, a in zip(init_positions, init_angles):
        x0.append(p[0])
        x0.append(p[1])
        x0.append(a)
    x0.append(init_alpha)
    
    x0 = np.array(x0)
    
    # --- Optimization ---
    # Bounds: positions within [-10, 10], angles [0, 2pi]
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 2 * math.pi))
    bounds.append((0, 2 * math.pi))  # outer rotation
    
    best_result = None
    best_value = float('inf')
    
    # Try Nelder-Mead first (fast for small N)
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_value:
                best_value = res.fun
                best_result = res.x
        except Exception as e:
            pass
    
    # Try a few random restarts with Nelder-Mead if time permits
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb initial guess
            x_perturbed = x0.copy()
            for i in range(len(x_perturbed)):
                if i < 3 * N:
                    if i % 3 == 0 or i % 3 == 1:
                        x_perturbed[i] += random.gauss(0, 1.0)
                    else:
                        x_perturbed[i] = random.uniform(0, 2 * math.pi)
                else:
                    x_perturbed[i] = random.uniform(0, 2 * math.pi)
            
            try:
                res = minimize(objective, x_perturbed, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
                if res.fun < best_value:
                    best_value = res.fun
                    best_result = res.x
            except Exception as e:
                pass
    
    # If we have a good result, use it. Otherwise, fallback to seed.
    if best_result is not None and best_value < 40.0:
        # Extract results
        inner = []
        for i in range(N):
            x = best_result[i * 3]
            y = best_result[i * 3 + 1]
            a = best_result[i * 3 + 2]
            # Convert radians to degrees
            a_deg = math.degrees(a) % 360.0
            inner.append([x, y, a_deg])
        
        side = best_value
        # Outer hexagon is centered at (0,0) with rotation alpha
        outer_center = (0.0, 0.0)
        outer_side = side
        outer_rotation = best_result[3 * N]  # radians
        
        # The interface expects: inner, (x, y), side, angle
        # angle is the rotation of the outer hexagon in degrees?
        # The task says: "outer hexagon center (x, y), side length and rotation"
        # So we return the rotation angle in degrees.
        outer_angle_deg = math.degrees(outer_rotation) % 360.0
        
        return inner, outer_center, outer_side, outer_angle_deg
    
    # Fallback: seed result
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
