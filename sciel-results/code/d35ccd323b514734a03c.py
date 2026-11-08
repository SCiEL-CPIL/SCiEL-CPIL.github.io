import numpy as np
import time
import math
import random

N = 11

# Precompute R_min(alpha) for alpha in [0, pi/3]
# R_min(alpha) is the minimum distance between centers of two unit hexagons
# with relative rotation alpha such that they do not overlap.

def hex_vertices(angle=0.0):
    """Return vertices of a unit regular hexagon centered at origin, rotated by angle."""
    # Precomputed cos/sin for base angles
    # k * pi/3
    # cos(0)=1, sin(0)=0
    # cos(pi/3)=0.5, sin(pi/3)=sqrt(3)/2
    # etc.
    # But angle is variable.
    return np.array([
        [math.cos(angle + k * math.pi / 3), math.sin(angle + k * math.pi / 3)]
        for k in range(6)
    ])

def minkowski_sum_inradius(v1, v2):
    """
    Compute the inradius of the Minkowski sum of two convex polygons v1 and v2.
    """
    from scipy.spatial import ConvexHull
    sums = []
    for a in v1:
        for b in v2:
            sums.append(a + b)
    sums = np.array(sums)
    
    try:
        hull = ConvexHull(sums)
        vertices = sums[hull.vertices]
    except Exception:
        return 0.0

    min_dist = float('inf')
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        cross = abs(p1[0] * p2[1] - p1[1] * p2[0])
        length = math.sqrt(dx*dx + dy*dy)
        if length > 1e-9:
            dist = cross / length
            if dist < min_dist:
                min_dist = dist
    return min_dist

# Precompute R_min table
# Reduced steps to 36 (10 degrees) to speed up initialization
ALPHA_STEPS = 36 
R_MIN_TABLE = []
v1_base = hex_vertices(0.0)
for i in range(ALPHA_STEPS + 1):
    alpha = (i / ALPHA_STEPS) * (math.pi / 3)
    v2_rot = hex_vertices(-alpha)
    r_min = minkowski_sum_inradius(v1_base, v2_rot)
    R_MIN_TABLE.append(r_min)

def get_r_min(alpha):
    """Interpolate R_min for a given relative angle alpha."""
    alpha = alpha % (math.pi / 3)
    if alpha < 0:
        alpha += math.pi / 3
    idx = alpha / (math.pi / 3) * ALPHA_STEPS
    i0 = int(idx)
    i1 = min(i0 + 1, ALPHA_STEPS)
    frac = idx - i0
    return R_MIN_TABLE[i0] * (1 - frac) + R_MIN_TABLE[i1] * frac

def calculate_outer_side_length(positions, angles):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, 
    vertices at 0, 60, ... degrees) that contains all hexagons.
    """
    max_s = 0.0
    # Precompute constants
    sqrt3_half = math.sqrt(3) / 2.0
    
    for i in range(N):
        px = positions[i][0]
        py = positions[i][1]
        ang = angles[i]
        
        # Vertices of the inner hexagon
        # v_k = (px + cos(ang + k*pi/3), py + sin(ang + k*pi/3))
        for k in range(6):
            theta = ang + k * math.pi / 3
            vx = px + math.cos(theta)
            vy = py + math.sin(theta)
            
            r_sq = vx*vx + vy*vy
            if r_sq < 1e-9:
                continue
            
            # phi = atan2(vy, vx)
            # Instead of atan2, we can use the property that the distance to the apothem 
            # determines the required side length.
            # The apothem directions are at pi/6 + k*pi/3.
            # We need the angle of the vertex relative to the nearest apothem direction.
            
            phi = math.atan2(vy, vx)
            if phi < 0:
                phi += 2 * math.pi
            
            # Nearest apothem angle
            # delta = (phi - pi/6) % (pi/3)
            delta = (phi - math.pi / 6) % (math.pi / 3)
            if delta > math.pi / 6:
                delta = math.pi / 3 - delta
            
            # r = S * (sqrt(3)/2) / cos(delta)  => S = r * cos(delta) / (sqrt(3)/2)
            r = math.sqrt(r_sq)
            s_val = r * math.cos(delta) / sqrt3_half
            if s_val > max_s:
                max_s = s_val
    return max_s

def calculate_penalty(positions, angles):
    """
    Calculate penalty for overlapping inner hexagons.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist_sq = dx*dx + dy*dy
            # Early exit if clearly separated
            # Min possible R_min is sqrt(3) (approx 1.732) for parallel hexagons touching.
            # Max R_min is 2 (opposite orientations).
            # If dist_sq > 4, no overlap possible.
            if dist_sq > 4.0:
                continue
            
            dist = math.sqrt(dist_sq)
            rel_angle = angles[i] - angles[j]
            r_min = get_r_min(rel_angle)
            if dist < r_min:
                overlap = r_min - dist
                penalty += overlap * overlap
    return penalty

def objective(state):
    """
    state: list of size 3*N: [x1, y1, theta1, x2, y2, theta2, ...]
    """
    positions = []
    angles = []
    for i in range(N):
        x = state[3*i]
        y = state[3*i+1]
        th = state[3*i+2]
        positions.append((x, y))
        angles.append(th)
    
    s_outer = calculate_outer_side_length(positions, angles)
    penalty = calculate_penalty(positions, angles)
    
    return s_outer + 100.0 * penalty

def simulated_annealing(initial_state, time_limit=40.0):
    """
    Perform Simulated Annealing.
    """
    current_state = list(initial_state)
    current_obj = objective(current_state)
    best_state = current_state[:]
    best_obj = current_obj
    
    start_time = time.time()
    t_end = start_time + time_limit
    
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.999 # Slower cooling to explore more
    
    n_moves = 0
    while time.time() < t_end:
        # Generate neighbor
        neighbor = current_state[:]
        idx = random.randint(0, len(neighbor) - 1)
        if idx % 3 == 2: # Angle
            step = random.uniform(-0.1, 0.1)
        else: # Position
            step = random.uniform(-0.5, 0.5)
        neighbor[idx] += step
        
        neighbor_obj = objective(neighbor)
        
        delta = neighbor_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_obj = neighbor_obj
            
            if current_obj < best_obj:
                best_state = current_state[:]
                best_obj = current_obj
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        n_moves += 1
        # Check time every iteration for strict compliance
        if time.time() > t_end:
            break
                
    return best_state, best_obj

def get_initial_hex_lattice():
    positions = []
    angles = [0.0] * N
    positions.append([0.0, 0.0])
    for k in range(6):
        angle = k * math.pi / 3
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        positions.append([x, y])
    for k in range(4):
        angle = k * math.pi / 3
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        positions.append([x, y])
        
    state = []
    for p, a in zip(positions, angles):
        state.extend([p[0], p[1], a])
    return state

def get_initial_random():
    positions = []
    angles = []
    for _ in range(N):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        a = random.uniform(0, 2 * math.pi)
        positions.append([x, y])
        angles.append(a)
    state = []
    for p, a in zip(positions, angles):
        state.extend([p[0], p[1], a])
    return state

def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin for 120s limit
    
    best_state = None
    best_score = float('inf')
    
    # Run 1: Hex lattice
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        init1 = get_initial_hex_lattice()
        t1 = min(40.0, remaining - 5.0) # Leave 5s for next
        if t1 > 1.0:
            state1, score1 = simulated_annealing(init1, time_limit=t1)
            if score1 < best_score:
                best_score = score1
                best_state = state1

    # Run 2: Random
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        init2 = get_initial_random()
        t2 = min(40.0, remaining - 5.0)
        if t2 > 1.0:
            state2, score2 = simulated_annealing(init2, time_limit=t2)
            if score2 < best_score:
                best_score = score2
                best_state = state2

    # Run 3: Random (if time permits)
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        init3 = get_initial_random()
        t3 = max(1.0, remaining - 2.0)
        if t3 > 1.0:
            state3, score3 = simulated_annealing(init3, time_limit=t3)
            if score3 < best_score:
                best_score = score3
                best_state = state3

    if best_state is None:
        # Fallback
        best_state = get_initial_hex_lattice()
        best_score = objective(best_state)

    positions = []
    angles = []
    for i in range(N):
        x = best_state[3*i]
        y = best_state[3*i+1]
        a = best_state[3*i+2]
        positions.append([x, y, a * 180.0 / math.pi])
        angles.append(a)
        
    final_s = calculate_outer_side_length(
        [(p[0], p[1]) for p in positions], 
        angles
    )
    
    return positions, (0.0, 0.0), final_s, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
