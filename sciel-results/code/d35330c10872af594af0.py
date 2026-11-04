import numpy as np
import time
import math

N = 12

# Helper functions

def get_hex_vertices(x, y, theta):
    """Returns 6 vertices of a unit hexagon centered at (x,y) rotated by theta."""
    # Vertices of unit hexagon at origin, rot 0:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    verts = []
    for k in range(6):
        angle = k * math.pi / 3
        vx = math.cos(angle)
        vy = math.sin(angle)
        # Rotate
        rx = vx * cos_t - vy * sin_t
        ry = vx * sin_t + vy * cos_t
        verts.append([x + rx, y + ry])
    return np.array(verts)

def calc_containment_penalty(verts_all, L):
    """
    verts_all: array of shape (N, 6, 2)
    L: outer side length
    Returns sum of violations for all vertices.
    Outer hexagon normals: 0, 60, ... 300 deg.
    Apothem = L * sqrt(3) / 2
    """
    apothem = L * math.sqrt(3) / 2.0
    # Normals
    normals = []
    for k in range(6):
        angle = k * math.pi / 3
        normals.append([math.cos(angle), math.sin(angle)])
    normals = np.array(normals) # (6, 2)
    
    # Project all vertices onto all normals
    # verts_all: (N, 6, 2)
    # normals: (6, 2)
    # We want max projection for each vertex.
    # proj[i, j, k] = dot(vert[i, j], normal[k])
    
    # Reshape for broadcasting
    # v: (N*6, 2)
    v = verts_all.reshape(-1, 2)
    # proj: (N*6, 6)
    proj = v @ normals.T
    # max_proj: (N*6,)
    max_proj = np.max(proj, axis=1)
    # violations: (N*6,)
    violations = np.maximum(0, max_proj - apothem)
    return np.sum(violations**2)

def calc_overlap_penalty(verts_all):
    """
    verts_all: array of shape (N, 6, 2)
    Returns sum of squared overlap depths for all pairs.
    """
    penalty = 0.0
    N = len(verts_all)
    for i in range(N):
        for j in range(i+1, N):
            v1 = verts_all[i]
            v2 = verts_all[j]
            # Calculate min distance / overlap depth
            # Using SAT-like approach for convex polygons
            # Axes from v1 and v2 edges
            
            # Get edges vectors
            e1 = np.roll(v1, -1, axis=0) - v1
            e2 = np.roll(v2, -1, axis=0) - v2
            
            # Normals to edges
            # Rotate 90 deg
            n1 = np.column_stack([-e1[:, 1], e1[:, 0]])
            n2 = np.column_stack([-e2[:, 1], e2[:, 0]])
            
            # Normalize normals
            n1_norms = np.linalg.norm(n1, axis=1, keepdims=True)
            n2_norms = np.linalg.norm(n2, axis=1, keepdims=True)
            n1 = n1 / n1_norms
            n2 = n2 / n2_norms
            
            axes = np.vstack([n1, n2])
            
            # Project v1 and v2 onto all axes
            p1 = v1 @ axes.T
            p2 = v2 @ axes.T
            
            min1 = np.min(p1, axis=1)
            max1 = np.max(p1, axis=1)
            min2 = np.min(p2, axis=1)
            max2 = np.max(p2, axis=1)
            
            # Overlap on each axis
            # If max1 < min2 or max2 < min1, they are separated.
            # Separation distance = min2 - max1 or max2 - min1
            # Overlap depth = min(max1, max2) - max(min1, min2)
            
            # We want the MINIMUM separation distance (which is negative if overlapping)
            # sep_k = min(max1, max2) - max(min1, min2)
            sep = np.minimum(max1, max2) - np.maximum(min1, min2)
            
            # If they are disjoint, there exists an axis with sep < 0.
            # The distance between them is max(-sep) over axes where sep < 0?
            # Actually, the minimum translation distance is min(sep) if we define sep as overlap?
            # Let's define overlap_k = min(max1, max2) - max(min1, min2).
            # If overlap_k < 0, they are separated on this axis by -overlap_k.
            # If overlap_k > 0, they overlap on this axis.
            # If they overlap in 2D, overlap_k > 0 for all k.
            # If they are disjoint, overlap_k < 0 for at least one k.
            
            # We want a penalty that is 0 if disjoint, and positive if overlapping.
            # If disjoint, min_overlap = min(overlap_k) will be negative.
            # We can use penalty = max(0, min_overlap)^2 ?
            # If they are disjoint, min_overlap < 0, penalty = 0.
            # If they overlap, min_overlap > 0 (smallest overlap depth), penalty = min_overlap^2.
            # This is a good smooth penalty.
            
            min_overlap = np.min(sep)
            if min_overlap > 0:
                penalty += min_overlap**2
    return penalty

def calc_total_penalty(config, L):
    """
    config: list of (x, y, theta)
    L: outer side
    """
    verts = []
    for x, y, theta in config:
        verts.append(get_hex_vertices(x, y, theta))
    verts = np.array(verts)
    
    pen_cont = calc_containment_penalty(verts, L)
    pen_ovl = calc_overlap_penalty(verts)
    
    # Weights: Overlap is critical. Containment is also critical.
    # If we are binary searching L, we want to minimize the "infeasibility".
    # If we can make both 0, it's feasible.
    return pen_cont + pen_ovl

def solve_packing(L, time_limit=5.0):
    """
    Runs SA to find a valid packing for side L.
    Returns best config and penalty.
    """
    start_time = time.time()
    
    # Initial config: Random or Grid
    # Let's try a hexagonal grid initialization
    # Center at 0,0
    # 12 hexes.
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 5 (to make 12)
    
    init_config = []
    # Center
    init_config.append([0.0, 0.0, 0.0])
    # Ring 1
    for k in range(6):
        angle = k * math.pi / 3
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        init_config.append([x, y, 0.0])
    # Ring 2 (partial)
    # Add 5 more
    for k in range(5):
        angle = k * math.pi / 3 + math.pi/6
        # Distance for ring 2 center
        # Hex grid spacing:
        # r=1: sqrt(3)
        # r=2: 2*sqrt(3)? No.
        # Let's just place them roughly
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        init_config.append([x, y, 0.0])
    
    # Add random jitter
    np.random.seed(42) # Reproducible initial, but SA will move
    for i in range(len(init_config)):
        init_config[i][0] += np.random.uniform(-0.5, 0.5)
        init_config[i][1] += np.random.uniform(-0.5, 0.5)
        init_config[i][2] += np.random.uniform(-0.1, 0.1)
        
    best_config = [c[:] for c in init_config]
    best_penalty = calc_total_penalty(best_config, L)
    
    current_config = [c[:] for c in init_config]
    current_penalty = best_penalty
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.99
    step_size = 0.5
    
    while time.time() - start_time < time_limit:
        # Perturb
        new_config = [c[:] for c in current_config]
        for i in range(N):
            new_config[i][0] += np.random.uniform(-step_size, step_size)
            new_config[i][1] += np.random.uniform(-step_size, step_size)
            new_config[i][2] += np.random.uniform(-step_size, step_size)
            
        new_penalty = calc_total_penalty(new_config, L)
        
        # Acceptance
        if new_penalty < current_penalty:
            current_config = new_config
            current_penalty = new_penalty
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_config = [c[:] for c in current_config]
        else:
            delta = new_penalty - current_penalty
            if np.random.random() < math.exp(-delta / T):
                current_config = new_config
                current_penalty = new_penalty
                
        T *= alpha
        if T < T_min:
            break
            
    return best_config, best_penalty

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Binary search on L
    L_min = 3.5
    L_max = 6.0
    
    best_L = L_max
    best_config = None
    
    # We want to find the smallest L.
    # We will do a few iterations of binary search.
    # Each check takes ~5s.
    # We can do ~20 checks in 100s.
    
    num_checks = 0
    while time.time() < deadline - 10.0:
        L_mid = (L_min + L_max) / 2.0
        
        config, penalty = solve_packing(L_mid, time_limit=4.0)
        num_checks += 1
        
        # If penalty is very small, we consider it feasible
        if penalty < 1e-3:
            best_L = L_mid
            best_config = config
            L_max = L_mid
        else:
            L_min = L_mid
            
        if L_max - L_min < 1e-3:
            break
            
    # Final polish?
    # If we have a best_config, we can try to refine it with tighter SA
    # But for now, return best_L
    
    # Ensure we return a valid format
    # inner: (x, y, angle_degrees)
    # angle in construct is in degrees?
    # "angle_degrees"
    
    if best_config is None:
        # Fallback to seed-like behavior if nothing found
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    inner = []
    for x, y, theta in best_config:
        angle_deg = math.degrees(theta)
        inner.append([x, y, angle_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0
