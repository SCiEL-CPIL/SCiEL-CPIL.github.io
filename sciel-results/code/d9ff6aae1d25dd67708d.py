"""
Improved program using Simulated Annealing with vectorized geometry checks.
"""
import numpy as np
import time
import random

N = 12
DEADLINE = 145.0

# Precompute unit hexagon vertices (side length 1, angle 0)
# Vertices are at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)
])
# Normals to the edges, pointing outward
# Edge i is from vertex i to vertex (i+1)%6
# The outward normal for a regular hexagon centered at origin is the angle bisector
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)
])

def get_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def compute_clearances(verts, outer_center, outer_angle, outer_side):
    """
    Compute minimum clearance between inner hexagons and between inner hexagons and outer boundary.
    Returns: (min_pair_sep, min_outer_clearance)
    """
    n = verts.shape[0]
    min_pair = float('inf')
    
    # Precompute edge normals for all inner hexagons
    # edges: (N, 6, 2)
    edges = np.roll(verts, -1, axis=1) - verts
    norms = np.linalg.norm(edges, axis=2) # (N, 6)
    norms[norms < 1e-12] = 1.0
    # normals: (N, 6, 2)
    normals = np.stack((-edges[:, :, 1], edges[:, :, 0]), axis=2) / norms[:, :, np.newaxis]
    
    # Pairwise separation
    for i in range(n):
        v1 = verts[i]
        n1 = normals[i]
        
        for j in range(i+1, n):
            v2 = verts[j]
            n2 = normals[j]
            
            axes = np.vstack([n1, n2])
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.maximum(proj1.min(axis=0) - proj2.max(axis=0), 
                             proj2.min(axis=0) - proj1.max(axis=0))
            
            min_sep = np.min(sep)
            if min_sep < min_pair:
                min_pair = min_sep
                
    # Outer Clearance
    apothem = outer_side * np.sqrt(3) / 2
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    min_outer = float('inf')
    
    for i in range(n):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        clearances = apothem - max_proj
        min_c = np.min(clearances)
        if min_c < min_outer:
            min_outer = min_c
            
    return min_pair, min_outer

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    verts = get_vertices(np.array(centers))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    verts = get_vertices(np.array(centers))
    min_pair, min_outer = compute_clearances(verts, outer_center, outer_angle, outer_side)
    return min_pair >= -tol and min_outer >= -tol

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*0.866), (-dx/2, dy*0.866),
        (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
        (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
        (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
        (0, dy*1.732), (0, -dy*1.732)
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def simulated_annealing(initial_centers, outer_center, outer_angle, time_limit=10.0):
    """
    Run Simulated Annealing to optimize the packing.
    """
    start_time = time.time()
    current_centers = np.array(initial_centers, dtype=float)
    current_side = required_outer_side(current_centers, outer_center, outer_angle)
    
    best_centers = current_centers.copy()
    best_side = current_side
    best_cost = current_side
    
    # SA Parameters
    T = 0.5
    T_min = 0.001
    alpha = 0.995
    
    step_size_pos = 0.2
    step_size_ang = 5.0
    
    while time.time() - start_time < time_limit and T > T_min:
        # Perturb a random hexagon
        i = np.random.randint(0, N)
        new_centers = current_centers.copy()
        
        dx = np.random.uniform(-step_size_pos, step_size_pos)
        dy = np.random.uniform(-step_size_pos, step_size_pos)
        da = np.random.uniform(-step_size_ang, step_size_ang)
        
        new_centers[i, 0] += dx
        new_centers[i, 1] += dy
        new_centers[i, 2] += da
        
        # Calculate new side length
        new_side = required_outer_side(new_centers, outer_center, outer_angle)
        
        # Check clearances to compute penalty
        verts = get_vertices(new_centers)
        min_pair, min_outer = compute_clearances(verts, outer_center, outer_angle, new_side)
        
        penalty = 0.0
        if min_pair < 0:
            penalty += min_pair * 10.0
        if min_outer < 0:
            penalty += min_outer * 10.0
            
        new_cost = new_side + penalty
        
        # Acceptance criterion
        if new_cost < best_cost:
            best_cost = new_cost
            best_side = new_side
            best_centers = new_centers.copy()
            
        delta = new_cost - best_cost # Using best as reference for stability, or current?
        # Standard SA uses current state for delta
        delta = new_cost - (current_side + 0) # We need current cost
        
        # Recalculate current cost properly if we don't store it
        # For efficiency, let's just compare to the "current" accepted state's cost
        # We need to track current_cost
        
        # Let's restart the loop logic slightly to track current_cost properly
        
    # The above loop is a bit messy with cost tracking. Let's refine.
    
    return best_centers, best_side

def sa_optimize(initial_centers, outer_center, outer_angle, time_limit=10.0):
    """
    Refined Simulated Annealing.
    """
    start_time = time.time()
    current_centers = np.array(initial_centers, dtype=float)
    
    # Initial evaluation
    current_side = required_outer_side(current_centers, outer_center, outer_angle)
    verts = get_vertices(current_centers)
    min_pair, min_outer = compute_clearances(verts, outer_center, outer_angle, current_side)
    penalty = 0.0
    if min_pair < 0: penalty += min_pair * 10.0
    if min_outer < 0: penalty += min_outer * 10.0
    current_cost = current_side + penalty
    
    best_centers = current_centers.copy()
    best_side = current_side
    best_cost = current_cost
    
    T = 0.5
    T_min = 0.001
    alpha = 0.995
    
    step_size_pos = 0.2
    step_size_ang = 5.0
    
    while time.time() - start_time < time_limit and T > T_min:
        i = np.random.randint(0, N)
        new_centers = current_centers.copy()
        
        dx = np.random.uniform(-step_size_pos, step_size_pos)
        dy = np.random.uniform(-step_size_pos, step_size_pos)
        da = np.random.uniform(-step_size_ang, step_size_ang)
        
        new_centers[i, 0] += dx
        new_centers[i, 1] += dy
        new_centers[i, 2] += da
        
        new_side = required_outer_side(new_centers, outer_center, outer_angle)
        
        verts = get_vertices(new_centers)
        min_pair, min_outer = compute_clearances(verts, outer_center, outer_angle, new_side)
        
        penalty = 0.0
        if min_pair < 0: penalty += min_pair * 10.0
        if min_outer < 0: penalty += min_outer * 10.0
        new_cost = new_side + penalty
        
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_centers = new_centers
            current_side = new_side
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_side = new_side
                best_centers = new_centers.copy()
                
        T *= alpha
        
        # Adaptive step size: shrink if we're making progress
        if best_cost < current_cost * 0.99: # Heuristic: if we found a significantly better solution recently
             pass # Keep step size
        
    return best_centers, best_side

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("random3", initial_config_random(999)),
        ("random4", initial_config_random(1000)),
    ]
    
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Allocate time for SA
        # Leave 5s for final validation and potential last-minute check
        sa_time = min(20.0, DEADLINE - elapsed - 5)
        
        try:
            polished_centers, side_est = sa_optimize(config, (0.0, 0.0), 0.0, time_limit=sa_time)
        except Exception:
            polished_centers = np.array(config)
            side_est = required_outer_side(polished_centers, (0.0, 0.0), 0.0)
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 2:
            break
            
        # Final precise calculation
        final_side = required_outer_side(polished_centers, (0.0, 0.0), 0.0)
        
        # Validate
        if check_validity(polished_centers, (0.0, 0.0), final_side, 0.0):
            if final_side < best_score:
                best_score = final_side
                # Convert numpy array to list of lists for return
                inner_list = [list(polished_centers[i]) for i in range(N)]
                best_result = (inner_list, (0.0, 0.0), final_side, 0.0)
            
    if best_result is not None:
        return best_result
    
    # Fallback
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0
