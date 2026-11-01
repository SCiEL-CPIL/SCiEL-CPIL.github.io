"""
Combined solution: Lattice-based initialization + Physics-based relaxation.
Uses Lennard-Jones-like repulsion for non-overlap and smooth penalty for containment.
Optimizes using BFGS for fast convergence on the smooth landscape.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin for 150s limit

# Unit hexagon geometry
# Vertices at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges (pointing to edge midpoints)
# Midpoints at 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_hex_vertices(centers):
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

def compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        # Project onto outward normals
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center=(0.0, 0.0), outer_side=None, outer_angle=0.0, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    if outer_side is None:
        outer_side = compute_required_side(centers, outer_center, outer_angle)
        
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True

def generate_lattice_configurations():
    """
    Generate diverse, compact initial configurations using hexagonal lattice patterns.
    """
    configs = []
    
    # Hexagonal lattice basis vectors for side-1 hexagons (center-to-center distance sqrt(3))
    # Actually, distance between centers of adjacent hexagons in a close pack is 2 * apothem = sqrt(3).
    # Let's use standard hex grid coords.
    # a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5)
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    def make_config(coords, angle=0.0):
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], angle])
        return centers

    # Pattern 1: 3-4-3-2 (Compact)
    p1 = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p1))
    
    # Pattern 2: 2-4-4-2 (Rectangular-ish)
    p2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p2))
    
    # Pattern 3: 1-3-4-3-1 (Diamond-ish)
    p3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    configs.append(make_config(p3))
    
    # Pattern 4: 4-4-4 (Three rows)
    p4 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    configs.append(make_config(p4))
    
    # Pattern 5: 2-3-4-3 (Asymmetric)
    p5 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(make_config(p5))
    
    # Pattern 6: 3-3-3-3 (Staggered rows)
    p6 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(make_config(p6))
    
    # Add some random perturbations to break symmetry
    for i in range(3):
        base = configs[i % len(configs)]
        np.random.seed(42 + i)
        perturbed = []
        for c in base:
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            da = np.random.uniform(-5, 5)
            perturbed.append([c[0] + dx, c[1] + dy, c[2] + da])
        configs.append(perturbed)
        
    return configs

def physics_objective(params, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Objective function for physics-based relaxation.
    Minimizes: Repulsion (non-overlap) + Containment Penalty (smooth).
    """
    centers = np.array(params).reshape(N, 3)
    
    # 1. Repulsion Term (Lennard-Jones like)
    # We want to push centers apart if they are too close.
    # Minimum distance between centers of non-overlapping side-1 hexagons is sqrt(3) ~ 1.732.
    # We use a soft barrier.
    repulsion = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i, :2] - centers[j, :2])
            # If d < threshold, apply strong repulsion
            threshold = 1.732 
            if d < threshold:
                # Inverse 4th power for strong repulsion
                repulsion += 1000.0 * ((threshold - d) / threshold) ** 4
            elif d < threshold * 1.5:
                # Mild repulsion to keep them separated
                repulsion += 10.0 * ((threshold * 1.5 - d) / threshold) ** 2
            
    # 2. Containment Term
    # We want to keep all vertices inside a hexagon of side S.
    # To minimize S, we treat S as a variable? No, we minimize the max extent.
    # Smooth approximation of max extent:
    # Let M = max extent. We want to minimize M.
    # Instead, we can penalize vertices that go beyond a "current best" boundary.
    # But a simpler approach: minimize the sum of squared distances of vertices from the center,
    # constrained by the hexagonal shape.
    # Actually, the most effective physics model is to minimize the "energy" of the system
    # where the boundary is a soft wall.
    
    # Let's use a different approach:
    # Minimize the sum of squared distances of all vertices from the origin,
    # but weighted by the hexagonal norm.
    # This tends to shrink the cluster towards the center.
    
    verts = get_hex_vertices(centers)
    
    # Compute hexagonal distance for each vertex
    # d_hex(v) = max_k |v . n_k|
    # We want to minimize the maximum d_hex.
    # Smooth max: log-sum-exp or power-sum.
    # Let's use power-sum with high power p=8.
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    v_rel = verts - np.array(outer_center)
    v_rot = np.einsum('ij,njk->nik', v_rot, R_o) # Rotate all vertices
    
    # Project onto normals
    # projs shape: (N, 6, 6) -> N hexagons, 6 vertices, 6 normals
    projs = np.einsum('nki,ki->nk', v_rot, OUTER_NORMALS)
    abs_projs = np.abs(projs)
    
    # For each vertex, get the max projection (hex distance)
    # d_hex shape: (N, 6)
    d_hex = np.max(abs_projs, axis=2)
    
    # We want to minimize the global maximum.
    # Use a smooth approximation: (sum(d^p))^(1/p)
    p = 8
    smooth_max = (np.sum(d_hex ** p) / (N * 6)) ** (1.0 / p)
    
    # The side length is proportional to this smooth max.
    # side = smooth_max / (sqrt(3)/2)
    containment_cost = smooth_max ** 2 # Minimize squared side length (faster convergence)
    
    total_cost = repulsion + containment_cost
    
    return total_cost

def relax_configuration(initial_centers, time_limit=10.0):
    """
    Perform physics-based relaxation using BFGS.
    """
    start_time = time.time()
    
    # Flatten parameters
    x0 = np.array(initial_centers).flatten()
    
    # Define objective wrapper with time check
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return physics_objective(x)
    
    # BFGS is good for smooth problems
    # We don't provide gradients, so BFGS will approximate them.
    # This can be slow, but the landscape is smooth.
    
    try:
        res = minimize(
            obj, x0, method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-5, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
    except Exception:
        new_centers = np.array(initial_centers)
        
    return new_centers

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_lattice_configurations()
    
    # Shuffle to ensure diversity in time usage
    np.random.shuffle(configs)
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
            
        # Phase 1: Physics Relaxation
        # Give each config a reasonable time budget
        # We have ~140s total. ~10 configs. ~14s each.
        # But BFGS might be slow. Let's cap at 10s per config.
        relax_time = min(10.0, DEADLINE - elapsed - 5)
        
        if relax_time < 2.0:
            continue
            
        try:
            relaxed = relax_configuration(config, time_limit=relax_time)
        except Exception:
            relaxed = np.array(config)
            
        # Phase 2: Compute actual score and validate
        # We need to check if the relaxed configuration is valid.
        # The physics objective minimized a smooth approximation, so it might still overlap slightly
        # or violate the boundary slightly due to the soft nature.
        
        # Let's try to "snap" or refine if it's close to valid.
        # For now, just check validity.
        
        # Compute required side
        req_side = compute_required_side(relaxed)
        
        # Check validity
        if check_validity(relaxed, outer_side=req_side):
            if req_side < best_score:
                best_score = req_side
                best_result = (relaxed.tolist(), (0.0, 0.0), req_side, 0.0)
        else:
            # If invalid, try a quick local repair?
            # Or just skip. The lattice starts are usually valid, and relaxation
            # with strong repulsion should keep them valid.
            # If it becomes invalid, it's likely due to numerical issues or weak repulsion.
            # Let's try to use the original config if it's better?
            orig_side = compute_required_side(config)
            if check_validity(config, outer_side=orig_side):
                 if orig_side < best_score:
                     best_score = orig_side
                     best_result = (config, (0.0, 0.0), orig_side, 0.0)
                     
    if best_result is not None:
        return best_result
    
    # Fallback: Simple circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = compute_required_side(inner_default)
    return inner_default, (0.0, 0.0), required_side, 0.0

# Ensure the function signature matches the requirement
# construct() -> (inner, center, side, angle)
# inner: list of [x, y, angle_degrees]
# center: [x, y]
# side: float
# angle: float (outer rotation)

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
