import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 120.0  # Keep margin from 150s limit

def get_hex_vertices(center, angle_deg=0.0):
    """
    Returns the 6 vertices of a unit regular hexagon.
    Center is (x, y).
    Angle is rotation in degrees.
    Vertices are at distance 1 from center.
    """
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    # Vertices at angles theta + k * 60
    angles = theta + np.linspace(0, 300, 6, endpoint=False) * np.pi / 180.0
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def compute_min_side(centers, angles, outer_angle_deg=0.0):
    """
    Computes the minimum side length S of a regular hexagon centered at origin
    with rotation outer_angle_deg that contains all inner hexagons.
    """
    # Get all vertices of all inner hexagons
    all_vertices = []
    for c, a in zip(centers, angles):
        verts = get_hex_vertices(c, a)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices) # Shape (66, 2)
    
    # Normals of the outer hexagon
    theta = np.deg2rad(outer_angle_deg)
    # Normals are perpendicular to sides.
    # Sides are at angles theta + k*60.
    # Normals are at angles theta + k*60 + 90? 
    # A regular hexagon centered at origin with "flat" top has normals at 90, 150, 210, 270, 330, 30.
    # Let's define normals at angles: theta + np.linspace(0, 300, 6, endpoint=False) * np.pi/180 + np.pi/2
    normal_angles = theta + np.linspace(0, 300, 6, endpoint=False) * np.pi / 180.0 + np.pi / 2.0
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles))) # Shape (6, 2)
    
    # Project all vertices onto all normals
    # all_vertices: (66, 2)
    # normals: (6, 2)
    # projections: (66, 6)
    projections = all_vertices @ normals.T
    
    # Max absolute projection
    max_proj = np.max(np.abs(projections))
    
    # The apothem (distance from center to side) is max_proj.
    # Side length S = 2 * apothem / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def overlap_penalty(centers):
    """
    Penalty if any two inner hexagons overlap.
    Min distance between centers of non-overlapping unit hexagons is sqrt(3).
    """
    penalty = 0.0
    n = len(centers)
    centers = np.array(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < np.sqrt(3.0):
                # Quadratic penalty
                penalty += (np.sqrt(3.0) - d) ** 2
    return penalty

def objective(vars):
    """
    vars: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, outer_angle]
    """
    n = N
    centers = []
    angles = []
    for i in range(n):
        x = vars[3*i]
        y = vars[3*i + 1]
        a = vars[3*i + 2]
        centers.append((x, y))
        angles.append(a)
    outer_angle = vars[3*n]
    
    S = compute_min_side(centers, angles, outer_angle)
    pen = overlap_penalty(centers)
    
    # We want to minimize S, but ensure no overlap.
    # If there is overlap, S is not valid. We add a large penalty.
    # If pen > 0, the configuration is invalid.
    # To guide the optimizer, we return S + 1000 * pen.
    return S + 1000.0 * pen

def construct():
    start_time = time.time()
    
    # Initial configuration: 7 flower + 4 extensions
    # Centers in axial-like coordinates converted to Cartesian
    # (q, r) -> x = sqrt(3) * (q + r/2), y = 1.5 * r
    # Let's just hardcode the Cartesian coords for the 11 points
    
    sqrt3 = np.sqrt(3.0)
    init_centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (-sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5),
        (-sqrt3/2, -1.5),
        (2*sqrt3, 0.0),
        (-2*sqrt3, 0.0),
        (sqrt3, 3.0),
        (-sqrt3, -3.0)
    ]
    
    # Initial angles all 0
    init_angles = [0.0] * N
    init_outer_angle = 0.0
    
    # Flatten to vector
    x0 = []
    for c, a in zip(init_centers, init_angles):
        x0.append(c[0])
        x0.append(c[1])
        x0.append(a)
    x0.append(init_outer_angle)
    x0 = np.array(x0)
    
    best_S = compute_min_side(init_centers, init_angles, init_outer_angle)
    best_config = (init_centers, init_angles, init_outer_angle)
    best_penalty = overlap_penalty(init_centers)
    
    # Optimization
    # Use Powell method which doesn't require gradients
    # We will run a few iterations or until time limit
    
    # To avoid local minima, we can try a few random restarts if time permits
    # But first, let's run one optimization on the initial guess
    
    try:
        # Limit number of function evaluations to ensure speed
        # Powell can be slow, so we use maxiter
        res = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'maxfev': 5000, 'disp': False})
        
        if res.success or res.fun < objective(x0):
            best_vars = res.x
            # Unpack
            best_centers = []
            best_angles = []
            for i in range(N):
                best_centers.append((best_vars[3*i], best_vars[3*i+1]))
                best_angles.append(best_vars[3*i+2])
            best_outer_angle = best_vars[3*N]
            
            best_penalty = overlap_penalty(best_centers)
            if best_penalty < 1e-6: # Valid configuration
                best_S = compute_min_side(best_centers, best_angles, best_outer_angle)
                best_config = (best_centers, best_angles, best_outer_angle)
            else:
                # If penalty is high, the S might be artificially low or high due to penalty
                # We should check if the raw S of the best valid config is better
                pass
                
    except Exception:
        pass
    
    # If time remains, try a random restart with perturbed initial guess
    if time.time() - start_time < DEADLINE - 30:
        # Perturb initial centers
        np.random.seed(42)
        perturbed_centers = [(c[0] + np.random.normal(0, 0.5), c[1] + np.random.normal(0, 0.5)) for c in init_centers]
        x0_pert = []
        for c, a in zip(perturbed_centers, init_angles):
            x0_pert.append(c[0])
            x0_pert.append(c[1])
            x0_pert.append(a)
        x0_pert.append(init_outer_angle)
        x0_pert = np.array(x0_pert)
        
        try:
            res2 = minimize(objective, x0_pert, method='Powell', options={'maxiter': 500, 'maxfev': 5000, 'disp': False})
            if res2.success:
                best_vars = res2.x
                best_centers = []
                best_angles = []
                for i in range(N):
                    best_centers.append((best_vars[3*i], best_vars[3*i+1]))
                    best_angles.append(best_vars[3*i+2])
                best_outer_angle = best_vars[3*N]
                
                pen2 = overlap_penalty(best_centers)
                if pen2 < 1e-6:
                    S2 = compute_min_side(best_centers, best_angles, best_outer_angle)
                    if S2 < best_S:
                        best_S = S2
                        best_config = (best_centers, best_angles, best_outer_angle)
        except Exception:
            pass

    # Final check: ensure we return a valid configuration
    # If the optimized one was invalid (high penalty), we should fall back to initial if it was valid
    # But our initial guess has no overlaps.
    
    centers, angles, outer_angle = best_config
    
    # Format output
    inner = []
    for c, a in zip(centers, angles):
        inner.append([c[0], c[1], a])
    
    return inner, (0.0, 0.0), best_S, outer_angle

# EVOLVE-BLOCK-START
# The function construct is defined above.
# We just need to ensure it's the one returned.
# The task asks to keep the function names and signatures.
# The current program has:
# def construct():
#     ...
#     return inner, (0.0, 0.0), 40.0, 0.0
#
# My code defines construct() and returns the tuple.
# I will wrap it in the EVOLVE-BLOCK as requested by the format?
# "Keep the function names and signatures of the seed program."
# "Output the COMPLETE program in a single python code block."
# The seed program has the EVOLVE-BLOCK markers. I should probably keep the structure similar or just provide the code.
# The prompt says "Rewrite the program from scratch... Output the COMPLETE program".
# I will provide the full script.
# EVOLVE-BLOCK-END
