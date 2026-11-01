import numpy as np
import time
import random

N = 12
SQRT3 = np.sqrt(3.0)

# Precompute vertex offsets for a unit hexagon at angle 0
# Vertices at angles 0, 60, 120, 180, 240, 300
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

# Precompute rotation matrices for a grid of angles (e.g., 0 to 359 in steps of 5)
# We use a coarse grid for SA moves, but final output can be refined.
# Actually, for SA, we can just compute rotation on the fly for the specific angle.
# To speed up, we precompute for 360 possible degrees.
ROT_MATRICES = np.zeros((360, 6, 2))
for deg in range(360):
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    # Rotate base vertices
    # BASE_VERTS is (6, 2). R @ v
    ROT_MATRICES[deg] = (R @ BASE_VERTS.T).T

def hexagon_vertices(cx, cy, angle_deg):
    """
    Returns (6, 2) array of vertices for a unit hexagon at (cx, cy) with rotation angle_deg.
    """
    # Use precomputed if possible, else compute
    deg_idx = int(angle_deg) % 360
    # For simplicity and precision, we compute directly. 
    # Precomputation helps if we do many evals, but N=12 is small.
    # Let's use the precomputed for speed if angle is integer, else compute.
    if angle_deg == deg_idx:
        verts = ROT_MATRICES[deg_idx]
    else:
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        verts = (R @ BASE_VERTS.T).T
    
    return verts + np.array([cx, cy])

def check_validity(positions, angles, tol=1e-5):
    """
    Check if all hexagons are pairwise disjoint using SAT.
    positions: (N, 2)
    angles: (N,)
    """
    vertices_list = []
    for i in range(N):
        x, y = positions[i]
        verts = hexagon_vertices(x, y, angles[i])
        vertices_list.append(verts)
    
    for i in range(N):
        for j in range(i + 1, N):
            p1 = vertices_list[i]
            p2 = vertices_list[j]
            
            # SAT check
            disjoint = False
            # Check edges of p1
            for i_sat in range(6):
                j_sat = (i_sat + 1) % 6
                edge = p1[j_sat] - p1[i_sat]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal /= norm
                proj1 = p1.dot(normal)
                proj2 = p2.dot(normal)
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    disjoint = True
                    break
            
            if not disjoint:
                # Check edges of p2
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p2[j_sat] - p2[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                if not disjoint:
                    return False
    return True

def required_outer_side(positions, angles):
    """
    Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
    that contains all unit hexagons at given positions and angles.
    """
    verts_flat = np.zeros((N * 6, 2))
    for i in range(N):
        verts_flat[i*6:(i+1)*6] = hexagon_vertices(positions[i, 0], positions[i, 1], angles[i])
    
    x = verts_flat[:, 0]
    y = verts_flat[:, 1]
    
    # The 6 linear forms for the hexagon sides (normalizing so that the value is the distance from center to side)
    # For a regular hexagon of side s centered at origin with flat-top orientation (vertex at 0, s):
    # Wait, standard "flat-top" has vertices at 0, 60...
    # The distance from center to side (apothem) is s * sqrt(3) / 2.
    # The linear forms L_k are projections onto the normals.
    # Normals are at angles 30, 90, 150, 210, 270, 330.
    # Normal 1: (cos30, sin30) = (sqrt3/2, 1/2). Projection: x*sqrt3/2 + y*1/2.
    # Max projection P. Side length s = P / (sqrt3/2) = 2P/sqrt3.
    
    # Let's use the forms from the seed program which seem to work:
    # L1 = x + y / sqrt3  -> Normal (1, 1/sqrt3). Norm = sqrt(1 + 1/3) = 2/sqrt3.
    # If max L is M, distance is M / (2/sqrt3) = M * sqrt3 / 2.
    # Side length s = distance / (sqrt3/2) = M.
    # So the side length is simply the max of these 6 linear forms.
    
    L1 = x + y / SQRT3
    L2 = y * 2.0 / SQRT3
    L3 = -x + y / SQRT3
    L4 = -x - y / SQRT3
    L5 = -y * 2.0 / SQRT3
    L6 = x - y / SQRT3
    
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    return np.max(all_L)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial configuration: Hexagonal lattice
    # This is a known good starting point.
    # Centers of hexagons in a lattice with spacing 2 (touching)
    # But we need to fit them in a smaller hexagon.
    # Let's start with a compact cluster.
    
    # Hexagonal lattice coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # For N=12, a 4x3 grid or similar?
    # Let's use a centered hexagon of radius 1 (7 hexes) + 5 more?
    # Or just a 3x4 grid.
    # Let's try a circular arrangement as a second init.
    
    def get_initial_config():
        # Init 1: Lattice
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        pos = []
        ang = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pos.append([x, y])
            ang.append(0.0)
        return np.array(pos), np.array(ang)

    positions, angles = get_initial_config()
    
    # Ensure initial validity
    if not check_validity(positions, angles):
        # Scale up until valid
        scale = 1.0
        while scale < 5.0:
            pos_scaled = positions * scale
            if check_validity(pos_scaled, angles):
                positions = pos_scaled
                break
            scale += 0.1
    
    current_side = required_outer_side(positions, angles)
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = current_side
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    # We want to finish before deadline
    # Each step: 1 move generation + validity check (O(N^2)) + objective (O(N))
    # N=12, pairs=66. Validity is fast.
    # Let's estimate steps.
    
    steps = 0
    max_steps = 50000 # Cap steps
    
    while time.time() < deadline and steps < max_steps and T > T_min:
        # Generate a move
        i = random.randint(0, N - 1)
        move_type = random.random()
        
        if move_type < 0.7:
            # Translate
            delta = np.random.normal(0, 0.5, 2)
            new_positions = positions.copy()
            new_positions[i] += delta
            new_angles = angles.copy()
        else:
            # Rotate
            delta_angle = random.uniform(-15, 15)
            new_positions = positions.copy()
            new_angles = angles.copy()
            new_angles[i] += delta_angle
            # Keep angles in 0-360 for precomp usage if needed, though not strictly required
            new_angles[i] %= 360
        
        # Check validity
        if not check_validity(new_positions, new_angles):
            # Reject invalid moves
            T *= cooling_rate
            steps += 1
            continue
        
        new_side = required_outer_side(new_positions, new_angles)
        delta_obj = new_side - current_side
        
        # Acceptance probability
        if delta_obj < 0 or random.random() < np.exp(-delta_obj / T):
            # Accept
            positions = new_positions
            angles = new_angles
            current_side = new_side
            
            if current_side < best_side:
                best_side = current_side
                best_positions = positions.copy()
                best_angles = angles.copy()
        
        T *= cooling_rate
        steps += 1
        
        # Adaptive cooling: if we've been stuck, cool faster
        # Or just let it run.
        
    # Final polish: Use a local gradient descent or simple coordinate descent to refine
    # Since we have a valid configuration, we can try to minimize the side length further
    # using a smooth approximation of the objective, ignoring the validity constraint
    # but checking it afterwards.
    
    # Let's do a few steps of gradient descent on the active constraints
    # This is risky if it breaks validity, but we can revert.
    
    # Simple local search: try small perturbations around the best solution
    # to see if we can shave off more.
    
    for _ in range(1000):
        if time.time() > deadline - 5.0:
            break
        i = random.randint(0, N - 1)
        # Small step
        if random.random() < 0.5:
            delta = np.random.normal(0, 0.01, 2)
            new_positions = best_positions.copy()
            new_positions[i] += delta
            new_angles = best_angles.copy()
        else:
            delta_angle = random.uniform(-1, 1)
            new_positions = best_positions.copy()
            new_angles = best_angles.copy()
            new_angles[i] += delta_angle
            new_angles[i] %= 360
        
        if not check_validity(new_positions, new_angles):
            continue
        
        new_side = required_outer_side(new_positions, new_angles)
        if new_side < best_side:
            best_side = new_side
            best_positions = new_positions
            best_angles = new_angles

    # Return best found
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
