"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Helper to check if a point is inside a hexagon of side L centered at origin (flat-top)
    def point_in_hexagon(x, y, L):
        # Flat top hexagon inequalities:
        # |x| <= L
        # |y| <= sqrt(3)/2 * L
        # |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * L
        if abs(x) > L: return False
        if abs(y) > np.sqrt(3)/2 * L: return False
        if abs(np.sqrt(3)/2 * x + 0.5 * y) > np.sqrt(3)/2 * L: return False
        return True

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint (max distance between any two points on unit hexagons)
        if dist_sq > 4.0:
            return True
        
        # Precise SAT check for all other cases
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # Check all 12 edge normals (6 from each hexagon)
        for i in range(6):
            # Edge normal from h1
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            nn1 = np.linalg.norm(n1)
            if nn1 > 1e-12:
                n1 = n1 / nn1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
            
            # Edge normal from h2
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            nn2 = np.linalg.norm(n2)
            if nn2 > 1e-12:
                n2 = n2 / nn2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def objective(params):
        """Objective: minimize outer side length while ensuring disjointness with smooth penalty."""
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness with soft penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    penalty += 10.0
        
        if penalty > 0:
            return 1e4 + penalty
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        return side

    def initial_guess():
        params = np.zeros(3*N + 3)
        positions = []
        # Hexagonal packing for 12 hexagons: 1 center, 6 ring 1, 5 ring 2
        positions.append((0, 0, 0))
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    current_params = best_params.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    step_size = 0.3
    max_steps = 100000 # Hard limit on iterations to prevent infinite loop if time check fails
    
    step_count = 0
    while time.time() < deadline and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        # Perturb one or two random parameters for better exploration
        n_perturb = 1 if random.random() < 0.7 else 2
        for _ in range(n_perturb):
            idx = random.randint(0, 3*N + 2)
            current_step = step_size * (0.5 + 0.5 * T)
            new_params[idx] += random.gauss(0, current_step)
        
        new_score = objective(new_params)
        
        # Accept
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Polish pass: refine with smaller steps
    if time.time() < deadline - 5.0:
        polish_start = time.time()
        polish_steps = 5000
        for _ in range(polish_steps):
            if time.time() - polish_start > 10.0:
                break
            new_params = best_params.copy()
            idx = random.randint(0, 3*N + 2)
            new_params[idx] += random.gauss(0, 0.05)
            new_score = objective(new_params)
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final check: ensure validity (disjointness is already checked in objective, 
    # but let's be safe with a quick final validation)
    # If invalid, return a large valid box
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback: use a known valid configuration (hexagonal grid)
        inner = []
        positions = [(0, 0, 0)]
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        for pos in positions:
            inner.append([float(pos[0]), float(pos[1]), float(pos[2])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
