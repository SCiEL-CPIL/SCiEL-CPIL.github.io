"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import scipy.optimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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
        
        # If dist > 2.0, definitely disjoint (circumradius of unit hexagon is 1)
        if dist_sq > 4.0:
            return True
        
        # Precise SAT check
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        all_edges = []
        for i in range(6):
            all_edges.append(v1[i] - v1[(i+1) % 6])
            all_edges.append(v2[i] - v2[(i+1) % 6])
        
        for edge in all_edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            if np.max(proj1) < np.min(proj2) - 1e-12 or np.max(proj2) < np.min(proj1) - 1e-12:
                return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Compute minimum outer side length required to contain all inner hexagons."""
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon (pointing outward)
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / SQRT3

    def initial_guess():
        """Create a valid initial configuration using hexagonal packing."""
        params = np.zeros(3*N + 3)
        positions = []
        
        # Center hexagon
        positions.append((0.0, 0.0, 0.0))
        
        # Ring 1: 6 hexagons at distance 2 (touching center)
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2.0*np.cos(theta), 2.0*np.sin(theta), 0.0))
        
        # Ring 2: Place 5 more hexagons in the second ring
        # Second ring positions in hexagonal lattice are at axial coords (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-2), (2,-1), (1,0), (0,-1)
        # Convert to Cartesian: for axial (q, r), Cartesian = (3/2*q, sqrt(3)/2*(q + 2*r))
        # Let's pick 5 well-separated positions from the second ring
        # Using angles at 30, 90, 150, 210, 270 degrees at appropriate radius
        # The second ring "corner" positions are at distance 4 from center
        # The second ring "edge midpoint" positions are at distance 2*sqrt(3) ≈ 3.464
        
        # Place at the 5 "edge" positions of the second ring (not corners) to maximize spacing
        # Actually, let's place them at the 6 corner positions minus 1 to avoid overlap with ring 1
        # Corners of ring 2 are at angles 0, 60, 120, 180, 240, 300 at radius 4
        # But these might overlap with ring 1 hexagons. Let me check.
        # Ring 1 hexagon at angle 0 is at (2, 0). Ring 2 corner at angle 0 is at (4, 0).
        # Distance = 2, so they touch. That's fine for disjoint (touching is allowed? No, they must be disjoint).
        # Actually "disjoint" means no interior intersection. Touching at boundary might be OK or not.
        # Let's assume we need strict separation. So let's place ring 2 at slightly larger radius or different angles.
        
        # Better approach: place the 5 remaining hexagons at the "gap" positions between ring 1 hexagons
        # These are at angles 30, 90, 150, 210, 270, 330 at radius 2*sqrt(3) ≈ 3.464
        # Pick 5 of these 6 positions
        gap_angles = [30, 90, 150, 210, 270, 330]
        gap_radius = 2.0 * SQRT3
        for i in range(5):
            theta = np.radians(gap_angles[i])
            positions.append((gap_radius*np.cos(theta), gap_radius*np.sin(theta), 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def objective(params):
        """Objective: minimize outer side length while ensuring disjointness."""
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        
        side = compute_min_side_fast(inner, center, angle)
        return side

    best_params = initial_guess()
    best_score = objective(best_params)
    
    current_params = best_params.copy()
    current_score = best_score
    
    # Simulated annealing
    T = 2.0
    T_min = 0.001
    cooling_rate = 0.9997
    step_size = 0.3
    max_steps = 30000
    
    step_count = 0
    while time.time() < deadline - 25.0 and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        # Perturb 1-2 random parameters
        n_perturb = random.randint(1, 2)
        for _ in range(n_perturb):
            idx = random.randint(0, 3*N + 2)
            current_step = step_size * (0.5 + 0.5 * T)
            new_params[idx] += random.gauss(0, current_step)
        
        new_score = objective(new_params)
        
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / max(T, T_min)):
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Polish with Nelder-Mead using smooth objective
    def smooth_penalty(params):
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                if d < 2.0:
                    penalty += (2.0 - d) ** 2
        return penalty

    def polish_objective(params):
        inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        side = compute_min_side_fast(inner, center, angle)
        pen = smooth_penalty(params)
        return side + 20.0 * pen

    if time.time() < deadline - 10.0:
        try:
            result = scipy.optimize.minimize(
                polish_objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            p = result.x
            inner = [(p[3*i], p[3*i+1], p[3*i+2]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not hexagons_disjoint_fast(inner[i], inner[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                new_side = compute_min_side_fast(inner, (p[3*N], p[3*N+1]), p[3*N+2])
                if new_side < best_score:
                    best_score = new_side
                    best_params = p
        except Exception:
            pass

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
