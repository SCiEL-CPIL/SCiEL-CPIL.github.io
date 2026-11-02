"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import scipy.optimize

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
        
        all_edges = []
        for i in range(6):
            all_edges.append(v1[i] - v1[(i+1) % 6])
            all_edges.append(v2[i] - v2[(i+1) % 6])
        
        for edge in all_edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal = normal / norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
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
        """Objective: minimize outer side length while ensuring disjointness."""
        # Unpack
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
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        return side

    def initial_guess():
        params = np.zeros(3*N + 3)
        positions = []
        # Hexagonal packing for 12 hexagons: 1 center, 6 ring 1, 5 ring 2
        # Ring 1: 6 hexagons at distance 2 from center
        # Ring 2: 5 hexagons? 
        # A perfect hexagonal number packing for 12 is not standard.
        # 1 + 6 = 7. 
        # Next ring has 12. We only need 5 more.
        # Let's place them in the second ring positions.
        positions.append((0, 0, 0))
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        
        # Second ring positions are at distance 4 (for axial coordinates (2,0) etc)
        # In Cartesian, the "corners" of the second ring are at distance 4.
        # The "edges" of the second ring are at distance 2*sqrt(3) ~ 3.464.
        # Let's just scatter them roughly in the second ring area.
        # Better: Use a known good layout for 12.
        # 1 center.
        # 6 around it.
        # 5 in the gaps of the second ring?
        # Let's try placing them at angles 30, 90, 150, 210, 270 at radius 3.5?
        # Or just randomize the last 5 in the annulus.
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            r = 3.8 # Slightly less than 4 to fit
            positions.append((r*np.cos(theta), r*np.sin(theta), 0))
        
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
    step_size = 0.5
    max_steps = 20000 # Reduced to leave time for polish
    
    step_count = 0
    while time.time() < deadline - 20.0 and step_count < max_steps: # Reserve 20s for polish
        step_count += 1
        new_params = current_params.copy()
        
        # Perturb one random parameter
        idx = random.randint(0, 3*N + 2)
        
        # Adaptive step size
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

    # Polish with Nelder-Mead
    # We need a smooth objective for Nelder-Mead to work well.
    # The current objective is hard (1e6 penalty).
    # Let's create a smooth penalty version for the polish.
    def smooth_penalty(params):
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    # Calculate a rough overlap measure
                    # Distance between centers
                    d = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                    # If d < 2, they might overlap.
                    # Penalty increases as d decreases.
                    if d < 2.0:
                        penalty += (2.0 - d) ** 2
        return penalty

    def polish_objective(params):
        side = compute_min_side_fast(
            [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)],
            (params[3*N], params[3*N+1]),
            params[3*N+2]
        )
        pen = smooth_penalty(params)
        return side + 10.0 * pen

    if time.time() < deadline - 5.0:
        try:
            # Nelder-Mead
            result = scipy.optimize.minimize(
                polish_objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            # Check if the polished result is valid and better
            if result.fun < best_score:
                # Verify validity with strict check
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
                    best_score = compute_min_side_fast(
                        inner, 
                        (p[3*N], p[3*N+1]), 
                        p[3*N+2]
                    )
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
    
    # Final check: ensure validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to initial guess if polish broke it (should not happen if we checked)
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
