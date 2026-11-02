"""Packing 12 unit hexagons in a regular hexagon - Improved SA with fast collision + analytical outer hex."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        vertices = np.empty((6, 2))
        for i in range(6):
            theta = angle + i * np.pi / 3
            vertices[i, 0] = cx + side * np.cos(theta)
            vertices[i, 1] = cy + side * np.sin(theta)
        return vertices

    def hexagons_disjoint_fast(h1, h2):
        """Check if two unit hexagons are disjoint using SAT with early exit."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        # If centers far enough apart, definitely disjoint
        if dist_sq > 4.0:
            return True
        
        # Compute vertices
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # SAT test: check all 12 edge normals (6 from each hex)
        for i in range(6):
            # Edge from v1[i] to v1[(i+1)%6]
            ex1 = v1[(i+1) % 6, 0] - v1[i, 0]
            ey1 = v1[(i+1) % 6, 1] - v1[i, 1]
            nx = -ey1
            ny = ex1
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-10:
                nx /= norm
                ny /= norm
                proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
                proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
                proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
                proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
                if proj1_max < proj2_min or proj2_max < proj1_min:
                    return True
        
        for i in range(6):
            # Edge from v2[i] to v2[(i+1)%6]
            ex2 = v2[(i+1) % 6, 0] - v2[i, 0]
            ey2 = v2[(i+1) % 6, 1] - v2[i, 1]
            nx = -ey2
            ny = ex2
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-10:
                nx /= norm
                ny /= norm
                proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
                proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
                proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
                proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
                if proj1_max < proj2_min or proj2_max < proj1_min:
                    return True
        
        return False

    def compute_min_side_analytical(inner, center, angle):
        """Compute minimum side length of enclosing regular hexagon using support function."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        # Check 6 directions (normals of outer hexagon sides)
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            extents = offsets[:, 0] * nx + offsets[:, 1] * ny
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def objective(params):
        """Hard objective: returns side length if valid, else large penalty."""
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+3]
        
        # Check all pairs for overlap
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        
        side = compute_min_side_analytical(inner, center, angle)
        return side

    def smooth_objective(params, beta=30.0):
        """Smooth surrogate for SA: soft containment + separation penalty."""
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+3]
        
        sqrt3 = np.sqrt(3)
        
        # Soft-max containment using all vertices
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        max_extent = 0.0
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            extents = offsets[:, 0] * nx + offsets[:, 1] * ny
            # Soft-max for smoothness
            max_e = (1.0/beta) * np.log(np.sum(np.exp(beta * extents)))
            if max_e > max_extent:
                max_extent = max_e
        
        side = max_extent * 2.0 / sqrt3
        
        # Separation penalty: penalize pairs that are too close
        sep_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist_sq = dx*dx + dy*dy
                # Minimum distance for non-overlapping unit hexagons is ~1.732 (sqrt(3))
                # Use a softer threshold to allow SA to navigate
                threshold = 2.5
                if dist_sq < threshold:
                    sep_penalty += (threshold - dist_sq) ** 2
        
        return side + 5.0 * sep_penalty

    def initial_guess_honeycomb():
        """Create initial configuration with honeycomb-like arrangement."""
        params = np.zeros(3*N + 4)
        
        # Place hexagons in a compact cluster
        # Center hexagon
        positions = [(0, 0, 0)]
        
        # 6 hexagons around center at distance 2 (touching)
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        
        # 5 more hexagons in outer ring
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+3] = 0.0
        return params

    def initial_guess_random():
        """Create random initial configuration."""
        params = np.zeros(3*N + 4)
        for i in range(N):
            params[3*i] = random.uniform(-3, 3)
            params[3*i+1] = random.uniform(-3, 3)
            params[3*i+2] = random.uniform(0, 60)
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+3] = 0.0
        return params

    # Try multiple initializations
    best_params = None
    best_score = float('inf')
    
    for init_type in ['honeycomb', 'random1', 'random2']:
        if init_type == 'honeycomb':
            params = initial_guess_honeycomb()
        else:
            params = initial_guess_random()
        
        score = objective(params)
        if score < best_score:
            best_score = score
            best_params = params.copy()
    
    # Simulated Annealing with smooth objective
    current_params = best_params.copy()
    current_score = smooth_objective(current_params, 30.0)
    
    T = 2.0
    T_min = 0.0005
    cooling_rate = 0.9995
    max_steps = 100000
    
    step_count = 0
    last_best_step = 0
    patience = 5000  # Steps without improvement before early stop
    
    while time.time() < deadline - 15.0 and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        # Choose which parameter to perturb
        idx = random.randint(0, 3*N + 3)
        if idx == 3*N + 2:  # Skip center y (symmetry)
            idx = random.randint(0, 3*N + 1)
        
        # Adaptive step size
        current_step = 0.3 * (0.3 + 0.7 * T / T_min)
        new_params[idx] += random.gauss(0, current_step)
        
        new_score = smooth_objective(new_params, 30.0)
        
        # Accept if better or with probability based on temperature
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / max(T, 1e-10)):
            current_params = new_params
            current_score = new_score
            
            # Periodically check hard validity
            if step_count % 100 == 0 or current_score < best_score - 0.01:
                hard_score = objective(current_params)
                if hard_score < best_score:
                    best_score = hard_score
                    best_params = current_params.copy()
                    last_best_step = step_count
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # Early stop if no improvement for too long
        if step_count - last_best_step > patience and T < 0.01:
            break

    # L-BFGS-B polish phase with multiple attempts
    if time.time() < deadline - 5.0 and best_score < 1e5:
        for polish_attempt in range(3):
            if time.time() >= deadline - 5.0:
                break
            
            start_params = best_params.copy()
            # Add small random perturbation for diversity
            if polish_attempt > 0:
                start_params += np.random.normal(0, 0.05, start_params.shape)
            
            try:
                result = minimize(
                    smooth_objective,
                    start_params,
                    method='L-BFGS-B',
                    args=(30.0,),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < 1e5:
                    polished = result.x
                    polished_score = objective(polished)
                    if polished_score < best_score:
                        best_score = polished_score
                        best_params = polished
            except Exception:
                pass

    # Final local search: try small perturbations around best
    if time.time() < deadline - 3.0 and best_score < 1e5:
        for _ in range(50):
            if time.time() >= deadline - 3.0:
                break
            perturbed = best_params.copy()
            idx = random.randint(0, 3*N + 3)
            perturbed[idx] += random.gauss(0, 0.01)
            score = objective(perturbed)
            if score < best_score:
                best_score = score
                best_params = perturbed

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+3])
    side = compute_min_side_analytical(inner, center, angle)
    
    # Final validation
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
