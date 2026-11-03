import math
import time
import random
import numpy as np
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = 115.0  # Safety margin before 120s wall-clock kill

# Precompute unit hexagon vertices and edge normals
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.0:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # SAT Check
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        try:
            lo = max(0.0, best_phi - 2.0)
            hi = min(60.0, best_phi + 2.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-9})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    def objective_function(params):
        """
        Objective for DE: Minimizes outer side length.
        If configuration is invalid (overlaps), returns a large penalty.
        """
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:3*N]
        
        # Quick check for obvious overlaps to penalize early
        # Use a simplified distance check first? No, SAT is fast enough for N=12 if we early exit.
        angles_rad = np.radians(ang)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Penalty: return a value larger than any valid side length
                    # Max valid side is roughly 5-6. Let's use 100 + overlap_metric?
                    # Just returning 100 is sufficient for ranking, but gradient info is lost.
                    # For DE, ranking is key.
                    return 100.0
        
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        return side

    # --- Initialization ---
    # Use a "Snowball" packing to generate a good initial population seed
    # Or just random grid. Let's use a mix.
    
    # Helper to create a random valid-ish config
    def generate_random_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Place center
        pos[0] = [0, 0]
        ang[0] = 0
        # Place others randomly in a ring, then push apart?
        # Simple random within a bounding box
        for i in range(1, N):
            pos[i] = [random.uniform(-4, 4), random.uniform(-4, 4)]
            ang[i] = random.uniform(0, 360)
        return pos, ang

    # Population Size
    POP_SIZE = 50
    # DE Parameters
    F = 0.8  # Scaling factor
    CR = 0.9 # Crossover rate
    
    # Initialize Population
    population = []
    
    # Seed 0: The radial cluster from the original program (known good)
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    base_positions[0] = [0, 0]
    for i in range(6):
        angle = i * math.pi / 3
        r = math.sqrt(3)
        base_positions[i+1] = [r * math.cos(angle), r * math.sin(angle)]
    gap_angles = [(i + 0.5) * math.pi / 3 for i in range(5)]
    r2 = 2 * math.sqrt(3)
    for i in range(5):
        base_positions[i+7] = [r2 * math.cos(gap_angles[i]), r2 * math.sin(gap_angles[i])]
    
    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = base_positions[i, 0]
        init_params[3*i+1] = base_positions[i, 1]
        init_params[3*i+2] = base_angles[i]
    population.append(init_params)
    
    # Seed 1-49: Random configurations
    for _ in range(POP_SIZE - 1):
        pos, ang = generate_random_config()
        p = np.zeros(3*N)
        for i in range(N):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        population.append(p)
        
    # Evaluate initial population
    fitness = []
    for p in population:
        fitness.append(objective_function(p))
        
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    # --- Differential Evolution Loop ---
    max_generations = 200
    time_budget = DEADLINE - 10.0 # Leave time for final polish
    
    gen = 0
    while time.time() - t0 < time_budget and gen < max_generations:
        gen += 1
        new_population = []
        
        for i in range(POP_SIZE):
            # Select 3 distinct random parents
            a, b, c = random.sample(range(POP_SIZE), 3)
            while a == i: a = random.randint(0, POP_SIZE-1)
            while b == i: b = random.randint(0, POP_SIZE-1)
            while c == i: c = random.randint(0, POP_SIZE-1)
            
            # Mutation
            mutant = population[a] + F * (population[b] - population[c])
            
            # Crossover
            trial = population[i].copy()
            j_rand = random.randint(0, 3*N - 1)
            for j in range(3*N):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Evaluation
            trial_fit = objective_function(trial)
            
            # Selection
            if trial_fit <= fitness[i]:
                new_population.append(trial)
                # Update fitness immediately for this slot? 
                # Standard DE replaces at the end, but we can update best here.
                if trial_fit < best_fitness:
                    best_fitness = trial_fit
                    best_params = trial.copy()
            else:
                new_population.append(population[i])
                
        population = new_population
        # Update fitness array to match new population?
        # For efficiency, we only strictly need the best. But DE needs population diversity.
        # We should re-evaluate or keep track. 
        # Since objective_function is expensive, let's just keep the best and restart if stuck?
        # Actually, standard DE keeps the population. Let's just track the best found so far.
        
    # --- Polish Phase ---
    # Use Nelder-Mead on the best DE result
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 5.0:
        def polish_objective(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
                
            angles_rad = np.radians(ang)
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                    h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        return 1e6
            
            side, phi, cx, cy = compute_min_outer_side(pos, ang)
            return side

        try:
            res = minimize(polish_objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_fitness:
                best_params = res.x
                best_fitness = res.fun
        except Exception:
            pass

    # Extract final result
    final_pos = np.zeros((N, 2))
    final_ang = np.zeros(N)
    for i in range(N):
        final_pos[i, 0] = best_params[3*i]
        final_pos[i, 1] = best_params[3*i+1]
        final_ang[i] = best_params[3*i+2]
        
    if not is_valid_configuration(final_pos, final_ang):
        # Fallback to base
        final_pos = base_positions.copy()
        final_ang = base_angles.copy()
        
    final_side, final_phi, final_cx, final_cy = compute_min_outer_side(final_pos, final_ang)
    
    inner = []
    for i in range(N):
        inner.append([float(final_pos[i, 0]), float(final_pos[i, 1]), float(final_ang[i])])
        
    center = (float(final_cx), float(final_cy))
    side = float(final_side)
    angle = float(final_phi)
    
    return inner, center, side, angle
