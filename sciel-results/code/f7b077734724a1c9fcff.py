import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 150.0

def _hex_vertices(cx, cy, s, rot):
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    ox, oy = outer_c
    max_proj = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / math.sqrt(3.0)

def _get_penetration(A, B):
    """
    Calculate the minimum translation distance to separate two convex polygons A and B.
    Returns 0 if they are disjoint or touching.
    Uses Separating Axis Theorem (SAT).
    """
    min_overlap = float('inf')
    
    # Collect all edge normals from both polygons
    edges = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            # Normal vector
            nx, ny = -ay, ax
            # Normalize
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9:
                continue
            nx /= norm
            ny /= norm
            edges.append((nx, ny))
            
    # Check projection on each axis
    for nx, ny in edges:
        minA = min(p[0]*nx + p[1]*ny for p in A)
        maxA = max(p[0]*nx + p[1]*ny for p in A)
        minB = min(p[0]*nx + p[1]*ny for p in B)
        maxB = max(p[0]*nx + p[1]*ny for p in B)
        
        # If separated, return 0
        if maxA < minB or maxB < minA:
            return 0.0
            
        # Calculate overlap on this axis
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap < min_overlap:
            min_overlap = overlap
            
    if min_overlap == float('inf'):
        # Should not happen for convex polygons
        return 0.0
        
    return min_overlap

def _valid(inner, outer_c, s, rot):
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    
    # Check containment
    for i in range(len(polys)):
        for p in polys[i]:
            inside = False
            # Simple point in polygon check using cross products
            n = len(outer_pts)
            sign = 0
            for j in range(n):
                c = (outer_pts[(j+1)%n][0]-outer_pts[j][0])*(p[1]-outer_pts[j][1]) - (outer_pts[(j+1)%n][1]-outer_pts[j][1])*(p[0]-outer_pts[j][0])
                if abs(c) < 1e-9:
                    continue
                sgn = 1 if c > 0 else -1
                if sign == 0:
                    sign = sgn
                elif sign != sgn:
                    break
            else:
                inside = True
            if not inside:
                return False

    # Check overlaps
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _get_penetration(polys[i], polys[j]) > 1e-7:
                return False
    return True

def _hexagonal_seed():
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _make_seed(rng):
    inner = [[rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 10.0, 0.0

def _objective(params, rot, oc):
    """
    Objective function for optimization.
    params: array of length 3*N, containing [x1, y1, a1, ..., x11, y11, a11]
    """
    pts = params.reshape(N, 3)
    
    # 1. Calculate the outer side length required to contain all inner hexagons
    outer_s = _min_outer_side(pts, oc, rot)
    
    # 2. Calculate overlap penalty
    overlap_penalty = 0.0
    polys = [_inner_hex(pts[i,0], pts[i,1], pts[i,2]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            pen = _get_penetration(polys[i], polys[j])
            if pen > 0:
                overlap_penalty += pen * pen
    
    # 3. Total cost
    penalty_weight = 100.0
    total_cost = outer_s + penalty_weight * overlap_penalty
    
    return total_cost

def _try_refine(inner, outer_c, s, rot, rng, iters, budget):
    best = (inner, outer_c, s, rot)
    cur = (inner, outer_c, s, rot)
    
    s_tight = _min_outer_side(cur[0], cur[1], cur[3])
    if s_tight < cur[2]:
        cur = (cur[0], cur[1], s_tight, cur[3])
        if _valid(cur[0], cur[1], cur[2], cur[3]):
            best = cur
    
    it = 0
    while it < iters:
        if time.time() > budget:
            break
        it += 1
        cand = [[x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), a + rng.uniform(-15, 15)] for (x, y, a) in cur[0]]
        cand_oc = (cur[1][0] + rng.uniform(-0.2, 0.2), cur[1][1] + rng.uniform(-0.2, 0.2))
        cand_rot = cur[3] + rng.uniform(-10, 10)
        
        cand_s = _min_outer_side(cand, cand_oc, cand_rot)
        
        if _valid(cand, cand_oc, cand_s, cand_rot):
            cur = (cand, cand_oc, cand_s, cand_rot)
            if cand_s < best[2]:
                best = cur
    
    return best

def _scipy_optimize(init_params, rot, oc, budget, maxiter=500):
    try:
        res = minimize(
            _objective,
            init_params,
            args=(rot, oc),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7}
        )
        if res.fun < float('inf') and time.time() < budget:
            return res.x
    except Exception:
        pass
    return None

def _pso_optimize(init_params, rot, oc, budget, particles=50, iterations=200):
    """
    Particle Swarm Optimization for the packing problem.
    """
    dim = 3 * N
    # Initialize particles
    # p_pos: positions, p_vel: velocities
    # We need to handle angle wrapping (0-60)
    
    def decode_params(p):
        # p is a flat array of length 3N
        # Angles are indices 2, 5, 8, ...
        decoded = p.copy()
        for i in range(N):
            idx = 3 * i + 2
            # Wrap angle to [0, 60)
            decoded[idx] = decoded[idx] % 60.0
        return decoded

    # Initialize
    p_pos = np.zeros((particles, dim))
    p_vel = np.zeros((particles, dim))
    
    # Seed one particle with the best known solution
    p_pos[0] = init_params
    # Seed others with random perturbations
    for i in range(1, particles):
        p_pos[i] = init_params + rng.normal(0, 0.5, dim)
    
    # Calculate initial fitness
    p_fit = np.zeros(particles)
    for i in range(particles):
        params = decode_params(p_pos[i])
        p_fit[i] = _objective(params, rot, oc)
    
    # Personal bests
    p_best_pos = p_pos.copy()
    p_best_fit = p_fit.copy()
    
    # Global best
    g_best_idx = np.argmin(p_best_fit)
    g_best_pos = p_best_pos[g_best_idx].copy()
    g_best_fit = p_best_fit[g_best_idx]
    
    # PSO parameters
    w = 0.7  # Inertia weight
    c1 = 1.5 # Cognitive coefficient
    c2 = 1.5 # Social coefficient
    
    for it in range(iterations):
        if time.time() > budget:
            break
        
        for i in range(particles):
            # Update velocity
            r1 = rng.random()
            r2 = rng.random()
            
            # Velocity update
            # v = w*v + c1*r1*(pbest - p) + c2*r2*(gbest - p)
            # Note: We need to be careful with angle dimensions.
            # Since we wrap angles in decode, the velocity update for angles might be tricky.
            # For simplicity, we treat angles as continuous in the update, but wrap in evaluation.
            # However, the gradient of the objective function w.r.t angle is periodic.
            # Nelder-Mead handles this better, but PSO is global.
            
            p_pos[i] += p_vel[i] * 0.0 # placeholder, we update below
            
            # Calculate new position
            # We update p_vel first
            p_vel[i] = w * p_vel[i] + c1 * r1 * (p_best_pos[i] - p_pos[i]) + c2 * r2 * (g_best_pos - p_pos[i])
            
            # Limit velocity
            max_vel = 1.0
            p_vel[i] = np.clip(p_vel[i], -max_vel, max_vel)
            
            # Update position
            p_pos[i] += p_vel[i]
            
            # Evaluate
            params = decode_params(p_pos[i])
            fit = _objective(params, rot, oc)
            
            # Update personal best
            if fit < p_best_fit[i]:
                p_best_fit[i] = fit
                p_best_pos[i] = p_pos[i].copy()
                
                # Update global best
                if fit < g_best_fit:
                    g_best_fit = fit
                    g_best_pos = p_pos[i].copy()
        
        # Linearly decrease w
        w = 0.9 - 0.4 * (it / iterations)
        
        if it % 10 == 0 and time.time() > budget:
            break
            
    return g_best_pos, g_best_fit

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0
    rng = random.Random(42)
    np.random.seed(42)
    
    # Try multiple structured seeds
    seeds = [_hexagonal_seed()]
    
    # Add rotated versions of the hexagonal seed
    base_seed = _hexagonal_seed()
    for rot_offset in [15, 30, 45]:
        inner_rot = [[x, y, a + rot_offset] for (x, y, a) in base_seed[0]]
        seeds.append((inner_rot, base_seed[1], base_seed[2], base_seed[3] + rot_offset))
    
    # Add random seeds
    for _ in range(5):
        seeds.append(_make_seed(rng))
    
    for seed in seeds:
        if time.time() > budget - 15:
            break
        inner, oc, s, rot = seed
        s_tight = _min_outer_side(inner, oc, rot)
        
        # Local refinement
        res = _try_refine(inner, oc, s_tight, rot, rng, iters=50, budget=budget - 10)
        inner_ls, oc_ls, s_ls, rot_ls = res
        if _valid(inner_ls, oc_ls, s_ls, rot_ls):
            if s_ls < best_score:
                best_score = s_ls
                best = (inner_ls, oc_ls, s_ls, rot_ls)
        
        # Scipy optimization from the refined starting point
        if time.time() < budget - 20:
            init_params = np.array([p for (x, y, a) in inner_ls for p in [x, y, a]])
            result = _scipy_optimize(init_params, rot_ls, oc_ls, budget - 15, maxiter=300)
            if result is not None:
                pts = result.reshape(N, 3)
                inner_sc = [[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i in range(N)]
                s_sc = _min_outer_side(inner_sc, oc_ls, rot_ls)
                if _valid(inner_sc, oc_ls, s_sc, rot_ls):
                    if s_sc < best_score:
                        best_score = s_sc
                        best = (inner_sc, oc_ls, s_sc, rot_ls)
        
        # PSO optimization
        if time.time() < budget - 25:
            init_params = np.array([p for (x, y, a) in inner_ls for p in [x, y, a]])
            pso_result, pso_fit = _pso_optimize(init_params, rot_ls, oc_ls, budget - 20, particles=30, iterations=100)
            if pso_result is not None:
                pts = pso_result.reshape(N, 3)
                # Ensure angles are wrapped
                for i in range(N):
                    pts[i, 2] = pts[i, 2] % 60.0
                inner_pso = [[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i in range(N)]
                s_pso = _min_outer_side(inner_pso, oc_ls, rot_ls)
                if _valid(inner_pso, oc_ls, s_pso, rot_ls):
                    if s_pso < best_score:
                        best_score = s_pso
                        best = (inner_pso, oc_ls, s_pso, rot_ls)
    
    # Final strict validation
    if best is not None:
        inner, oc, s, rot = best
        s_tight = _min_outer_side(inner, oc, rot)
        if _valid(inner, oc, s_tight, rot):
            return inner, (oc[0], oc[1]), s_tight, rot
        else:
            # Result is invalid, fall back to valid seed
            pass

    # Fallback to a known valid hexagonal lattice packing
    inner, oc, s, rot = _hexagonal_seed()
    s_tight = _min_outer_side(inner, oc, rot)
    return inner, (oc[0], oc[1]), s_tight, rot
