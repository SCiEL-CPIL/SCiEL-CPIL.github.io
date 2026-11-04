import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    
    # 1. Vectorized distance pre-check
    # If distance < 2 * inradius, they MUST overlap (necessary condition for non-overlap is dist >= 2*inradius? No, that's for circles. 
    # For hexagons, the minimum distance between centers for non-overlap is 2 * inradius only if aligned. 
    # Actually, the "kissing number" geometry implies if dist < 2*inradius, they overlap.
    # 2 * sqrt(3)/2 = sqrt(3) ~ 1.732.
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

    # 2. Exact SAT check for pairs that pass distance check
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            # Re-check distance to be safe, though we already did it
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            if dx*dx + dy*dy < min_dist_sq:
                return False
                
            vj = verts_list[j]
            aj = axes_list[j]
            
            overlap = True
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    overlap = False
                    break
            
            if overlap:
                for axis in aj:
                    minA, maxA = _project_polygon(vi, axis)
                    minB, maxB = _project_polygon(vj, axis)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        overlap = False
                        break
            
            if overlap:
                return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _quick_score(cx_arr, cy_arr, angle_arr):
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    max_need = 0.0
    for i in range(N):
        a = math.radians(angle_arr[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_honeycomb_initial():
        # Standard hex grid packing
        # Center at 0,0
        pts = [[0.0, 0.0, 0.0]]
        r1 = 2 * INRADIUS # Distance to adjacent hex centers in same row? 
        # In a hex grid of side 1 hexes:
        # Horizontal spacing: 2 * inradius = sqrt(3)
        # Vertical spacing: 1.5 (apothem * 2 / 2? No. Height is sqrt(3). Row shift is sqrt(3)/2)
        # Let's use exact coordinates for a 1-2-3-2-1 or similar cluster.
        # A 12 hex cluster:
        # Row 0: 1 hex at (0,0)
        # Row 1: 2 hexes at (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
        # Row 2: 3 hexes at (-sqrt(3), 0), (0, 0), (sqrt(3), 0) -- wait, (0,0) is occupied.
        
        # Let's generate a valid set manually or algorithmically.
        # Algorithmic: Start with center, add neighbors.
        # Hex grid coordinates (axial):
        # (0,0)
        # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,1 is taken), (0,1 is taken)
        
        # Axial to Cartesian:
        # x = size * 3/2 * q
        # y = size * sqrt(3) * (r + q/2)
        # size = 1 (side length)
        
        axial_coords = []
        # Center
        axial_coords.append((0,0))
        # Ring 1 (6 hexes)
        for i in range(6):
            q = math.cos(math.radians(60*i))
            r = math.sin(math.radians(60*i))
            # This is not integer axial. 
            # Directions: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
            dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
            axial_coords.append(dirs[i])
            
        # Ring 2 (need 5 more to make 12)
        # (2,0), (2,-1), (2,-2), (1,-2), (0,-2)
        extra = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2)]
        axial_coords.extend(extra)
        
        pts = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = math.sqrt(3) * (r + q/2.0)
            pts.append([x, y, 0.0])
            
        return pts

    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_honeycomb_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    
    # 1. Initialize
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _quick_score(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
    
    if best_cfg is None:
        # Fallback: scaled honeycomb
        cfg = make_honeycomb_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)

    # 2. Differential Evolution (Global Search)
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    pop_size = 20
    population = []
    population.append([cx_arr.copy(), cy_arr.copy(), angle_arr.copy(), best_side])
    
    for _ in range(pop_size - 1):
        p = best_cfg[:]
        for i in range(N):
            p[i][0] += random.gauss(0, 0.5)
            p[i][1] += random.gauss(0, 0.5)
            p[i][2] += random.gauss(0, 10.0)
        
        pcx = np.array([q[0] for q in p])
        pcy = np.array([q[1] for q in p])
        pang = np.array([q[2] for q in p])
        
        if _is_valid_fast(pcx, pcy, pang):
            score = _quick_score(pcx, pcy, pang)
            population.append([pcx, pcy, pang, score])
        else:
            population.append([pcx, pcy, pang, float('inf')])
    
    F = 0.8
    CR = 0.9
    de_generations = 150
    
    for gen in range(de_generations):
        if time.time() - t0 > TIME_LIMIT - 40:
            break
        
        for i in range(pop_size):
            if time.time() - t0 > TIME_LIMIT - 40:
                break
            
            a, b, c = random.sample(range(pop_size), 3)
            
            trial_cx = population[a][0].copy()
            trial_cy = population[a][1].copy()
            trial_ang = population[a][2].copy()
            
            for j in range(N):
                trial_cx[j] = population[a][0][j] + F * (population[b][0][j] - population[c][0][j])
                trial_cy[j] = population[a][1][j] + F * (population[b][1][j] - population[c][1][j])
                trial_ang[j] = population[a][2][j] + F * (population[b][2][j] - population[c][2][j])
            
            j_rand = random.randrange(N)
            # Use temporary arrays for crossover to avoid modifying population directly until accepted
            cand_cx = population[i][0].copy()
            cand_cy = population[i][1].copy()
            cand_ang = population[i][2].copy()
            
            for j in range(N):
                if random.random() < CR or j == j_rand:
                    cand_cx[j] = trial_cx[j]
                    cand_cy[j] = trial_cy[j]
                    cand_ang[j] = trial_ang[j]
            
            if _is_valid_fast(cand_cx, cand_cy, cand_ang):
                new_score = _quick_score(cand_cx, cand_cy, cand_ang)
                if new_score < population[i][3]:
                    population[i][0] = cand_cx
                    population[i][1] = cand_cy
                    population[i][2] = cand_ang
                    population[i][3] = new_score
                    
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cand_cx[k], cand_cy[k], cand_ang[k]] for k in range(N)]

    # 3. Local Refinement with Nelder-Mead
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    x0 = np.concatenate([cx_arr, cy_arr, angle_arr])
    
    def objective(x):
        cx = x[:N]
        cy = x[N:2*N]
        ang = x[2*N:3*N]
        
        # Penalty for invalidity to guide optimizer away
        if not _is_valid_fast(cx, cy, ang):
            return 100.0 + _quick_score(cx, cy, ang)
        
        return _quick_score(cx, cy, ang)
    
    # Run Nelder-Mead
    if time.time() - t0 < TIME_LIMIT - 20:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.success or res.fun < best_side:
            x_opt = res.x
            cx_opt = x_opt[:N]
            cy_opt = x_opt[N:2*N]
            ang_opt = x_opt[2*N:3*N]
            
            if _is_valid_fast(cx_opt, cy_opt, ang_opt):
                new_side = _quick_score(cx_opt, cy_opt, ang_opt)
                if new_side < best_side:
                    best_side = new_side
                    best_cfg = [[cx_opt[i], cy_opt[i], ang_opt[i]] for i in range(N)]

    # 4. Optimize Outer Hexagon
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = (0.0, 0.0)
    best_final_angle = 0.0
    
    # Precompute vertices for best config
    verts_best = [_hex_vertices(cx_best[i], cy_best[i], angle_best[i]) for i in range(N)]
    
    # Sweep outer angle
    for outer_angle in range(0, 30, 1):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        # Optimize center (cx, cy)
        # We want to minimize max_proj = max( dot(n, v - c) )
        # This is equivalent to maximizing min( dot(n, c) - dot(n, v) )? No.
        # Let f(c) = max_i,k dot(n_i, v_k - c)
        # We want to find c that minimizes this.
        # This is a convex optimization problem (min of max of linear functions).
        # We can use a simple gradient step or grid search around centroid.
        
        centroid = (np.mean(cx_best), np.mean(cy_best))
        cx_c, cy_c = centroid
        
        # Simple local search for center
        for _iter in range(20):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            # Try small shifts
            step = 0.05
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_best[i]:
                            for (nx, ny) in normals_out:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, cy_c
        
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback to valid initial
        cfg = make_honeycomb_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_final_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_final_center, best_final_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
