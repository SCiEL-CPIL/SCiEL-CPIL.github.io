import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute unit hexagon normals (outward pointing) for angle 0
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    unit_normals_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_hex_normals(angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return unit_normals_0 @ rot.T

    def calc_outer_side(params, outer_center=(0.0, 0.0), outer_angle=0.0):
        """
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Returns min side length of outer hexagon containing all inner hexagons.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        oc = np.array(outer_center)
        
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang_rad[i]
            verts = get_hex_verts(cx, cy, 1.0, np.degrees(a_rad))
            verts_rel = verts - oc
            projs = verts_rel @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        side = max_proj * 2.0 / SQRT3
        return side

    def sat_overlap_depth(poly1, poly2):
        """
        Returns the minimum penetration depth if poly1 and poly2 overlap.
        Returns 0.0 if they are disjoint or just touching.
        Uses Separating Axis Theorem.
        """
        min_overlap = 0.0
        axes = []
        
        # Axes from poly1 edges
        for k in range(6):
            edge = poly1[(k+1)%6] - poly1[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n/norm)
        
        # Axes from poly2 edges
        for k in range(6):
            edge = poly2[(k+1)%6] - poly2[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n/norm)
                
        for axis in axes:
            proj1 = np.dot(poly1, axis)
            proj2 = np.dot(poly2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Overlap amount: min(max1, max2) - max(min1, min2)
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0 # Separated on this axis
            if overlap > min_overlap:
                min_overlap = overlap
                
        return min_overlap

    def calc_penalty(params):
        """
        Returns sum of squared overlap depths for all pairs.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        polys = []
        for i in range(N):
            cx, cy = pos[i]
            verts = get_hex_verts(cx, cy, 1.0, np.degrees(ang_rad[i]))
            polys.append(verts)
            
        penalty = 0.0
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]
                
                # Quick bounding box check
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue # No overlap possible
                
                depth = sat_overlap_depth(p1, p2)
                if depth > 0:
                    penalty += depth * depth * 10.0
                    
        return penalty

    def objective(params):
        side = calc_outer_side(params)
        penalty = calc_penalty(params)
        return side + penalty

    def polish(params):
        """
        Local search refinement to reduce penalty and side length.
        """
        best_p = params.copy()
        best_f = objective(best_p)
        
        # Simple coordinate descent / local search
        step = 0.05
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            improved = False
            for k in range(len(best_p)):
                for delta in [step, -step]:
                    trial = best_p.copy()
                    trial[k] += delta
                    f_trial = objective(trial)
                    if f_trial < best_f:
                        best_p = trial
                        best_f = f_trial
                        improved = True
            if not improved:
                step *= 0.7
                if step < 1e-5:
                    break
                    
        return best_p

    # --- GENERATE INITIAL CONFIGURATIONS ---
    
    inits = []
    
    # 1. Honeycomb lattice (standard dense packing)
    # Lattice vectors: v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append(np.concatenate([init_pos.flatten(), init_ang]))
    
    # 2. Ring configuration
    ring_pos = []
    ring_ang = []
    r = 2.0
    for i in range(N):
        a = 2 * math.pi * i / N
        ring_pos.append([r * math.cos(a), r * math.sin(a)])
        ring_ang.append(a)
    ring_pos = np.array(ring_pos)
    ring_pos -= np.mean(ring_pos, axis=0)
    inits.append(np.concatenate([ring_pos.flatten(), np.array(ring_ang)]))
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(8):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, 2*math.pi, N)
        inits.append(np.concatenate([pos.flatten(), ang]))
        
    # 4. "Star" or "Flower" pattern (1 center, 6 around, 5 more?)
    # Just add a few more structured randoms
    for _ in range(5):
        # Cluster around a few centers
        centers = np.random.uniform(-2, 2, (4, 2))
        pos = []
        ang = []
        for i in range(N):
            c = centers[i % 4]
            offset = np.random.normal(0, 0.5, 2)
            pos.append(c + offset)
            ang.append(np.random.uniform(0, 2*math.pi))
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        inits.append(np.concatenate([pos.flatten(), np.array(ang)]))

    best_side = float('inf')
    best_params = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    # --- SIMULATED ANNEALING SEARCH ---
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        current = x0.copy()
        current_side = calc_outer_side(current)
        current_penalty = calc_penalty(current)
        current_obj = current_side + current_penalty
        
        best_local = current.copy()
        best_local_obj = current_obj
        
        # Annealing schedule
        T = 1.0
        cooling = 0.95
        max_iters = 3000
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Perturbation
            trial = current.copy()
            i = np.random.randint(N)
            dx = np.random.normal(0, 0.15)
            dy = np.random.normal(0, 0.15)
            da = np.random.normal(0, 0.15)
            
            trial[2*i] += dx
            trial[2*i+1] += dy
            trial[2*N+i] += da
            
            trial_side = calc_outer_side(trial)
            trial_penalty = calc_penalty(trial)
            trial_obj = trial_side + trial_penalty
            
            delta = trial_obj - current_obj
            
            if delta < 0 or np.random.rand() < math.exp(-delta / T):
                current = trial
                current_obj = trial_obj
                
                if current_obj < best_local_obj:
                    best_local = current.copy()
                    best_local_obj = current_obj
            
            T *= cooling
            
            if T < 0.001:
                break
        
        # Polish the best local found
        if best_local_obj < best_side:
            polished = polish(best_local)
            pol_side = calc_outer_side(polished)
            pol_penalty = calc_penalty(polished)
            pol_obj = pol_side + pol_penalty
            
            if pol_obj < best_side and pol_penalty < 1e-4:
                best_side = pol_side
                best_params = polished.copy()
        
        # Also check raw best_local
        if best_local_obj < best_side:
            bl_side = calc_outer_side(best_local)
            bl_penalty = calc_penalty(best_local)
            if bl_penalty < 1e-4 and bl_side < best_side:
                best_side = bl_side
                best_params = best_local.copy()

    # --- FINAL POLISH: Optimize outer center and angle ---
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        ang = best_params[2*N:]
        
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2] # radians
            side = calc_outer_side(best_params, outer_center=oc, outer_angle=np.degrees(oa))
            # We must ensure no overlap is introduced? 
            # Moving outer hex doesn't change inner positions, so overlap is unchanged.
            # We just want to minimize side length.
            return side

        centroid = np.mean(pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])
        
        try:
            res_outer = minimize(
                outer_obj, 
                x0_outer, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res_outer.fun < best_side:
                best_side = res_outer.fun
                best_outer_center = res_outer.x[:2]
                best_outer_angle = np.degrees(res_outer.x[2])
        except:
            pass

    # Fallback
    if best_params is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        best_params = np.concatenate([pos.flatten(), ang])
        best_side = calc_outer_side(best_params)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    # Construct return values
    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle
