"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """Compute the minimum side length S of the outer hexagon (centered at ocx, ocy, rot oangle) to contain all vertices."""
        # Transform vertices to container's local coordinate system
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                # Translate
                lx = v[0] - ocx
                ly = v[1] - ocy
                # Rotate by -oangle to align container axes
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                # Project onto 6 normals
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def sat_overlap(v1, v2):
        """Check if two convex polygons overlap using Separating Axis Theorem."""
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def soft_overlap_penalty(v1, v2):
        """Compute a smooth penalty for potential overlap using vertex distance."""
        # Compute minimum distance between any vertex of v1 and any vertex of v2
        # This is a smooth surrogate for the SAT overlap check
        min_dist_sq = float('inf')
        for p1 in v1:
            for p2 in v2:
                d2 = np.sum((p1 - p2) ** 2)
                if d2 < min_dist_sq:
                    min_dist_sq = d2
        min_dist = np.sqrt(min_dist_sq)
        # If min_dist < 1.0, hexagons are likely overlapping (circumradius = 1)
        # Use a smooth penalty that kicks in when dist < 1.5
        threshold = 1.5
        if min_dist < threshold:
            excess = threshold - min_dist
            return 500.0 * excess ** 2
        return 0.0

    def objective(params):
        """
        Compute the outer side length S and penalty for overlaps.
        params layout:
        [0:3]   outer center x, y, angle
        [3:3+3N] inner hexagons (cx, cy, angle)
        """
        ocx = params[0]
        ocy = params[1]
        oangle = params[2]
        
        vertices_list = []
        for i in range(N):
            cx = params[3 + 3 * i]
            cy = params[3 + 3 * i + 1]
            angle = params[3 + 3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Overlap penalty - use both SAT for hard check and soft penalty for smoothness
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.5:  # Only check if potentially overlapping
                    # Hard SAT check
                    if sat_overlap(vertices_list[i], vertices_list[j]):
                        penalty += 10000.0 * (2.0 - dist) ** 2
                    else:
                        # Soft penalty for near-misses to guide optimization
                        penalty += soft_overlap_penalty(vertices_list[i], vertices_list[j])
        
        return S + penalty

    def get_initial_params():
        """Generate initial parameters with the flower pattern."""
        p = np.zeros(3 + 3 * N)
        # Outer
        p[0:3] = [0, 0, 0]
        
        # Inner
        idx = 0
        # Center
        p[3 + 3*idx : 3 + 3*idx + 3] = [0, 0, 0]
        idx += 1
        # Ring 1: 6 hexagons at distance sqrt(3) from center
        for k in range(6):
            theta = k * np.pi / 3
            cx = np.sqrt(3) * np.cos(theta)
            cy = np.sqrt(3) * np.sin(theta)
            p[3 + 3*idx : 3 + 3*idx + 3] = [cx, cy, 0]
            idx += 1
        # Ring 2: 4 hexagons at distance 2*sqrt(3) from center
        angles_r2 = [0, 120, 240, 180]
        for deg in angles_r2:
            theta = np.radians(deg)
            cx = 2 * np.sqrt(3) * np.cos(theta)
            cy = 2 * np.sqrt(3) * np.sin(theta)
            p[3 + 3*idx : 3 + 3*idx + 3] = [cx, cy, 0]
            idx += 1
        return p

    def get_random_initial_params(base_params, scale=0.8):
        """Generate perturbed initial parameters."""
        p = base_params.copy()
        noise_inner = np.random.normal(0, scale, size=3 * N)
        noise_outer = np.random.normal(0, scale * 0.2, size=3)
        p[0:3] += noise_outer
        p[3:] += noise_inner
        return p

    best_params = get_initial_params()
    best_score = objective(best_params)
    
    # Multi-start optimization with time management
    np.random.seed(42)
    n_restarts = 25
    
    for i in range(n_restarts):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 10.0:
            break
        
        p = best_params.copy()
        if i > 0:
            scale = 0.6 + 0.2 * (i % 3)  # Vary perturbation scale
            p = get_random_initial_params(best_params, scale)
            
        # Use Powell with moderate iterations
        res = minimize(objective, p, method='Powell', 
                      options={'maxiter': 800, 'maxfev': 4000})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
        
        # Check time after each restart
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 15.0:
            break
    
    # Final high-precision polish pass
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 5.0:
        # Try Nelder-Mead for final polish
        try:
            res_polish = minimize(objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'maxfev': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_params = res_polish.x
        except:
            pass
    
    # Extract results
    ocx, ocy, oangle = best_params[0], best_params[1], best_params[2]
    inner = []
    for i in range(N):
        cx = best_params[3 + 3 * i]
        cy = best_params[3 + 3 * i + 1]
        angle = best_params[3 + 3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # Recalculate S for the best params to be safe
    vertices_list = []
    for i in range(N):
        cx = best_params[3 + 3 * i]
        cy = best_params[3 + 3 * i + 1]
        angle = best_params[3 + 3 * i + 2]
        verts = hex_vertices(cx, cy, angle)
        vertices_list.append(verts)
        
    S_final = compute_S_for_container(vertices_list, ocx, ocy, oangle)
    
    return inner, (ocx, ocy), S_final, np.degrees(oangle)
# EVOLVE-BLOCK-END
