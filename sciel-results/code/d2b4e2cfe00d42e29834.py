import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
MIN_DIST = SQRT3 - 1e-9  # Minimum center distance for disjointness

def construct():
    deadline = time.time() + 142.0  # Safety margin for 150s limit
    
    # --- Helper Functions ---
    
    def hex_vertices(cx, cy, angle_deg):
        """Calculate vertices of a unit hexagon."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * (math.pi / 3)
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return verts

    def check_disjoint_fast(centers, angles):
        """
        Fast check for disjointness based on center distances.
        This is a necessary condition but not sufficient if rotations are arbitrary,
        but for regular hexagons, min distance between centers is sqrt(3) if aligned,
        and slightly larger if rotated? Actually, min distance is sqrt(3) for parallel edges.
        If rotated 30 deg, the "width" is 2, height is sqrt(3).
        The minimum distance between centers of two disjoint unit regular hexagons is indeed sqrt(3).
        So this check is necessary. Is it sufficient?
        If dist >= sqrt(3), they are disjoint?
        Yes, for regular hexagons of side 1, the minimum distance between centers of two disjoint hexagons is sqrt(3).
        """
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx * dx + dy * dy < MIN_DIST * MIN_DIST:
                    return False
        return True

    def check_disjoint_sat(centers, angles):
        """
        Rigorous SAT check for disjointness.
        """
        n = len(centers)
        verts_list = [np.array(hex_vertices(centers[i][0], centers[i][1], angles[i])) for i in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                v1 = verts_list[i]
                v2 = verts_list[j]
                
                # Edges
                e1 = np.roll(v1, -1, axis=0) - v1
                e2 = np.roll(v2, -1, axis=0) - v2
                
                # Normals
                n1 = np.column_stack([-e1[:, 1], e1[:, 0]])
                n2 = np.column_stack([-e2[:, 1], e2[:, 0]])
                
                norms = np.vstack([n1, n2])
                norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-15)
                
                # Projections
                proj1 = v1 @ norms.T
                proj2 = v2 @ norms.T
                
                min1 = proj1.min(axis=0)
                max1 = proj1.max(axis=0)
                min2 = proj2.min(axis=0)
                max2 = proj2.max(axis=0)
                
                # Separating axis theorem
                if np.any(max1 < min2 - 1e-9) or np.any(max2 < min1 - 1e-9):
                    continue
                else:
                    return False
        return True

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """
        Compute the minimum side length of a regular hexagon centered at (ocx, ocy)
        with rotation otheta_deg that contains all inner hexagons.
        """
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        
        # The 6 normal directions of the outer hexagon
        for k in range(6):
            na = otheta + math.pi / 6 + k * (math.pi / 3)
            nx = math.cos(na)
            ny = math.sin(na)
            
            for i in range(N):
                cx, cy = centers[i]
                ang = math.radians(angles[i])
                # Vertices of inner hexagon i
                for vi in range(6):
                    va = ang + vi * (math.pi / 3)
                    vx = cx + math.cos(va)
                    vy = cy + math.sin(va)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # For a regular hexagon, side length s = 2 * apothem / sqrt(3)
        # apothem is the max projection from center to boundary
        return 2.0 * max_proj / SQRT3

    def unpack(x):
        """Unpack 39-dim vector."""
        centers = [[x[3*i], x[3*i+1]] for i in range(N)]
        angles = [x[3*i+2] for i in range(N)]
        ocx, ocy, otheta = x[3*N], x[3*N+1], x[3*N+2]
        return centers, angles, ocx, ocy, otheta

    def pack_to_vector(centers, angles, ocx, ocy, otheta):
        """Pack to 39-dim vector."""
        x = []
        for i in range(N):
            x.extend([centers[i][0], centers[i][1], angles[i]])
        x.extend([ocx, ocy, otheta])
        return x

    # --- Initialization ---
    
    def generate_ring_configs():
        """
        Generate initial configurations based on a central hexagon,
        a ring of 6, and a second ring of 5.
        """
        configs = []
        # Distance for touching hexagons
        d = SQRT3 + 1e-5
        
        # Basis vectors for the 6 neighbors
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*SQRT3/2.0])
        
        # 6 vectors for ring 1
        ring1_vectors = [
            u, v, v-u, -u, -v, u-v
        ]
        
        # Generate candidates for ring 2 (neighbors of ring 1, excluding ring 0 and ring 1)
        ring2_candidates = []
        for r1 in ring1_vectors:
            for b in [u, v, v-u, -u, -v, u-v]:
                p = r1 + b
                # Check if p is origin
                if np.linalg.norm(p) < 0.1:
                    continue
                # Check if p is in ring 1
                is_r1 = False
                for r1j in ring1_vectors:
                    if np.linalg.norm(p - r1j) < 0.1:
                        is_r1 = True
                        break
                if is_r1:
                    continue
                # Check if p is already in candidates
                dup = False
                for rc in ring2_candidates:
                    if np.linalg.norm(p - rc) < 0.1:
                        dup = True
                        break
                if not dup:
                    ring2_candidates.append(p)
        
        # Sort candidates by distance from origin to prefer tighter packing
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        
        # We need 5 vectors for the second ring.
        # Strategy 1: Take the 5 closest.
        # Strategy 2: Take different subsets to explore topology.
        
        # Config 0: Closest 5
        centers0 = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers0.append([vec[0], vec[1]])
        for vec in ring2_candidates[:5]:
            centers0.append([vec[0], vec[1]])
        configs.append((centers0, [0.0]*N))
        
        # Config 1-5: Random subsets of ring 2 candidates
        random.seed(42)
        used_subsets = [tuple(sorted([i for i in range(5) if ring2_candidates[i] in ring2_candidates[:5]]))] # Not strictly necessary to track, just generate diverse ones
        
        for k in range(5):
            # Pick 5 random distinct candidates
            # To ensure diversity, we can rotate the selection window or just shuffle
            idxs = list(range(len(ring2_candidates)))
            random.shuffle(idxs)
            selected_idxs = idxs[:5]
            
            centers_k = [[0.0, 0.0]]
            for vec in ring1_vectors:
                centers_k.append([vec[0], vec[1]])
            for idx in selected_idxs:
                vec = ring2_candidates[idx]
                centers_k.append([vec[0], vec[1]])
            
            # Basic validity check: min dist
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers_k[i][0] - centers_k[j][0]
                    dy = centers_k[i][1] - centers_k[j][1]
                    if dx*dx + dy*dy < MIN_DIST**2 - 1e-4:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                configs.append((centers_k, [0.0]*N))
                
        return configs

    # --- Optimization Routines ---
    
    def slsqp_optimize(centers, angles, max_iters=100):
        """
        SLSQP optimization with smooth constraints.
        Variables: x, y, angle for each hex, plus outer hex center and angle.
        """
        n = N
        x0 = pack_to_vector(centers, angles, 0.0, 0.0, 0.0)
        
        def objective(x):
            if time.time() > deadline - 20:
                return 1e10
            c, a, ocx, ocy, otheta = unpack(x)
            # Use fast check to avoid expensive SAT in objective if possible?
            # SLSQP needs differentiable objective.
            # The outer side calculation is differentiable almost everywhere.
            # Constraints are distance squared.
            return compute_outer_side(c, a, ocx, ocy, otheta)
        
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                def dist_constraint(x, i=i, j=j):
                    xi, yi = x[3*i], x[3*i+1]
                    xj, yj = x[3*j], x[3*j+1]
                    return (xi - xj)**2 + (yi - yj)**2 - MIN_DIST**2
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        
        bounds = []
        for i in range(n):
            bounds.extend([(-5, 5), (-5, 5), (-90, 90)])
        bounds.extend([(-3, 3), (-3, 3), (-30, 30)])
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints,
                             options={'maxiter': max_iters, 'ftol': 1e-10, 'disp': False})
            if result.fun < 1e8:
                c, a, ocx, ocy, otheta = unpack(result.x)
                # Validate with SAT
                if check_disjoint_sat(c, a):
                    side = compute_outer_side(c, a, ocx, ocy, otheta)
                    return c, a, ocx, ocy, otheta, side
        except Exception:
            pass
        return None

    def nm_polish(x_start, max_iter=500):
        """
        Nelder-Mead polish. Uses fast distance check for constraints during search.
        """
        def nm_objective(x):
            c, a, ocx, ocy, otheta = unpack(x)
            if not check_disjoint_fast(c, a):
                return 1e8
            return compute_outer_side(c, a, ocx, ocy, otheta)
        
        try:
            res = minimize(nm_objective, x_start, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-10})
            if res.fun < 1e7:
                c, a, ocx, ocy, otheta = unpack(res.x)
                # Final validation with SAT
                if check_disjoint_sat(c, a):
                    side = compute_outer_side(c, a, ocx, ocy, otheta)
                    return res.x, side
        except Exception:
            pass
        return x_start, nm_objective(x_start)

    def polish_outer(centers, angles, ocx, ocy, otheta):
        """
        Fine-tune the outer hexagon parameters (center and rotation)
        assuming inner positions are fixed.
        """
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        
        # Multi-scale local search
        scales = [0.1, 0.01, 0.001, 0.0001]
        for scale in scales:
            if time.time() > deadline - 5:
                break
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da in [-scale*10, 0, scale*10]:
                            s = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, best_otheta + da)
                            if s < best_side - 1e-12:
                                best_side = s
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta += da
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    # --- Main Execution ---
    
    best_x = None
    best_side = float('inf')
    
    configs = generate_ring_configs()
    
    # Phase 1: SLSQP from each initial config
    for centers, angles in configs:
        if time.time() > deadline - 40:
            break
        result = slsqp_optimize(centers, angles, max_iters=100)
        if result is not None:
            c, a, ocx, ocy, otheta, side = result
            if side < best_side:
                best_side = side
                best_x = pack_to_vector(c, a, ocx, ocy, otheta)
    
    # If no valid SLSQP result, use the first config
    if best_x is None:
        centers, angles = configs[0]
        best_x = pack_to_vector(centers, angles, 0.0, 0.0, 0.0)
        best_side = compute_outer_side(centers, angles, 0.0, 0.0, 0.0)
        
    # Phase 2: Nelder-Mead polish from best SLSQP result
    if time.time() < deadline - 25:
        best_x, best_side = nm_polish(best_x, max_iter=400)
        
    # Phase 3: Additional NM restarts from perturbed best solution
    if time.time() < deadline - 30:
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            x_trial = list(best_x)
            for i in range(N):
                # Perturb positions and angles slightly
                x_trial[3*i] += random.uniform(-0.05, 0.05)
                x_trial[3*i+1] += random.uniform(-0.05, 0.05)
                x_trial[3*i+2] += random.uniform(-10, 10)
            x_trial = np.array(x_trial)
            
            x_polished, side_polished = nm_polish(x_trial, max_iter=300)
            if side_polished < best_side:
                best_side = side_polished
                best_x = x_polished
                
    # Phase 4: Final high-precision polish of outer hexagon
    centers, angles, ocx, ocy, otheta = unpack(best_x)
    if time.time() < deadline - 5:
        ocx, ocy, otheta, best_side = polish_outer(centers, angles, ocx, ocy, otheta)
        
    # Build return value
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = otheta

    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
