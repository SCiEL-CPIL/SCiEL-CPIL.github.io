import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        """
        Returns True if v1 and v2 are disjoint (including touching).
        """
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        return False

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def calculate_cost(centers, angles, ocx, ocy, otheta):
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # Check for overlaps
        penalty = 0.0
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    penalty += 100.0
        
        return side + penalty

    def optimize_sa(centers_init, angles_init, ocx_init, ocy_init, otheta_init):
        # Simulated Annealing
        T = 1.0
        T_min = 0.001
        alpha = 0.995
        steps_per_temp = 50
        
        current_centers = [c.copy() for c in centers_init]
        current_angles = angles_init.copy()
        current_ocx = ocx_init
        current_ocy = ocy_init
        current_otheta = otheta_init
        
        current_cost = calculate_cost(current_centers, current_angles, current_ocx, current_ocy, current_otheta)
        
        best_centers = [c.copy() for c in current_centers]
        best_angles = current_angles.copy()
        best_ocx = current_ocx
        best_ocy = current_ocy
        best_otheta = current_otheta
        best_cost = current_cost
        
        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 10:
                    break
                
                # Generate neighbor
                new_centers = [c.copy() for c in current_centers]
                new_angles = current_angles.copy()
                new_ocx = current_ocx
                new_ocy = current_ocy
                new_otheta = current_otheta
                
                # Perturb one inner hexagon
                i = random.randint(0, 11)
                step_size = T * 0.5
                new_centers[i][0] += random.gauss(0, step_size)
                new_centers[i][1] += random.gauss(0, step_size)
                new_angles[i] += random.gauss(0, step_size * 10)
                
                # Occasionally perturb outer hexagon
                if random.random() < 0.1:
                    new_ocx += random.gauss(0, T * 0.1)
                    new_ocy += random.gauss(0, T * 0.1)
                    new_otheta += random.gauss(0, T * 0.1)
                
                new_cost = calculate_cost(new_centers, new_angles, new_ocx, new_ocy, new_otheta)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_centers = new_centers
                    current_angles = new_angles
                    current_ocx = new_ocx
                    current_ocy = new_ocy
                    current_otheta = new_otheta
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_centers = [c.copy() for c in current_centers]
                        best_angles = current_angles.copy()
                        best_ocx = current_ocx
                        best_ocy = current_ocy
                        best_otheta = current_otheta
            
            T *= alpha
            if time.time() > deadline - 10:
                break
                
        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost

    def optimize_slsqp(centers_init, angles_init, ocx_init, ocy_init, otheta_init):
        """
        Use SLSQP to optimize the packing with explicit separation constraints.
        """
        def flatten(x, y, a, ox, oy, oa):
            return [x[0], x[1], y[0], y[1], a, ox, oy, oa]
        
        def unflatten(v):
            x = (v[0], v[1])
            y = (v[2], v[3])
            a = v[4]
            ox = v[5]
            oy = v[6]
            oa = v[7]
            return x, y, a, ox, oy, oa
        
        def objective(v):
            if time.time() > deadline - 5:
                return 1e6
            x, y, a, ox, oy, oa = unflatten(v)
            centers = [x, y]
            angles = [a]
            return calculate_cost(centers, angles, ox, oy, oa)
        
        # Start with a simple 2-hexagon configuration to test
        # Then expand to full 12-hexagon
        
        # For now, just run SA and then do a local polish with SLSQP on the outer parameters
        best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init
        )
        
        # Now do a local optimization of the outer hexagon position and angle
        def outer_objective(v):
            if time.time() > deadline - 5:
                return 1e6
            ox, oy, oa = v[0], v[1], v[2]
            return compute_outer_side(best_centers, best_angles, ox, oy, oa)
        
        result = minimize(outer_objective, [best_ocx, best_ocy, best_otheta], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.success or result.fun < best_cost:
            best_ocx, best_ocy, best_otheta = result.x
            best_cost = result.fun
        
        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost

    # Generate initial configuration: Hexagonal Lattice
    d = math.sqrt(3) + 0.01
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    
    centers_init = [np.array([0.0, 0.0])]
    ring1_vectors = [
        u, v, v - u, -u, -v, u - v
    ]
    for vec in ring1_vectors:
        centers_init.append(vec.copy())
        
    ring2_candidates = []
    basis = [u, v, v - u, -u, -v, u - v]
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers_init.append(ring2_candidates[i].copy())
        
    angles_init = [30.0] * 12
    ocx_init = 0.0
    ocy_init = 0.0
    otheta_init = 0.0
    
    # Run optimization
    best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost = optimize_slsqp(
        centers_init, angles_init, ocx_init, ocy_init, otheta_init
    )
    
    # Final Polish: Grid search on outer parameters
    best_side = compute_outer_side(best_centers, best_angles, best_ocx, best_ocy, best_otheta)
    final_ocx, final_ocy, final_otheta = best_ocx, best_ocy, best_otheta
    
    # Refine outer hexagon orientation and position with finer grid
    for da in np.linspace(-10, 10, 21):
        for dx in np.linspace(-0.3, 0.3, 7):
            for dy in np.linspace(-0.3, 0.3, 7):
                s = compute_outer_side(best_centers, best_angles, best_ocx + dx, best_ocy + dy, best_otheta + da)
                if s < best_side:
                    best_side = s
                    final_ocx, final_ocy, final_otheta = best_ocx + dx, best_ocy + dy, best_otheta + da
                    
    # Verify final validity
    verts = []
    for i in range(12):
        v = get_hex_vertices(best_centers[i][0], best_centers[i][1], 1.0, best_angles[i])
        verts.append(v)
        
    valid = True
    for i in range(12):
        for j in range(i + 1, 12):
            if not are_disjoint_sat(verts[i], verts[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to initial lattice if SA resulted in invalid configuration
        best_centers = [c.copy() for c in centers_init]
        best_angles = angles_init
        final_ocx, final_ocy, final_otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(best_centers, best_angles, final_ocx, final_ocy, final_otheta)

    inner = [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)]
    outer_center = (final_ocx, final_ocy)
    outer_side = best_side
    outer_angle = final_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
