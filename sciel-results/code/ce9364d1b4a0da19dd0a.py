import math
import time
import random
import numpy as np
from scipy.optimize import minimize, Bounds

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute minimum outer hexagon side for given configuration."""
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.0001):
        """Check if all hexagons are disjoint (center distance >= sqrt(3))."""
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    return False
        return True

    def make_ring_config(ring1_vectors, ring2_vectors):
        """Create a configuration from ring vectors."""
        centers = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers.append([vec[0], vec[1]])
        for vec in ring2_vectors:
            centers.append([vec[0], vec[1]])
        return centers

    def optimize_with_slsqp(centers, angles, max_iters=100):
        """Use SLSQP to optimize positions and outer hexagon parameters."""
        n = len(centers)
        
        # Decision variables: x1,y1, a1, x2,y2, a2, ..., xn,yn, an, ocx, ocy, otheta
        # Total: 3n + 3 variables
        x0 = []
        for i in range(n):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        x0.extend([0.0, 0.0, 0.0])  # outer center and angle
        
        def objective(x):
            if time.time() > deadline - 30:
                return 1e10
            c = [[x[3*i], x[3*i+1]] for i in range(n)]
            a = [x[3*i+2] for i in range(n)]
            ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
            side = compute_outer_side(c, a, ocx, ocy, otheta)
            return side
        
        # Constraints: pairwise distance >= sqrt(3)
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                def dist_constraint(x, i=i, j=j):
                    xi = x[3*i]
                    yi = x[3*i+1]
                    xj = x[3*j]
                    yj = x[3*j+1]
                    return (xi - xj)**2 + (yi - yj)**2 - (math.sqrt(3) - 0.0001)**2
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.extend([(-5, 5), (-5, 5), (0, 60)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                             constraints=constraints, options={'maxiter': max_iters, 'ftol': 1e-10})
            if result.success:
                c = [[result.x[3*i], result.x[3*i+1]] for i in range(n)]
                a = [result.x[3*i+2] for i in range(n)]
                ocx, ocy, otheta = result.x[3*n], result.x[3*n+1], result.x[3*n+2]
                side = compute_outer_side(c, a, ocx, ocy, otheta)
                return c, a, ocx, ocy, otheta, side
        except Exception:
            pass
        
        return None
    
    def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
        """High-precision polish of outer hexagon parameters."""
        best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
        best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta
        
        # Multi-scale search
        scales = [1.0, 0.1, 0.01, 0.001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale*10, 0, scale*10]:
                            otheta = math.radians(best_otheta + da_deg)
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, otheta)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = (best_otheta + da_deg) % 60
                                improved = True
        
        return best_ocx, best_ocy, best_otheta, best_side

    # Generate multiple initial configurations
    eps = 0.001
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    basis = [u, v, v-u, -u, -v, u-v]
    
    best_result = None
    best_side = float('inf')
    
    # Try different ring 2 selections
    ring1_vectors = [u, v, v-u, -u, -v, u-v]
    
    # Generate all ring 2 candidates
    ring2_candidates = []
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=0.01)
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j, atol=0.01):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc, atol=0.01):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    # Try top K ring 2 selections
    num_configs_to_try = min(6, len(ring2_candidates))
    
    for config_idx in range(num_configs_to_try):
        if time.time() > deadline - 40:
            break
            
        # Select ring 2 vectors: prefer closest, but vary selection
        if config_idx == 0:
            ring2_sel = ring2_candidates[:5]
        else:
            # Different combinations
            idxs = list(range(len(ring2_candidates)))
            random.seed(config_idx * 12345)
            random.shuffle(idxs)
            ring2_sel = [ring2_candidates[i] for i in idxs[:5]]
        
        centers = make_ring_config(ring1_vectors, ring2_sel)
        angles = [30.0] * 12
        
        # Check if valid
        if not check_disjoint(centers, angles):
            continue
        
        # Try SLSQP optimization
        slsqp_result = optimize_with_slsqp(centers, angles, max_iters=50)
        
        if slsqp_result is not None:
            c, a, ocx, ocy, otheta, side = slsqp_result
            # Polish
            ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
            
            if side < best_side:
                best_side = side
                best_result = (c, a, ocx, ocy, otheta, side)
        
        # Also try simple local optimization as fallback
        if time.time() < deadline - 20:
            step_size = 0.05
            local_best_side = compute_outer_side(centers, angles, 0, 0, 0)
            local_best_centers = [c.copy() for c in centers]
            local_best_angles = angles.copy()
            
            for it in range(30):
                if time.time() > deadline - 15:
                    break
                improved = False
                for i in range(12):
                    for dx in [-step_size, 0, step_size]:
                        for dy in [-step_size, 0, step_size]:
                            new_centers = [c.copy() for c in local_best_centers]
                            new_centers[i][0] += dx
                            new_centers[i][1] += dy
                            if check_disjoint(new_centers, local_best_angles):
                                side = compute_outer_side(new_centers, local_best_angles, 0, 0, 0)
                                if side < local_best_side - 1e-10:
                                    local_best_side = side
                                    local_best_centers = [c.copy() for c in new_centers]
                                    improved = True
                if not improved:
                    break
            
            # Polish outer
            ocx, ocy, otheta, side = polish_outer(local_best_centers, local_best_angles, 0, 0, 0)
            
            if side < best_side:
                best_side = side
                best_result = (local_best_centers, local_best_angles, ocx, ocy, otheta, side)
    
    # If no good result, use default configuration
    if best_result is None:
        centers = make_ring_config(ring1_vectors, ring2_candidates[:5])
        angles = [30.0] * 12
        ocx, ocy, otheta, best_side = polish_outer(centers, angles, 0, 0, 0)
        best_result = (centers, angles, ocx, ocy, otheta, best_side)
    
    centers, angles, ocx, ocy, otheta, final_side = best_result
    
    # Final verification and adjustment
    if not check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.0005):
        # Relax slightly if needed
        if not check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.001):
            # Use original configuration
            centers = make_ring_config(ring1_vectors, ring2_candidates[:5])
            angles = [30.0] * 12
            ocx, ocy, otheta, final_side = polish_outer(centers, angles, 0, 0, 0)
    
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (ocx, ocy)
    outer_side = final_side
    outer_angle = otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
