import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0

    def get_hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def hex_gap(h1, h2, eps=1e-6):
        """Return positive gap if disjoint, negative if overlapping, 0 if touching."""
        x1, y1, a1 = h1
        x2, y2, a2 = h2
        v1 = get_hex_vertices(x1, y1, 1.0, math.radians(a1))
        v2 = get_hex_vertices(x2, y2, 1.0, math.radians(a2))
        
        # Get all edge normals from both hexagons
        axes = []
        for verts in [v1, v2]:
            for i in range(6):
                dx = verts[(i+1)%6, 0] - verts[i, 0]
                dy = verts[(i+1)%6, 1] - verts[i, 1]
                length = math.sqrt(dx*dx + dy*dy)
                if length > 1e-10:
                    axes.append(np.array([-dy/length, dx/length]))
        
        min_gap = float('inf')
        for axis in axes:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            gap = min(proj2.max() - proj1.min(), proj1.max() - proj2.min())
            if gap < min_gap:
                min_gap = gap
        
        return min_gap

    def hexagons_disjoint(inner, eps=1e-6):
        n = len(inner)
        for i in range(n):
            for j in range(i+1, n):
                if hex_gap(inner[i], inner[j], eps) < -eps:
                    return False
        return True

    def compute_min_outer_side(inner, cx, cy, theta_deg):
        """Compute minimum outer hexagon side length for given center and angle."""
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for x, y, a in inner:
                verts = get_hex_vertices(x, y, 1.0, math.radians(a))
                proj = (verts[:, 0] - cx) * nx + (verts[:, 1] - cy) * ny
                if proj.max() > max_proj:
                    max_proj = proj.max()
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        """Objective: outer side length + smooth barrier for overlaps."""
        if time.time() > deadline - 2.0:
            return 1e10
        
        inner = []
        for i in range(N):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            inner.append([x, y, math.degrees(a)])
        cx, cy, theta = params[3*N], params[3*N+1], params[3*N+2]
        theta_deg = math.degrees(theta)
        
        side = compute_min_outer_side(inner, cx, cy, theta_deg)
        
        # Smooth barrier for overlaps
        penalty = 0.0
        beta = 30.0
        margin = 0.005
        for i in range(N):
            for j in range(i+1, N):
                gap = hex_gap(inner[i], inner[j])
                if gap < 0.05:
                    val = gap + margin
                    penalty += beta * np.logaddexp(0.0, -beta * val)
        
        return side + penalty

    def local_search(inner, cx, cy, theta_deg, max_iter=300, step_size=0.02):
        """Random local search to improve configuration."""
        best_side = compute_min_outer_side(inner, cx, cy, theta_deg)
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, theta_deg
        
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, theta_deg
        current_side = best_side
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for _ in range(30):
                new_inner = [list(h) for h in current_inner]
                new_cx, new_cy = current_cx, current_cy
                new_theta = current_theta
                
                r = random.random()
                if r < 0.75:
                    idx = random.randint(0, len(new_inner)-1)
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_inner[idx][0] += dx
                    new_inner[idx][1] += dy
                elif r < 0.88:
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_cx += dx
                    new_cy += dy
                else:
                    dtheta = random.uniform(-1.0, 1.0)
                    new_theta += dtheta
                
                if not hexagons_disjoint(new_inner):
                    continue
                
                new_side = compute_min_outer_side(new_inner, new_cx, new_cy, new_theta)
                
                if new_side < current_side - 1e-9:
                    current_inner = new_inner
                    current_cx, current_cy, current_theta = new_cx, new_cy, new_theta
                    current_side = new_side
                    improved = True
                    step_size *= 0.97
                    break
            
            if not improved:
                step_size *= 0.85
                if step_size < 0.0005:
                    break
            
            if current_side < best_side - 1e-11:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        return best_inner, best_cx, best_cy, best_theta, best_side

    def scipy_optimize(inner, cx, cy, theta_deg, max_time=25.0):
        """Use scipy SLSQP to optimize with constraints."""
        if time.time() > deadline - 10.0:
            return None
        
        params = []
        for x, y, a in inner:
            params.extend([x, y, math.radians(a)])
        params.extend([cx, cy, math.radians(theta_deg)])
        
        best_side = compute_min_outer_side(inner, cx, cy, theta_deg)
        best_result = None
        
        try:
            result = minimize(
                objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 800, 'maxfun': 5000, 'maxls': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_side:
                opt_inner = []
                for i in range(N):
                    x, y, a = result.x[3*i], result.x[3*i+1], result.x[3*i+2]
                    opt_inner.append([x, y, math.degrees(a)])
                opt_cx, opt_cy = result.x[3*N], result.x[3*N+1]
                opt_theta = math.degrees(result.x[3*N+2])
                
                if hexagons_disjoint(opt_inner):
                    opt_side = compute_min_outer_side(opt_inner, opt_cx, opt_cy, opt_theta)
                    if opt_side < best_side:
                        best_result = (opt_inner, opt_cx, opt_cy, opt_theta, opt_side)
                        best_side = opt_side
                        
        except Exception:
            pass
        
        return best_result

    def generate_ring12():
        """Generate 1+6+5 ring topology with touching hexagons."""
        d = math.sqrt(3)
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v-u, -u, -v, u-v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        
        # Generate ring 2 candidates
        ring2_candidates = []
        basis = [u, v, v-u, -u, -v, u-v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=1e-9)
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=1e-9):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc, atol=1e-9):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
        
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
        
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_linear12():
        """Generate a linear chain of 12 hexagons."""
        d = math.sqrt(3)
        centers = []
        for i in range(12):
            centers.append(np.array([i * d, 0.0]))
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_2x6_grid():
        """Generate a 2x6 grid of hexagons."""
        d = math.sqrt(3)
        row_spacing = d * math.sqrt(3)
        centers = []
        for row in range(2):
            for col in range(6):
                x = col * d
                y = row * row_spacing
                centers.append(np.array([x, y]))
        return [[c[0], c[1], 30.0] for c in centers]

    def evaluate_topology(inner):
        """Evaluate a topology and return best configuration."""
        all_verts = []
        for x, y, a in inner:
            verts = get_hex_vertices(x, y, 1.0, math.radians(a))
            all_verts.extend(verts)
        
        cx = sum(p[0] for p in all_verts) / len(all_verts)
        cy = sum(p[1] for p in all_verts) / len(all_verts)
        
        best_side = float('inf')
        best_angle = 0.0
        
        for theta_deg in np.linspace(0, 60, 121):
            s = compute_min_outer_side(inner, cx, cy, theta_deg)
            if s < best_side:
                best_side = s
                best_angle = theta_deg
        
        return inner, cx, cy, best_angle, best_side

    # Generate multiple topologies
    topologies = [
        generate_ring12(),
        generate_linear12(),
        generate_2x6_grid(),
    ]
    
    best_overall = None
    best_overall_side = float('inf')
    
    for inner in topologies:
        if time.time() > deadline - 30.0:
            break
        
        if not hexagons_disjoint(inner):
            continue
        
        inner_eval, cx_eval, cy_eval, angle_eval, side_eval = evaluate_topology(inner)
        
        if side_eval < best_overall_side:
            best_overall = (inner_eval, cx_eval, cy_eval, angle_eval, side_eval)
            best_overall_side = side_eval
    
    if best_overall is None:
        inner = generate_ring12()
        best_overall = evaluate_topology(inner)
        best_overall_side = best_overall[4]
    
    inner, cx, cy, best_angle, best_side = best_overall
    
    # Local optimization
    if time.time() < deadline - 25.0:
        inner_opt, cx_opt, cy_opt, theta_opt, side_opt = local_search(
            inner, cx, cy, best_angle, max_iter=200, step_size=0.02
        )
        if side_opt < best_side:
            inner = inner_opt
            cx, cy = cx_opt, cy_opt
            best_angle = theta_opt
            best_side = side_opt
    
    # Scipy optimization
    if time.time() < deadline - 15.0:
        scipy_result = scipy_optimize(inner, cx, cy, best_angle, max_time=20.0)
        if scipy_result is not None:
            inner_sc, cx_sc, cy_sc, theta_sc, side_sc = scipy_result
            if side_sc < best_side:
                inner = inner_sc
                cx, cy = cx_sc, cy_sc
                best_angle = theta_sc
                best_side = side_sc
    
    # Final polish
    if time.time() < deadline - 5.0:
        inner_polish, cx_polish, cy_polish, theta_polish, side_polish = local_search(
            inner, cx, cy, best_angle, max_iter=500, step_size=0.005
        )
        if side_polish < best_side:
            inner = inner_polish
            cx, cy = cx_polish, cy_polish
            best_angle = theta_polish
            best_side = side_polish
    
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
