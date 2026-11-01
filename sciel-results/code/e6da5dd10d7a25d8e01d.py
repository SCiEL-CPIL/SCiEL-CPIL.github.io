import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def check_validity(positions, tol=1e-5):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                # SAT check
                disjoint = False
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                
                if not disjoint:
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    def objective_and_grad(params):
        """
        Objective function: Outer side length + Penalty for overlaps.
        params: (2N,) array of [x0, y0, x1, y1, ...]
        """
        positions = params.reshape(N, 2)
        
        # 1. Compute outer side length
        side = required_outer_side(positions)
        
        # 2. Compute overlap penalty
        # We approximate the minimum distance between hexagons.
        # For unit hexagons, if distance between centers < 2, they might overlap.
        # A more accurate penalty uses the actual geometric distance.
        # However, for gradient-based optimization, a smooth approximation is often sufficient.
        # Let's use a penalty based on the sum of squared overlaps of bounding circles (radius 1)
        # or better, a smooth approximation of the hexagon distance.
        
        # Actually, let's use a simpler, robust penalty:
        # Sum of max(0, (dist - 2)^2) is not right because hexagons touch at distance 2 only in specific directions.
        # The minimum distance between two unit hexagons with parallel sides is 2 (flat-to-flat) or sqrt(3) (point-to-point)?
        # Flat-to-flat distance is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732.
        # Point-to-point distance is 2 * side = 2.
        # So if center distance < sqrt(3), they definitely overlap.
        # If center distance > 2, they definitely don't.
        
        # Let's use a penalty that is zero when distance >= 2 and grows smoothly when < 2.
        # This is conservative but safe.
        penalty = 0.0
        grad_penalty = np.zeros_like(positions)
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 2.0:
                    # Smooth penalty: 0.5 * (2 - dist)^2
                    # This is 0 when dist >= 2
                    # Gradient w.r.t pos_i: (2 - dist) * (diff / dist)
                    # Gradient w.r.t pos_j: (2 - dist) * (-diff / dist)
                    if dist > 1e-6:
                        factor = (2.0 - dist) / dist
                        penalty += 0.5 * (2.0 - dist) ** 2
                        grad_penalty[i] += factor * diff
                        grad_penalty[j] -= factor * diff
                    else:
                        # If they are on top of each other, push them apart randomly
                        penalty += 100.0
                        grad_penalty[i] += np.random.normal(0, 1, 2)
                        grad_penalty[j] -= np.random.normal(0, 1, 2)
        
        # Total objective
        total = side + 10.0 * penalty # Weight for penalty
        
        # Gradient of side length
        # The side length is the max of 6 linear functions of the vertices.
        # We need to propagate the gradient from the max to the positions.
        # This is complex to derive analytically for the max of maxes.
        # Instead, we can use finite differences for the side length part if analytic is too hard,
        # or rely on the fact that L-BFGS-B can handle non-differentiable points poorly.
        
        # Better approach: Use the analytic gradient of the side length.
        # side = max_k ( c_k . v )
        # where v are vertices.
        # The gradient of side w.r.t position i is the sum of gradients of the active constraints.
        
        # Let's compute the active constraints for the side length.
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        max_val = np.max(all_L)
        
        # Active constraints
        active = np.isclose(all_L, max_val, atol=1e-6)
        
        # Gradient of side w.r.t vertices
        grad_side_verts = np.zeros_like(verts_flat)
        # L1: dx=1, dy=1/sqrt3
        # L2: dx=0, dy=2/sqrt3
        # L3: dx=-1, dy=1/sqrt3
        # L4: dx=-1, dy=-1/sqrt3
        # L5: dx=0, dy=-2/sqrt3
        # L6: dx=1, dy=-1/sqrt3
        
        # We need to map the gradient from vertices back to positions.
        # Vertex v of hexagon i is:
        # x_v = x_i + cos(a_k)
        # y_v = y_i + sin(a_k)
        # So d(side)/dx_i = sum over active constraints of d(L)/dx_v * d(x_v)/dx_i
        # d(x_v)/dx_i = 1, d(y_v)/dx_i = 0
        # d(x_v)/dy_i = 0, d(y_v)/dy_i = 1
        
        grad_side_pos = np.zeros((N, 2))
        
        # Coefficients for L1..L6
        coeffs = np.array([
            [1, 1/sqrt3],
            [0, 2/sqrt3],
            [-1, 1/sqrt3],
            [-1, -1/sqrt3],
            [0, -2/sqrt3],
            [1, -1/sqrt3]
        ])
        
        # For each hexagon i, and each vertex k
        for i in range(N):
            for k in range(6):
                # Vertex index in flat array
                v_idx = i * 6 + k
                # Check if this vertex is part of any active constraint
                # Actually, the max is over all vertices and all 6 directions.
                # So we need to sum the contributions from all active (direction, vertex) pairs.
                
                # Let's restructure:
                # active is (N*6, 6)
                # active[v_idx, d] is True if vertex v_idx in direction d is active.
                
                # Gradient contribution from this vertex to position i
                gx = 0.0
                gy = 0.0
                for d in range(6):
                    if active[v_idx, d]:
                        gx += coeffs[d, 0]
                        gy += coeffs[d, 1]
                
                grad_side_pos[i, 0] += gx
                grad_side_pos[i, 1] += gy
        
        # Total gradient
        grad_total = grad_side_pos + 10.0 * grad_penalty
        grad_total = grad_total.flatten()
        
        return total, grad_total

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1))
    
    # 2. Circular arrangement
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init3)
    
    best_positions = None
    best_side = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, spread out
        if not check_validity(init):
            for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = init * scale
                if check_validity(scaled):
                    init = scaled
                    break
            else:
                continue
        
        # Run L-BFGS-B
        x0 = init.flatten()
        try:
            res = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_side:
                pos = res.x.reshape(N, 2)
                if check_validity(pos):
                    side = required_outer_side(pos)
                    if side < best_side:
                        best_side = side
                        best_positions = pos
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        best_positions = configs[0]
        best_side = required_outer_side(best_positions)
    
    # Final validation
    if not check_validity(best_positions):
        # Try to find a valid config
        for init in configs:
            pos = init.copy()
            for scale in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = pos * scale
                if check_validity(scaled):
                    best_positions = scaled
                    best_side = required_outer_side(best_positions)
                    break
            if check_validity(best_positions):
                break
        else:
            # Last resort: spread out linearly
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
