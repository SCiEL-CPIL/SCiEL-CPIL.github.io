import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    random.seed(42)
    np.random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon with given center, side, and rotation."""
        angle = math.radians(angle_deg)
        thetas = angle + np.arange(6) * (math.pi / 3)
        xs = cx + side * np.cos(thetas)
        ys = cy + side * np.sin(thetas)
        return np.column_stack((xs, ys))

    def hexagon_distance(v1, v2):
        """Compute minimum distance between two hexagons (0 if overlapping)."""
        # Use SAT (Separating Axis Theorem) for convex polygons
        # For two regular hexagons, we can use the 6 edge normals of each
        
        def project_polygon(vertices, axis):
            dots = vertices @ axis
            return np.min(dots), np.max(dots)
        
        # Get all 12 edge normals (6 from each hexagon)
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                edge = v[(i+1) % 6] - v[i]
                normal = np.array([-edge[1], edge[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-15)
                axes.append(normal)
        
        min_dist = float('inf')
        for axis in axes:
            min1, max1 = project_polygon(v1, axis)
            min2, max2 = project_polygon(v2, axis)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                return 0.0  # Overlapping
            
            if gap < min_dist:
                min_dist = gap
        
        return min_dist

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Compute the minimum side length of outer hexagon containing all inner hexagons."""
        otheta_rad = math.radians(otheta_deg)
        # The 6 outward normals of the outer hexagon
        normal_angles = otheta_rad + math.pi / 6 + np.arange(6) * (math.pi / 3)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        max_proj = 0.0
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle_rad = math.radians(angles[i])
            thetas = angle_rad + np.arange(6) * (math.pi / 3)
            vx = cx + np.cos(thetas)
            vy = cy + np.sin(thetas)
            
            dx = vx - ocx
            dy = vy - ocy
            projs = dx * nx + dy * ny  # (6, 6)
            current_max = np.max(projs)
            if current_max > max_proj:
                max_proj = current_max
        
        # For a regular hexagon with inradius r, side = 2r/sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def check_validity(centers, angles, ocx, ocy, otheta_deg, side):
        """Check if all hexagons fit inside the outer hexagon and don't overlap."""
        # Check overlap between inner hexagons
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        for i in range(12):
            for j in range(i + 1, 12):
                if hexagon_distance(verts[i], verts[j]) < -1e-7:
                    return False
        
        # Check if all vertices are inside outer hexagon
        otheta_rad = math.radians(otheta_deg)
        normal_angles = otheta_rad + math.pi / 6 + np.arange(6) * (math.pi / 3)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        inradius = side * math.sqrt(3) / 2
        
        for i in range(12):
            v = verts[i]
            dx = v[:, 0] - ocx
            dy = v[:, 1] - ocy
            projs = dx * nx + dy * ny
            if np.any(np.max(projs, axis=1) > inradius + 1e-7):
                return False
        
        return True

    def generate_hex_cluster_config():
        """Generate a centered hexagonal cluster: 1 center + 6 ring1 + 5 ring2."""
        # For unit hexagons (side=1), the distance between centers of adjacent
        # hexagons in a honeycomb is sqrt(3) (the width of a hexagon)
        d = math.sqrt(3)
        
        centers = [np.array([0.0, 0.0])]
        
        # Ring 1: 6 hexagons around center
        for i in range(6):
            angle = i * math.pi / 3
            cx = d * math.cos(angle)
            cy = d * math.sin(angle)
            centers.append(np.array([cx, cy]))
        
        # Ring 2: Add 5 more (leave one gap for tighter packing)
        ring2_positions = []
        for i in range(6):
            angle = i * math.pi / 3
            # Position at distance 2*d along each direction
            cx = 2 * d * math.cos(angle)
            cy = 2 * d * math.sin(angle)
            ring2_positions.append(np.array([cx, cy]))
        
        # Also try intermediate positions
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            # Position between two ring1 hexagons, further out
            mid_angle = (angle1 + angle2) / 2
            cx = 1.5 * d * math.cos(mid_angle) * 2 / math.sqrt(3)
            cy = 1.5 * d * math.sin(mid_angle) * 2 / math.sqrt(3)
            ring2_positions.append(np.array([cx, cy]))
        
        # Select 5 from ring2_positions (try different combinations)
        # For now, take the first 5
        for i in range(5):
            centers.append(ring2_positions[i].copy())
        
        angles = [0.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_optimal_cluster():
        """Generate the theoretically optimal hexagonal cluster configuration."""
        # The optimal packing of 12 hexagons is a hexagonal number arrangement
        # with one gap. The key insight is that hexagons tile the plane,
        # so we can use the natural honeycomb arrangement.
        
        # Unit hexagon has width (flat-to-flat) = sqrt(3)
        # So adjacent centers are sqrt(3) apart
        
        d = math.sqrt(3)
        centers = []
        
        # Center hexagon
        centers.append(np.array([0.0, 0.0]))
        
        # 6 hexagons in first ring
        for i in range(6):
            angle = i * math.pi / 3
            centers.append(d * np.array([math.cos(angle), math.sin(angle)]))
        
        # 5 hexagons in second ring (one gap)
        # The second ring positions in a perfect hexagonal lattice:
        # Positions are at vectors: 2*e_i, e_i + e_{i+1} for i=0..5
        # where e_i are the 6 unit vectors at 60° apart
        ring2 = []
        for i in range(6):
            angle = i * math.pi / 3
            e = d * np.array([math.cos(angle), math.sin(angle)])
            next_angle = (i + 1) * math.pi / 3
            e_next = d * np.array([math.cos(next_angle), math.sin(next_angle)])
            
            # Two types of second-ring positions
            ring2.append(2 * e)
            ring2.append(e + e_next)
        
        # Remove duplicates and select 5
        unique_ring2 = []
        for p in ring2:
            is_dup = False
            for q in unique_ring2:
                if np.linalg.norm(p - q) < 1e-9:
                    is_dup = True
                    break
            if not is_dup:
                unique_ring2.append(p)
        
        # Sort by angle and select 5 (leave one gap)
        unique_ring2.sort(key=lambda p: math.atan2(p[1], p[0]))
        
        # Try leaving different gaps and pick the best
        best_centers = None
        best_side = float('inf')
        
        for gap_idx in range(len(unique_ring2)):
            selected = [p for idx, p in enumerate(unique_ring2) if idx != gap_idx]
            if len(selected) < 5:
                continue
            c = centers + selected[:5]
            angles = [0.0] * 12
            side = compute_outer_side(c, angles, 0.0, 0.0, 0.0)
            if side < best_side:
                best_side = side
                best_centers = c
        
        if best_centers is None:
            best_centers = centers + unique_ring2[:5]
        
        angles = [0.0] * 12
        return best_centers, angles, 0.0, 0.0, 0.0

    def generate_circular_config():
        """Generate a circular arrangement."""
        centers = [np.array([0.0, 0.0])]
        d = math.sqrt(3)
        
        # 6 in first ring
        for i in range(6):
            angle = i * math.pi / 3
            cx = d * math.cos(angle)
            cy = d * math.sin(angle)
            centers.append(np.array([cx, cy]))
        
        # 5 in second ring
        d2 = 2 * d
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 5
            cx = d2 * math.cos(angle)
            cy = d2 * math.sin(angle)
            centers.append(np.array([cx, cy]))
        
        angles = [0.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_compact_config():
        """Generate a compact 4x3 grid-like arrangement."""
        d = math.sqrt(3)
        h = d * math.sqrt(3) / 2  # vertical spacing for staggered rows
        centers = []
        
        # 3 rows of 4, staggered
        for row in range(3):
            y = row * h
            x_offset = (row % 2) * d / 2
            for col in range(4):
                centers.append(np.array([col * d + x_offset, y]))
        
        angles = [0.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def objective_and_constraints(params, outer_params):
        """Compute objective (outer side) and check constraints."""
        ocx, ocy, otheta = outer_params
        side = compute_outer_side(params['centers'], params['angles'], ocx, ocy, otheta)
        
        # Check validity
        valid = check_validity(params['centers'], params['angles'], ocx, ocy, otheta, side)
        
        if not valid:
            return side + 1000.0
        
        return side

    def optimize_single_config(centers_init, angles_init, ocx_init, ocy_init, otheta_init, time_budget):
        """Optimize a single configuration using scipy minimize."""
        
        def cost_func(v):
            if time.time() > deadline - 5:
                return 1e10
            
            # Parse parameters: 12 centers (24), 12 angles (12), outer (3) = 39 params
            centers = []
            angles = []
            idx = 0
            for i in range(12):
                cx, cy = v[idx], v[idx+1]
                centers.append(np.array([cx, cy]))
                idx += 2
            for i in range(12):
                angles.append(v[idx])
                idx += 1
            ocx, ocy, otheta = v[idx], v[idx+1], v[idx+2]
            
            # Compute outer side
            side = compute_outer_side(centers, angles, ocx, ocy, otheta)
            
            # Check for overlaps
            verts = []
            for i in range(12):
                v_hex = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                verts.append(v_hex)
            
            penalty = 0.0
            for i in range(12):
                for j in range(i + 1, 12):
                    dist = hexagon_distance(verts[i], verts[j])
                    if dist < 0:
                        penalty += (-dist) ** 2 * 100
            
            return side + penalty
        
        # Initial parameters
        v0 = []
        for c in centers_init:
            v0.extend([c[0], c[1]])
        v0.extend(angles_init)
        v0.extend([ocx_init, ocy_init, otheta_init])
        
        # Use Nelder-Mead for robustness
        result = minimize(cost_func, v0, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                  'adaptive': True})
        
        # Extract result
        v_opt = result.x
        centers_opt = []
        angles_opt = []
        idx = 0
        for i in range(12):
            centers_opt.append(np.array([v_opt[idx], v_opt[idx+1]]))
            idx += 2
        for i in range(12):
            angles_opt.append(v_opt[idx])
            idx += 1
        ocx_opt, ocy_opt, otheta_opt = v_opt[idx], v_opt[idx+1], v_opt[idx+2]
        
        return centers_opt, angles_opt, ocx_opt, ocy_opt, otheta_opt, result.fun

    def optimize_with_sa(centers_init, angles_init, ocx_init, ocy_init, otheta_init, time_budget):
        """Optimize using simulated annealing."""
        T = 0.5
        T_min = 0.001
        alpha = 0.99
        steps_per_temp = 10
        
        current_centers = [c.copy() for c in centers_init]
        current_angles = list(angles_init)
        current_ocx = ocx_init
        current_ocy = ocy_init
        current_otheta = otheta_init
        
        def calc_cost(centers, angles, ocx, ocy, otheta):
            side = compute_outer_side(centers, angles, ocx, ocy, otheta)
            
            verts = []
            for i in range(12):
                v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                verts.append(v)
            
            penalty = 0.0
            for i in range(12):
                for j in range(i + 1, 12):
                    dist = hexagon_distance(verts[i], verts[j])
                    if dist < 0:
                        penalty += (-dist) ** 2 * 100
            
            return side + penalty
        
        current_cost = calc_cost(current_centers, current_angles, current_ocx, current_ocy, current_otheta)
        
        best_centers = [c.copy() for c in current_centers]
        best_angles = list(current_angles)
        best_ocx = current_ocx
        best_ocy = current_ocy
        best_otheta = current_otheta
        best_cost = current_cost
        
        while T > T_min and time.time() < deadline - time_budget:
            for _ in range(steps_per_temp):
                if time.time() > deadline - time_budget:
                    break
                
                new_centers = [c.copy() for c in current_centers]
                new_angles = list(current_angles)
                new_ocx = current_ocx
                new_ocy = current_ocy
                new_otheta = current_otheta
                
                i = random.randint(0, 11)
                step_size = T * 0.2
                new_centers[i][0] += random.gauss(0, step_size)
                new_centers[i][1] += random.gauss(0, step_size)
                new_angles[i] += random.gauss(0, step_size * 2)
                
                if random.random() < 0.1:
                    new_ocx += random.gauss(0, T * 0.05)
                    new_ocy += random.gauss(0, T * 0.05)
                    new_otheta += random.gauss(0, T * 0.1)
                
                new_cost = calc_cost(new_centers, new_angles, new_ocx, new_ocy, new_otheta)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_centers = new_centers
                    current_angles = new_angles
                    current_ocx = new_ocx
                    current_ocy = new_ocy
                    current_otheta = new_otheta
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_centers = [c.copy() for c in current_centers]
                        best_angles = list(current_angles)
                        best_ocx = current_ocx
                        best_ocy = current_ocy
                        best_otheta = current_otheta
            
            T *= alpha
        
        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost

    # Generate multiple initial configurations
    configs = []
    
    # 1. Optimal hexagonal cluster
    c1, a1, ox1, oy1, oa1 = generate_optimal_cluster()
    configs.append((c1, a1, ox1, oy1, oa1))
    
    # 2. Hex cluster variant
    c2, a2, ox2, oy2, oa2 = generate_hex_cluster_config()
    configs.append((c2, a2, ox2, oy2, oa2))
    
    # 3. Circular arrangement
    c3, a3, ox3, oy3, oa3 = generate_circular_config()
    configs.append((c3, a3, ox3, oy3, oa3))
    
    # 4. Compact grid
    c4, a4, ox4, oy4, oa4 = generate_compact_config()
    configs.append((c4, a4, ox4, oy4, oa4))
    
    # 5. Perturbed optimal cluster
    c5 = [c1[i].copy() + np.random.normal(0, 0.05, 2) for i in range(12)]
    a5 = [a1[i] + np.random.normal(0, 0.1) for i in range(12)]
    configs.append((c5, a5, ox1, oy1, oa1))
    
    # Optimize each configuration
    best_overall_cost = float('inf')
    best_centers = None
    best_angles = None
    best_ocx = 0.0
    best_ocy = 0.0
    best_otheta = 0.0
    
    time_per_config = (deadline - time.time()) / len(configs) - 5
    
    for idx, (centers_init, angles_init, ocx_init, ocy_init, otheta_init) in enumerate(configs):
        if time.time() > deadline - 15:
            break
        
        remaining = min(time_per_config, deadline - time.time() - 10)
        
        # Use SA first, then polish with Nelder-Mead
        sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_with_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init, remaining * 0.6
        )
        
        # Polish with Nelder-Mead
        if time.time() < deadline - 10:
            nm_centers, nm_angles, nm_ocx, nm_ocy, nm_otheta, nm_cost = optimize_single_config(
                sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, remaining * 0.4
            )
            
            if nm_cost < best_overall_cost:
                best_overall_cost = nm_cost
                best_centers = nm_centers
                best_angles = nm_angles
                best_ocx = nm_ocx
                best_ocy = nm_ocy
                best_otheta = nm_otheta
        
        if sa_cost < best_overall_cost:
            best_overall_cost = sa_cost
            best_centers = sa_centers
            best_angles = sa_angles
            best_ocx = sa_ocx
            best_ocy = sa_ocy
            best_otheta = sa_otheta
    
    # Final polish: optimize only the outer hexagon parameters
    if best_centers is not None and time.time() < deadline - 3:
        def outer_obj(v):
            return compute_outer_side(best_centers, best_angles, v[0], v[1], v[2])
        
        result = minimize(outer_obj, [best_ocx, best_ocy, best_otheta], method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
        
        final_ocx, final_ocy, final_otheta = result.x
        final_side = result.fun
        
        # Verify validity
        if check_validity(best_centers, best_angles, final_ocx, final_ocy, final_otheta, final_side):
            best_ocx, best_ocy, best_otheta = final_ocx, final_ocy, final_otheta
            best_overall_cost = final_side
    
    # If we didn't find a valid configuration, use the theoretical optimum
    if best_centers is None:
        best_centers, best_angles, best_ocx, best_ocy, best_otheta = generate_optimal_cluster()
        best_overall_cost = compute_outer_side(best_centers, best_angles, best_ocx, best_ocy, best_otheta)
    
    # Final fine-tuning of outer hexagon
    if time.time() < deadline - 1:
        for da in np.linspace(-2, 2, 21):
            for dx in np.linspace(-0.01, 0.01, 11):
                for dy in np.linspace(-0.01, 0.01, 11):
                    s = compute_outer_side(best_centers, best_angles, 
                                          best_ocx + dx, best_ocy + dy, best_otheta + da)
                    if s < best_overall_cost:
                        best_overall_cost = s
                        best_ocx += dx
                        best_ocy += dy
                        best_otheta += da
    
    # Prepare output
    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_centers, best_angles)]
    outer_center = (float(best_ocx), float(best_ocy))
    outer_side = float(best_overall_cost)
    outer_angle = float(best_otheta)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
