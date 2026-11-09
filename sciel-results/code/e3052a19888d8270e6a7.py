import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator limit 150s)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Get axes for SAT (normals to edges)
    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_poly(vertices, axis):
        nx, ny = axis
        min_val = float('inf')
        max_val = float('-inf')
        for vx, vy in vertices:
            proj = vx * nx + vy * ny
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return min_val, max_val

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2, d_sq):
        """
        Fast disjoint check.
        1. If d_sq > 4.0 (d > 2.0), definitely disjoint.
        2. If d_sq < 3.0 (d < sqrt(3)), definitely overlap (min distance for disjoint is sqrt(3)).
        3. Otherwise, use SAT.
        """
        if d_sq > 4.0:
            return True
        if d_sq < 3.0:
            return False
            
        # SAT check
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all given hexagons.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # Generate candidate positions on a fine triangular lattice
    def generate_candidates(radius=4.0, spacing=0.1):
        """Generate candidate (x, y, angle) tuples on a triangular lattice."""
        candidates = []
        # Triangular lattice points within radius
        # Use axial coordinates for triangular lattice
        # For a triangular lattice with spacing s, the basis vectors are:
        # v1 = (s, 0), v2 = (s/2, s*sqrt(3)/2)
        # We'll generate points within a hexagonal region of radius 'radius'
        
        # Simpler: generate grid points and filter
        n_steps = int(radius / spacing) + 2
        angles = [0.0, 15.0, 30.0, 45.0, 60.0]
        
        for i in range(-n_steps, n_steps + 1):
            for j in range(-n_steps, n_steps + 1):
                # Convert grid to triangular lattice coordinates
                x = spacing * (i + 0.5 * j)
                y = spacing * (SQRT3 / 2) * j
                
                # Check if within radius
                if x * x + y * y > radius * radius:
                    continue
                
                for a in angles:
                    candidates.append((x, y, a))
        
        return candidates

    # Backtracking search with constraint propagation
    def backtracking_search(candidates, max_depth=11, time_limit=None):
        """
        Find N non-overlapping hexagons from candidates minimizing outer side.
        Uses backtracking with pruning.
        """
        if time_limit is None:
            time_limit = deadline - time.time()
        
        best_solution = None
        best_score = float('inf')
        current = []
        current_hexagons = []
        current_axes = []
        
        # Pre-compute vertices and axes for all candidates to avoid recomputation
        cand_verts = []
        cand_axes = []
        for x, y, a in candidates:
            verts = get_hex_vertices(x, y, a)
            axes = get_hex_axes(a)
            cand_verts.append(verts)
            cand_axes.append(axes)
        
        # Sort candidates by some heuristic (e.g., distance from origin) to find compact solutions first
        cand_indices = list(range(len(candidates)))
        cand_indices.sort(key=lambda i: candidates[i][0]**2 + candidates[i][1]**2)
        
        # Limit candidate pool for tractability
        # We can't use all candidates if there are too many
        if len(cand_indices) > 500:
            # Keep the most central 500
            cand_indices = cand_indices[:500]
        
        def backtrack(depth, start_idx):
            nonlocal best_solution, best_score
            
            if time.time() > deadline - 5.0:
                return
            
            if depth == N:
                # Check validity and compute score
                hexagons = [current_hexagons[i] for i in range(N)]
                side = calc_outer_side(hexagons, 0.0)
                if side < best_score:
                    best_score = side
                    best_solution = [current[i] for i in range(N)]
                return
            
            # Pruning: if even the best possible remaining can't beat best_score, skip
            # (This is hard to compute exactly, so we use a heuristic)
            
            for idx in range(start_idx, len(cand_indices)):
                if time.time() > deadline - 5.0:
                    return
                
                ci = cand_indices[idx]
                x, y, a = candidates[ci]
                verts = cand_verts[ci]
                axes = cand_axes[ci]
                
                # Check against all currently placed hexagons
                valid = True
                for placed_idx in range(len(current)):
                    dx = x - current[placed_idx][0]
                    dy = y - current[placed_idx][1]
                    d_sq = dx * dx + dy * dy
                    if not is_disjoint(verts, current_hexagons[placed_idx], axes, current_axes[placed_idx], d_sq):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Place this hexagon
                current.append(candidates[ci])
                current_hexagons.append(verts)
                current_axes.append(axes)
                
                backtrack(depth + 1, idx + 1)
                
                # Remove
                current.pop()
                current_hexagons.pop()
                current_axes.pop()
        
        backtrack(0, 0)
        
        return best_solution, best_score

    # Generate candidates
    candidates = generate_candidates(radius=4.0, spacing=0.2)
    
    # Shuffle to add randomness and avoid getting stuck
    random.shuffle(candidates)
    
    # Run backtracking search
    solution, score = backtracking_search(candidates)
    
    # If backtracking didn't find a solution, fall back to SA on rigid cluster
    if solution is None or score >= float('inf'):
        # Fallback: use the rigid cluster SA approach from the original
        def axial_to_cartesian(q, r, spacing=1.0):
            x = SQRT3 * spacing * (q + r / 2.0)
            y = 1.5 * spacing * r
            return x, y
        
        coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
        
        base_centers = []
        base_angles = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            base_centers.append((x, y))
            base_angles.append(0.0)
        
        cx_avg = sum(c[0] for c in base_centers) / N
        cy_avg = sum(c[1] for c in base_centers) / N
        base_centers = [(x - cx_avg, y - cy_avg) for x, y in base_centers]
        
        def evaluate_rigid(base_centers, base_angles, tx, ty, theta_deg, scale):
            theta_rad = math.radians(theta_deg)
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            
            hexagons = []
            hex_axes = []
            centers = []
            
            for i in range(N):
                bx, by = base_centers[i]
                rx = bx * scale
                ry = by * scale
                x = rx * cos_t - ry * sin_t + tx
                y = rx * sin_t + ry * cos_t + ty
                a = base_angles[i] + theta_deg
                centers.append((x, y))
                hex_verts = get_hex_vertices(x, y, a)
                hexagons.append(hex_verts)
                hex_axes.append(get_hex_axes(a))
            
            cx_avg = sum(c[0] for c in centers) / N
            cy_avg = sum(c[1] for c in centers) / N
            shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
            
            hexagons_final = []
            hex_axes_final = []
            for i in range(N):
                cx, cy = shifted_centers[i]
                a = base_angles[i] + theta_deg
                hex_verts = get_hex_vertices(cx, cy, a)
                hexagons_final.append(hex_verts)
                hex_axes_final.append(get_hex_axes(a))
            
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = shifted_centers[i][0] - shifted_centers[j][0]
                    dy = shifted_centers[i][1] - shifted_centers[j][1]
                    d_sq = dx*dx + dy*dy
                    if not is_disjoint(hexagons_final[i], hexagons_final[j], hex_axes_final[i], hex_axes_final[j], d_sq):
                        overlap_penalty += 1000.0
            
            side = calc_outer_side(hexagons_final, 0.0)
            return overlap_penalty + side
        
        rigid_params = np.array([0.0, 0.0, 0.0, 1.0])
        current_score = evaluate_rigid(base_centers, base_angles, *rigid_params)
        local_best_params = rigid_params.copy()
        local_best_score = current_score
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.995
        
        while T > T_min and time.time() < deadline - 10.0:
            new_params = rigid_params.copy()
            new_params[0] += random.gauss(0, 0.1)
            new_params[1] += random.gauss(0, 0.1)
            new_params[2] += random.gauss(0, 5.0)
            new_params[3] += random.gauss(0, 0.05)
            new_params[3] = max(0.8, min(1.2, new_params[3]))
            
            new_score = evaluate_rigid(base_centers, base_angles, *new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                rigid_params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = rigid_params.copy()
            
            T *= cooling_rate
        
        # Convert to solution
        tx, ty, theta_deg, scale = local_best_params
        theta_rad = math.radians(theta_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        
        solution = []
        for i in range(N):
            bx, by = base_centers[i]
            rx = bx * scale
            ry = by * scale
            x = rx * cos_t - ry * sin_t + ty * 0 + tx  # ty is 0 in rigid, but we center later
            y = rx * sin_t + ry * cos_t + ty
            a = base_angles[i] + theta_deg
            solution.append((x, y, a))
        
        score = local_best_score

    # If still no valid solution, use fallback
    if solution is None:
        def axial_to_cartesian(q, r, spacing=1.0):
            x = SQRT3 * spacing * (q + r / 2.0)
            y = 1.5 * spacing * r
            return x, y
        
        coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Center the solution
    centers = [(s[0], s[1]) for s in solution]
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_solution = [(x - cx_avg, y - cy_avg, a) for x, y, a in solution]
    
    # Validate
    hexagons = []
    hex_axes = []
    for x, y, a in shifted_solution:
        hex_verts = get_hex_vertices(x, y, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
    
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = shifted_solution[i][0] - shifted_solution[j][0]
            dy = shifted_solution[i][1] - shifted_solution[j][1]
            d_sq = dx*dx + dy*dy
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to simple configuration
        def axial_to_cartesian(q, r, spacing=1.0):
            x = SQRT3 * spacing * (q + r / 2.0)
            y = 1.5 * spacing * r
            return x, y
        
        coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Optimize outer angle
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
    
    inner_list = []
    for x, y, a in shifted_solution:
        inner_list.append([x, y, a])
    
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
