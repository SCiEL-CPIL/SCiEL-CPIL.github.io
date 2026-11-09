"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # safety margin before 150s kill
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """Compute minimum outer side to contain all inner hexagons."""
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        
        # Pre-compute normals for the outer hexagon in local coordinates
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def check_overlap_sat(pos1, pos2, tol=1e-9):
        """Check if two hexagons overlap using Separating Axis Theorem. Returns True if overlapping."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.0:
            return False
        
        v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        normals = []
        for i in range(6):
            ex = v1[(i+1)%6][0] - v1[i][0]
            ey = v1[(i+1)%6][1] - v1[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
            
            ex = v2[(i+1)%6][0] - v2[i][0]
            ey = v2[(i+1)%6][1] - v2[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
        
        for nx, ny in normals:
            proj1 = [v[0]*nx + v[1]*ny for v in v1]
            proj2 = [v[0]*nx + v[1]*ny for v in v2]
            
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            if max1 <= min2 + tol or max2 <= min1 + tol:
                return False
        
        return True

    def has_overlap(positions, tol=1e-9):
        """Check if any pair of hexagons overlaps."""
        n = len(positions)
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_sat(positions[i], positions[j], tol):
                    return True
        return False

    def relax_positions(positions, max_iters=100, tol=1e-9):
        """Iteratively push overlapping hexagons apart using SAT to find minimum separation."""
        n = len(positions)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            any_overlap = False
            for i in range(n):
                for j in range(i+1, n):
                    if check_overlap_sat(positions[i], positions[j], tol):
                        any_overlap = True
                        # Find the minimum separation needed
                        v1 = hexagon_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
                        v2 = hexagon_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                        
                        normals = []
                        for k in range(6):
                            ex = v1[(k+1)%6][0] - v1[k][0]
                            ey = v1[(k+1)%6][1] - v1[k][1]
                            nx = -ey
                            ny = ex
                            length = math.sqrt(nx*nx + ny*ny)
                            if length > 1e-9:
                                normals.append((nx/length, ny/length))
                            ex = v2[(k+1)%6][0] - v2[k][0]
                            ey = v2[(k+1)%6][1] - v2[k][1]
                            nx = -ey
                            ny = ex
                            length = math.sqrt(nx*nx + ny*ny)
                            if length > 1e-9:
                                normals.append((nx/length, ny/length))
                        
                        # Find the axis with minimum overlap
                        min_overlap = float('inf')
                        best_axis = None
                        for nx, ny in normals:
                            proj1 = [v[0]*nx + v[1]*ny for v in v1]
                            proj2 = [v[0]*nx + v[1]*ny for v in v2]
                            min1, max1 = min(proj1), max(proj1)
                            min2, max2 = min(proj2), max(proj2)
                            overlap = min(max1, max2) - max(min1, min2)
                            if overlap > 0 and overlap < min_overlap:
                                min_overlap = overlap
                                best_axis = (nx, ny)
                        
                        if best_axis is not None:
                            nx, ny = best_axis
                            # Determine which direction to push
                            cx1 = sum(v[0] for v in v1) / 6
                            cy1 = sum(v[1] for v in v1) / 6
                            cx2 = sum(v[0] for v in v2) / 6
                            cy2 = sum(v[1] for v in v2) / 6
                            dot1 = (cx1 - cx2) * nx + (cy1 - cy2) * ny
                            if dot1 > 0:
                                positions[i][0] += (min_overlap/2 + 0.001) * nx
                                positions[i][1] += (min_overlap/2 + 0.001) * ny
                                positions[j][0] -= (min_overlap/2 + 0.001) * nx
                                positions[j][1] -= (min_overlap/2 + 0.001) * ny
                            else:
                                positions[i][0] -= (min_overlap/2 + 0.001) * nx
                                positions[i][1] -= (min_overlap/2 + 0.001) * ny
                                positions[j][0] += (min_overlap/2 + 0.001) * nx
                                positions[j][1] += (min_overlap/2 + 0.001) * ny
            if not any_overlap:
                break
        return positions

    def objective_smooth(params, outer_angle):
        """Smooth objective using log-sum-exp for the max projection."""
        n = len(params) // 3
        positions = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(n)]
        
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))
        
        # Collect all projections
        projections = []
        for ix, iy, iang in positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx
                dy = vy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    projections.append(proj)
        
        projections = np.array(projections)
        # Log-sum-exp for smooth max
        m = np.max(projections)
        if m < 0:
            m = 0
        lse = m + math.log(sum(math.exp(p - m) for p in projections))
        smooth_max = lse
        
        # Add overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_sat(positions[i], positions[j], -0.01):  # Allow slight overlap for smoothness
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    penalty += max(0, 2.0 - dist) ** 2
        
        return smooth_max / (sqrt3/2) + 10.0 * penalty

    def generate_hexagonal_lattice(n_hex):
        """Generate initial positions on a hexagonal lattice."""
        coords = [(0, 0)]
        # First ring
        for k in range(6):
            angle = math.radians(60 * k)
            coords.append((round(math.cos(angle)), round(math.sin(angle))))
        
        # Second ring
        for q in range(-2, 3):
            for r in range(-2, 3):
                if max(abs(q), abs(r), abs(q+r)) == 2:
                    coords.append((q, r))
        
        # Sort by distance from center
        def hex_dist(q, r):
            return abs(q) + abs(r) + abs(q+r)
        coords.sort(key=lambda x: hex_dist(x[0], x[1]))
        coords = coords[:n_hex]
        
        positions = []
        for q, r in coords:
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            positions.append([x, y, 0.0])
        return positions

    # Generate multiple initial configurations
    def get_initial_configurations():
        configs = []
        
        # Hexagonal lattice
        configs.append(generate_hexagonal_lattice(N))
        
        # Hexagonal lattice with rotation
        base = generate_hexagonal_lattice(N)
        for rot_angle in [15, 30, 45]:
            rotated = []
            for x, y, a in base:
                rad = math.radians(rot_angle)
                nx = x * math.cos(rad) - y * math.sin(rad)
                ny = x * math.sin(rad) + y * math.cos(rad)
                rotated.append([nx, ny, a + rot_angle])
            configs.append(rotated)
        
        # Random perturbations of lattice
        for _ in range(3):
            perturbed = []
            for x, y, a in base:
                perturbed.append([x + random.gauss(0, 0.3), y + random.gauss(0, 0.3), random.uniform(0, 60)])
            configs.append(perturbed)
        
        return configs

    best_side = float('inf')
    best_positions = None
    best_outer_angle = 0.0

    initial_configs = get_initial_configurations()
    
    for config_idx, init_pos in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Relax to remove overlaps
        relaxed = [p[:] for p in init_pos]
        relaxed = relax_positions(relaxed, max_iters=50)
        
        if has_overlap(relaxed):
            continue
        
        # Try different outer angles
        for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = compute_min_side(relaxed, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_positions = [p[:] for p in relaxed]
                best_outer_angle = outer_angle
        
        # Try Nelder-Mead optimization
        if time.time() > deadline - 20:
            break
        
        def nm_objective(params):
            if time.time() > deadline - 5:
                return 1e10
            n = len(params) // 3
            positions = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(n)]
            
            # Check overlap
            for i in range(n):
                for j in range(i+1, n):
                    if check_overlap_sat(positions[i], positions[j]):
                        return 1e10
            
            return compute_min_side(positions, (0.0, 0.0), best_outer_angle)
        
        x0 = []
        for x, y, a in relaxed:
            x0.extend([x, y, a])
        
        try:
            result = minimize(nm_objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < best_side:
                n = len(result.x) // 3
                new_positions = [(result.x[i*3], result.x[i*3+1], result.x[i*3+2]) for i in range(n)]
                if not has_overlap(new_positions):
                    best_side = result.fun
                    best_positions = new_positions
        except:
            pass

    # Fine-tune outer angle with the best positions
    if best_positions is not None:
        if time.time() > deadline - 10:
            pass
        else:
            final_side = float('inf')
            final_angle = 0.0
            for outer_angle in np.arange(0, 60, 1.0):
                if time.time() > deadline - 2:
                    break
                side = compute_min_side(best_positions, (0.0, 0.0), outer_angle)
                if side < final_side:
                    final_side = side
                    final_angle = outer_angle
            
            # Refine around best angle
            if final_angle > 0 and final_angle < 59:
                for outer_angle in np.arange(max(0, final_angle-2), min(60, final_angle+2), 0.1):
                    if time.time() > deadline - 1:
                        break
                    side = compute_min_side(best_positions, (0.0, 0.0), outer_angle)
                    if side < final_side:
                        final_side = side
                        final_angle = outer_angle
            
            best_side = final_side
            best_outer_angle = final_angle

    # Ensure best_positions is valid
    if best_positions is None or has_overlap(best_positions):
        # Fallback to a known valid configuration
        best_positions = generate_hexagonal_lattice(N)
        best_positions = relax_positions(best_positions, max_iters=100)
        best_side = compute_min_side(best_positions, (0.0, 0.0), 0.0)
        best_outer_angle = 0.0

    # Final relaxation to ensure no overlaps
    best_positions = relax_positions(best_positions, max_iters=20)
    
    # Recompute side after final relaxation
    final_side = float('inf')
    final_angle = 0.0
    for outer_angle in [best_outer_angle, 0.0, 30.0]:
        side = compute_min_side(best_positions, (0.0, 0.0), outer_angle)
        if side < final_side:
            final_side = side
            final_angle = outer_angle

    return best_positions, (0.0, 0.0), final_side, final_angle
# EVOLVE-BLOCK-END
