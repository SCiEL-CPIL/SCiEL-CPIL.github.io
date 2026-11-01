import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (pointy-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_normals(verts):
        """Get outward normals for a convex polygon."""
        normals = []
        for i in range(len(verts)):
            edge = verts[(i + 1) % len(verts)] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normals.append(normal / norm)
        return normals

    def penetration_depth(v1, v2):
        """Calculate minimum penetration depth between two convex polygons."""
        normals = get_normals(v1) + get_normals(v2)
        min_overlap = np.inf
        for normal in normals:
            p1 = v1 @ normal
            p2 = v2 @ normal
            overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            if overlap < min_overlap:
                min_overlap = overlap
        return min_overlap

    def polygons_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for v in [v1, v2]:
            for i in range(len(v)):
                edge = v[(i + 1) % len(v)] - v[i]
                normal = np.array([-edge[1], edge[0]])
                if np.linalg.norm(normal) > 1e-12:
                    normal /= np.linalg.norm(normal)
                    axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        
        return False

    def hexagons_disjoint(positions, angles):
        """Check if all hexagons are pairwise disjoint."""
        n = len(positions)
        verts = [hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not polygons_disjoint_sat(verts[i], verts[j]):
                    return False
        return True

    def calc_min_outer_side_fast(all_verts, theta):
        """Calculate minimum outer hexagon side length for a given orientation theta."""
        angles = theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        max_width = np.max(widths)
        apothem = max_width / 2.0
        side = 2 * apothem / np.sqrt(3)
        return side

    def find_best_outer_orientation_fast(all_verts):
        """Find orientation and side length for minimum enclosing hexagon."""
        # Coarse search
        thetas_coarse = np.linspace(0, np.pi/3, 120)
        sides = np.array([calc_min_outer_side_fast(all_verts, th) for th in thetas_coarse])
        best_idx = np.argmin(sides)
        
        # Fine search around best
        lo = thetas_coarse[max(0, best_idx - 3)]
        hi = thetas_coarse[min(len(thetas_coarse)-1, best_idx + 3)]
        
        # Use minimize_scalar on a narrow window
        res = minimize_scalar(lambda th: calc_min_outer_side_fast(all_verts, th), 
                              bounds=(lo, hi), method='bounded', 
                              options={'xatol': 1e-7})
        return res.fun, res.x

    def inside_violation(inner_verts, outer_verts):
        """Calculate how much inner polygon violates outer polygon boundary."""
        outer_normals = get_normals(outer_verts)
        max_viol = 0.0
        for normal in outer_normals:
            # normal points outward from outer
            inner_max = (inner_verts @ normal).max()
            outer_max = (outer_verts @ normal).max()
            viol = inner_max - outer_max
            if viol > max_viol:
                max_viol = viol
        return max_viol

    def full_objective(params):
        """Objective: outer hexagon side length, with penalty for overlap and containment."""
        if time.time() > deadline:
            return 1e10
            
        n = len(params) // 3
        positions = np.array([[params[i*3], params[i*3+1]] for i in range(n)])
        angles = np.array([params[i*3+2] for i in range(n)])
        
        verts = [hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]) for i in range(n)]
        all_verts = np.vstack(verts)
        
        # Calculate optimal outer hexagon
        outer_side, outer_angle = find_best_outer_orientation_fast(all_verts)
        
        # Check if inner hexagons fit inside the computed outer hexagon
        outer_verts = hexagon_vertices(0, 0, outer_side, outer_angle)
        
        # Penalty for inner hexagons not being inside outer
        total_penalty = 0.0
        for i in range(n):
            viol = inside_violation(verts[i], outer_verts)
            total_penalty += max(0.0, viol)
        
        # Penalty for overlaps between inner hexagons
        for i in range(n):
            for j in range(i+1, n):
                pd = penetration_depth(verts[i], verts[j])
                total_penalty += max(0.0, pd)
        
        return outer_side + 10.0 * total_penalty

    # Generate diverse initial configurations
    def make_hex_config(d, rot=0.0):
        """Create a honeycomb-like configuration with spacing d."""
        pos = []
        ang = []
        pos.append([0.0, 0.0])
        ang.append(rot)
        for k in range(6):
            a = np.radians(60 * k) + rot
            pos.append([d * np.cos(a), d * np.sin(a)])
            ang.append(rot)
        for k in range(5):
            a = np.radians(60 * k) + rot
            pos.append([2 * d * np.cos(a), 2 * d * np.sin(a)])
            ang.append(rot)
        return np.array(pos), np.array(ang)

    def make_staggered_config(d, rot=0.0):
        """Create a staggered row configuration."""
        pos = []
        ang = []
        for row in range(3):
            for col in range(4):
                x = col * d
                y = row * d * np.sqrt(3)
                if row % 2 == 1:
                    x += d / 2
                pos.append([x, y])
                ang.append(rot)
        return np.array(pos), np.array(ang)

    def make_circular_config(r, rot=0.0):
        """Create a circular arrangement."""
        pos = []
        ang = []
        pos.append([0.0, 0.0])
        ang.append(rot)
        for k in range(5):
            a = np.radians(72 * k) + rot
            pos.append([r * np.cos(a), r * np.sin(a)])
            ang.append(rot)
        for k in range(6):
            a = np.radians(60 * k) + rot
            pos.append([2 * r * np.cos(a), 2 * r * np.sin(a)])
            ang.append(rot)
        return np.array(pos), np.array(ang)

    def make_two_rows_config(d, rot=0.0):
        """Create two rows of 6."""
        pos = []
        ang = []
        for row in range(2):
            for col in range(6):
                x = col * d
                y = row * d * np.sqrt(3)
                pos.append([x, y])
                ang.append(rot)
        return np.array(pos), np.array(ang)

    # Try multiple initial configurations
    configs = []
    configs.append(make_hex_config(2.0))
    configs.append(make_hex_config(2.0, rot=np.pi/6))
    configs.append(make_hex_config(2.1))
    configs.append(make_hex_config(2.1, rot=np.pi/3))
    configs.append(make_staggered_config(2.0))
    configs.append(make_staggered_config(2.1))
    configs.append(make_circular_config(1.1))
    configs.append(make_circular_config(1.2))
    configs.append(make_two_rows_config(2.0))
    configs.append(make_two_rows_config(2.1))

    best_obj = 1e10
    best_params = None

    for pos, ang in configs:
        if time.time() > deadline - 30:
            break
        params = np.concatenate([pos.flatten(), ang])
        obj = full_objective(params)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()

    # Local refinement with L-BFGS-B from best initial config
    if best_params is not None:
        bounds = []
        for i in range(12):
            bounds.append((-5.0, 5.0))
            bounds.append((-5.0, 5.0))
            bounds.append((0.0, np.pi))

        # Use L-BFGS-B for local optimization
        try:
            result = minimize(
                full_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

        # Try Nelder-Mead for further refinement
        try:
            result2 = minimize(
                full_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-8, 'xatol': 1e-6}
            )
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass

        # Try differential evolution with smaller population for more thorough search
        if time.time() < deadline - 40:
            try:
                de_result = differential_evolution(
                    full_objective,
                    bounds=bounds,
                    seed=42,
                    maxiter=15,
                    popsize=8,
                    tol=1e-3,
                    polish=False
                )
                if de_result.fun < best_obj:
                    best_obj = de_result.fun
                    best_params = de_result.x.copy()
            except Exception:
                pass

    # Extract best solution
    if best_params is not None and len(best_params) == 36:
        best_positions = np.array([[best_params[i*3], best_params[i*3+1]] for i in range(12)])
        best_angles = np.array([best_params[i*3+2] for i in range(12)])
        
        all_verts = []
        for i in range(12):
            v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        outer_side, outer_angle_rad = find_best_outer_orientation_fast(all_verts)
        angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        outer_center = tuple(c)
        outer_angle_deg = np.degrees(outer_angle_rad)
    else:
        outer_center = (0.0, 0.0)
        outer_side = 10.0
        outer_angle_deg = 0.0
        pos, ang = make_hex_config(2.5)
        best_positions = pos
        best_angles = ang

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(12)]
    
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END
