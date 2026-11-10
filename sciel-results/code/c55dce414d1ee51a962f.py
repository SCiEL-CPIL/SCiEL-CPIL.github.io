import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _compute_min_s(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s such that all inner hexagon vertices 
    are inside the outer hexagon with given center and angle.
    """
    outer_center = np.asarray(outer_center, dtype=float)
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= outer_center[0]
    inner_shifted[:, 1] -= outer_center[1]
    
    a_outer = np.deg2rad(outer_angle_deg)
    normal_angles = a_outer + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_s = 0.0
    for row in inner_shifted:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for n in normals:
                val = np.dot(n, v)
                required_s = val / (np.sqrt(3.0) / 2.0)
                if required_s > max_s:
                    max_s = required_s
    return max_s


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"
    
    # Outer hexagon vertices must be distinct
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    # Check unit edge length for inner hexagons
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"unit edge length differs: {edge_len}"
    
    # Check inner-inner separation using SAT
    def _separated(v1, v2):
        """Check if two hexagons are separated (non-overlapping) using SAT."""
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True
        
        return False
    
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner hexagons {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, f"inner vertex outside outer hexagon"
    
    return True, ""


def _is_valid_packing(inner, outer, s):
    """Quick validity check without expensive SAT for all pairs."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12.0) - 1e-12:
        return False
    
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False
    
    # Check inner vertices inside outer hexagon (fast check)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False
    
    # Quick center distance check for non-overlap (necessary but not sufficient)
    centers = inner[:, :2]
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 3.0 - 1e-6:
                return False
    
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: Use multiple hexagonal lattice configurations with different topologies,
    optimize rotation, scale, and outer hexagon angle for each, and keep the best result.
    Also try local perturbation of individual hexagon positions for further improvement.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    sqrt3 = np.sqrt(3.0)
    
    # Define multiple lattice configurations for 12 hexagons
    # Lattice vectors: e1 = (3, 0), e2 = (1.5, 3*sqrt(3)/2)
    e1 = np.array([3.0, 0.0])
    e2 = np.array([1.5, 3.0 * np.sqrt(3.0) / 2.0])
    lattice_basis = np.array([e1, e2]).T  # 2x2 matrix
    
    # Generate diverse configurations
    configs = []
    
    # Config 1: Center + 6 neighbors + 5 more (original)
    configs.append(np.array([
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [1, -1], [-1, 1], [-1, -1],
        [2, 0], [2, 1], [2, -1]
    ], dtype=float))
    
    # Config 2: More symmetric
    configs.append(np.array([
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [1, -1], [-1, 1], [-1, -1],
        [0, 2], [2, 0], [-2, 0]
    ], dtype=float))
    
    # Config 3: Compact cluster (3x4 rectangle)
    configs.append(np.array([
        [0, 0], [1, 0], [2, 0], [-1, 0],
        [0, 1], [1, 1], [2, 1], [-1, 1],
        [0, -1], [1, -1], [2, -1], [-1, -1]
    ], dtype=float))
    
    # Config 4: Diamond shape
    configs.append(np.array([
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, -1], [-1, 1], [1, -1],
        [2, 0], [-2, 0], [0, 2]
    ], dtype=float))
    
    # Config 5: Another arrangement
    configs.append(np.array([
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [1, -1], [-1, 1], [-1, -1],
        [2, 1], [-2, -1], [0, 2]
    ], dtype=float))
    
    # Config 6: Hexagonal arrangement with 3 rings
    configs.append(np.array([
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, -1], [-1, 1], [2, 0], [-2, 0],
        [0, 2], [0, -2], [2, 2]
    ], dtype=float))
    
    # Config 7: Elongated
    configs.append(np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [-1, 0], [0, 1], [1, 1], [2, 1],
        [3, 1], [-1, 1], [0, -1], [1, -1]
    ], dtype=float))
    
    # Config 8: Star pattern
    configs.append(np.array([
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, -1], [2, 1], [-2, -1],
        [2, -1], [-2, 1], [1, -1]
    ], dtype=float))
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    for cfg_idx, lattice_points in enumerate(configs):
        if time.time() > deadline - 40:
            break
        
        # Convert to Cartesian
        base_positions = lattice_points @ lattice_basis
        
        # Center at origin
        centroid = np.mean(base_positions, axis=0)
        base_positions -= centroid
        
        # Try multiple initial outer angles
        outer_angles_to_try = [0.0, 30.0, 60.0, 90.0]
        
        for outer_angle_deg in outer_angles_to_try:
            if time.time() > deadline - 20:
                break
            
            # Initial configuration
            inner_init = np.zeros((12, 3))
            inner_init[:, 0] = base_positions[:, 0]
            inner_init[:, 1] = base_positions[:, 1]
            inner_init[:, 2] = 0.0
            
            outer_init = np.array([0.0, 0.0, outer_angle_deg])
            s_init = _compute_min_s(inner_init, np.array([0.0, 0.0]), outer_angle_deg)
            
            # Verify validity
            valid, msg = _check_validity(inner_init, outer_init, s_init + 1e-4)
            if not valid:
                continue
            
            if s_init + 1e-4 < best_s:
                best_s = s_init + 1e-4
                best_inner = inner_init.copy()
                best_outer = outer_init.copy()
        
        # Optimize using Nelder-Mead: parameters are (rotation, log_scale, outer_angle)
        def objective(params, bp=base_positions):
            theta = params[0]
            log_t = params[1]
            outer_angle = params[2]
            
            t = np.exp(log_t)
            c, s_rot = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s_rot], [s_rot, c]])
            positions = (bp * t) @ R.T
            
            inner = np.zeros((12, 3))
            inner[:, 0] = positions[:, 0]
            inner[:, 1] = positions[:, 1]
            inner[:, 2] = theta * 180.0 / np.pi
            
            s_val = _compute_min_s(inner, np.array([0.0, 0.0]), outer_angle * 180.0 / np.pi)
            return s_val
        
        try:
            x0 = np.array([0.0, 0.0, 0.0])
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                      'adaptive': True})
            
            if result.fun < best_s - 1e-8:
                theta_opt = result.x[0]
                t_opt = np.exp(result.x[1])
                outer_angle_opt = result.x[2]
                c, s_rot = np.cos(theta_opt), np.sin(theta_opt)
                R = np.array([[c, -s_rot], [s_rot, c]])
                positions = (base_positions * t_opt) @ R.T
                
                inner_opt = np.zeros((12, 3))
                inner_opt[:, 0] = positions[:, 0]
                inner_opt[:, 1] = positions[:, 1]
                inner_opt[:, 2] = theta_opt * 180.0 / np.pi
                
                s_opt = _compute_min_s(inner_opt, np.array([0.0, 0.0]), outer_angle_opt * 180.0 / np.pi)
                outer_opt = np.array([0.0, 0.0, outer_angle_opt * 180.0 / np.pi])
                
                valid, msg = _check_validity(inner_opt, outer_opt, s_opt + 1e-4)
                if valid and s_opt + 1e-4 < best_s:
                    best_s = s_opt + 1e-4
                    best_inner = inner_opt.copy()
                    best_outer = outer_opt.copy()
                    print(f"Config {cfg_idx} optimized: s = {best_s:.6f}")
                    
        except Exception:
            pass
        
        # Try local perturbation of individual positions
        if best_inner is not None and time.time() < deadline - 60:
            # Start from best configuration found so far
            current_inner = best_inner.copy()
            current_outer = best_outer.copy()
            current_s = best_s
            
            # Perturb each hexagon slightly
            for iter_count in range(30):
                if time.time() > deadline - 30:
                    break
                
                # Try moving each hexagon in a small random direction
                improved = False
                for h_idx in range(12):
                    for trial in range(5):
                        dx = np.random.uniform(-0.3, 0.3)
                        dy = np.random.uniform(-0.3, 0.3)
                        
                        trial_inner = current_inner.copy()
                        trial_inner[h_idx, 0] += dx
                        trial_inner[h_idx, 1] += dy
                        
                        s_trial = _compute_min_s(trial_inner, current_outer[:2], current_outer[2])
                        
                        if s_trial + 1e-4 < current_s - 1e-8:
                            valid, _ = _check_validity(trial_inner, current_outer, s_trial + 1e-4)
                            if valid:
                                current_inner = trial_inner
                                current_s = s_trial + 1e-4
                                improved = True
                                if current_s < best_s:
                                    best_s = current_s
                                    best_inner = current_inner.copy()
                                    best_outer = current_outer.copy()
                
                if not improved:
                    break
    
    # Final validation
    if best_inner is None:
        # Fallback: use a simple valid configuration
        lattice_points = configs[0]
        base_positions = lattice_points @ lattice_basis
        centroid = np.mean(base_positions, axis=0)
        base_positions -= centroid
        
        inner_fb = np.zeros((12, 3))
        inner_fb[:, 0] = base_positions[:, 0]
        inner_fb[:, 1] = base_positions[:, 1]
        inner_fb[:, 2] = 0.0
        
        outer_fb = np.array([0.0, 0.0, 0.0])
        s_fb = _compute_min_s(inner_fb, np.array([0.0, 0.0]), 0.0) + 1e-4
        
        return inner_fb, outer_fb, s_fb
    
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        print(f"Final result invalid: {msg}, using fallback")
        # Use a safe fallback
        lattice_points = configs[0]
        base_positions = lattice_points @ lattice_basis
        centroid = np.mean(base_positions, axis=0)
        base_positions -= centroid
        
        inner_fb = np.zeros((12, 3))
        inner_fb[:, 0] = base_positions[:, 0]
        inner_fb[:, 1] = base_positions[:, 1]
        inner_fb[:, 2] = 0.0
        
        outer_fb = np.array([0.0, 0.0, 0.0])
        s_fb = _compute_min_s(inner_fb, np.array([0.0, 0.0]), 0.0) + 1e-4
        
        return inner_fb, outer_fb, s_fb
    
    print(f"Best result: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    return best_inner, best_outer, best_s
