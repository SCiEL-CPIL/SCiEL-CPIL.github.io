"""Combined program: smooth energy minimization with force-based gradient + simulated annealing."""
import numpy as np
import time

N = 12
DEADLINE = 145.0

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def hexagon_vertices_rad(cx, cy, side, angle_rad):
    """Vertices of a regular hexagon given center, side, angle in radians."""
    angles = angle_rad + np.radians(np.arange(0, 360, 60))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def sat_min_separation(c1, c2):
    """
    Compute minimum separation between two hexagons using SAT.
    Returns (min_sep, best_axis). Positive if separated, negative if overlapping.
    """
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    best_axis = None
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
            best_axis = axis
    
    return min_sep, best_axis

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
    
    return max_extent / (np.sqrt(3) / 2)

def calc_min_outer_side(all_verts, theta):
    """Calculate minimum outer hexagon side length for a given orientation theta (radians)."""
    angles = theta + np.radians(np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = all_verts @ normals.T
    widths = dots.max(axis=0) - dots.min(axis=0)
    max_width = np.max(widths)
    apothem = max_width / 2.0
    side = 2 * apothem / np.sqrt(3)
    return side

def find_best_outer_orientation(all_verts):
    """Find orientation and side length for minimum enclosing hexagon."""
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda th: calc_min_outer_side(all_verts, th), 
                          bounds=(0, np.pi/3), method='bounded', 
                          options={'xatol': 1e-6})
    return res.fun, res.x

def find_outer_center(all_verts, outer_angle_rad):
    """Find the center of the minimum enclosing regular hexagon."""
    angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = all_verts @ normals.T
    mins = dots.min(axis=0)
    maxs = dots.max(axis=0)
    mids = (mins + maxs) / 2.0
    c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
    return tuple(c)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep, _ = sat_min_separation(centers[i], centers[j])
            if sep < -1e-9:
                return False
    return True

def energy_smooth(centers, outer_center, outer_angle, beta=10.0):
    """
    Compute smooth energy using log-sum-exp for max containment and soft overlap.
    Returns (score, forces) where forces is list of (fx, fy, fa) for each inner hex.
    """
    n = len(centers)
    forces = np.zeros((n, 3))
    
    # Containment: compute all 6 face distances, use soft max
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    # Precompute outer hexagon normals
    outer_normals = []
    for k in range(6):
        theta = np.radians(60 * k)
        outer_normals.append((np.cos(theta), np.sin(theta)))
    
    # For each inner hex, compute max extent contribution
    # Use soft-max over all (vertex, face) pairs
    d_vals = []
    d_indices = []  # (hex_idx, vertex_idx, face_idx)
    
    for i in range(n):
        px, py, pa = centers[i]
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for vi, v in enumerate(inner_verts):
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k, (nx, ny) in enumerate(outer_normals):
                d = abs(rx * nx + ry * ny)
                d_vals.append(d)
                d_indices.append((i, vi, k))
    
    d_arr = np.array(d_vals)
    
    # Soft max using log-sum-exp
    beta_sm = 20.0
    max_d = np.max(d_arr)
    exp_vals = np.exp(beta_sm * (d_arr - max_d))
    log_sum_exp = max_d + np.log(np.sum(exp_vals)) / beta_sm
    req_side = log_sum_exp / (np.sqrt(3) / 2)
    
    # Overlap penalty: use soft min separation
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep, axis = sat_min_separation(centers[i], centers[j])
            if sep < 0:
                # Soft penalty: use exp-based smoothing
                penalty = np.exp(-sep * beta) - 1.0
                overlap_penalty += penalty
                # Gradient: d(penalty)/d(sep) = -beta * exp(-sep * beta)
                # Force on i: -axis * d(penalty)/d(sep) * (d(sep)/d(pos_i))
                # Force on j: +axis * d(penalty)/d(sep) * (d(sep)/d(pos_j))
                push_mag = beta * np.exp(-sep * beta)
                if axis is not None:
                    forces[i, 0] -= axis[0] * push_mag
                    forces[i, 1] -= axis[1] * push_mag
                    forces[j, 0] += axis[0] * push_mag
                    forces[j, 1] += axis[1] * push_mag
    
    # Containment forces: gradient of soft-max
    # d(soft_max)/d(d_i) = exp(beta_sm * (d_i - max_d)) / sum(exp(...))
    weights = exp_vals / np.sum(exp_vals)
    
    for idx, (i, vi, k) in enumerate(d_indices):
        w = weights[idx]
        if w < 1e-10:
            continue
        
        # Get the vertex and normal
        px, py, pa = centers[i]
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        v = inner_verts[vi]
        rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
        rx = rel[0] * cos_a - rel[1] * sin_a
        ry = rel[0] * sin_a + rel[1] * cos_a
        
        nx, ny = outer_normals[k]
        d_signed = rx * nx + ry * ny
        
        # Gradient of |d| w.r.t. (rx, ry)
        if d_signed > 0:
            grad_rx = nx
            grad_ry = ny
        else:
            grad_rx = -nx
            grad_ry = -ny
        
        # Rotate gradient back to world frame
        fx_world = grad_rx * np.cos(oa_rad) + grad_ry * np.sin(oa_rad)
        fy_world = -grad_rx * np.sin(oa_rad) + grad_ry * np.cos(oa_rad)
        
        # Scale by weight and convert to side units
        scale = w / (np.sqrt(3) / 2)
        forces[i, 0] += fx_world * scale
        forces[i, 1] += fy_world * scale
    
    score = req_side + 5.0 * overlap_penalty
    return score, forces

def initial_config_honeycomb():
    """Create a compact honeycomb initial configuration."""
    centers = []
    d = 2.1  # Center-to-center distance for non-overlapping unit hexagons
    
    # Center hexagon
    centers.append([0.0, 0.0, 0.0])
    
    # 6 surrounding
    for k in range(6):
        ang = np.radians(60 * k)
        x = d * np.cos(ang)
        y = d * np.sin(ang)
        centers.append([x, y, 0.0])
    
    # 5 more to make 12 (partial second ring)
    for k in range(5):
        ang = np.radians(60 * k)
        x = 2 * d * np.cos(ang)
        y = 2 * d * np.sin(ang)
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def force_based_local_search(centers, outer_center, outer_angle, max_iter=20000, 
                              step_scale=0.3, time_limit=100.0):
    """
    Local search using force-based gradient descent with momentum.
    """
    start_time = time.time()
    n = len(centers)
    
    # State
    pos = np.array([[c[0], c[1], c[2]] for c in centers], dtype=float)
    vel = np.zeros_like(pos)
    momentum = 0.85
    
    current_score, forces = energy_smooth(centers, outer_center, outer_angle)
    best_score = current_score
    best_pos = pos.copy()
    best_outer_center = list(outer_center)
    best_outer_angle = outer_angle
    
    step_size = step_scale
    min_step = 0.01
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Build force vector
        fvec = forces.copy()
        
        # Update velocity with momentum
        vel = momentum * vel + step_size * fvec
        
        # Clamp velocity
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        max_speed = 0.3
        mask = speed > max_speed
        vel[mask] = vel[mask] * (max_speed / speed[mask])
        
        # Update positions
        pos += vel
        
        # Wrap angles
        pos[:, 2] = pos[:, 2] % 60
        
        # Build centers list
        new_centers = [[pos[i, 0], pos[i, 1], pos[i, 2]] for i in range(n)]
        
        # Compute new energy
        new_score, new_forces = energy_smooth(new_centers, outer_center, outer_angle)
        
        if new_score < current_score:
            current_score = new_score
            forces = new_forces
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_pos = pos.copy()
        else:
            # Revert
            pos -= vel
            pos[:, 2] = pos[:, 2] % 60
            no_improve_count += 1
            # Reduce step size
            step_size = max(min_step, step_size * 0.95)
        
        # If stuck, try a small random perturbation
        if no_improve_count > 100:
            noise = np.random.normal(0, 0.1, pos.shape)
            pos += noise
            pos[:, 2] = pos[:, 2] % 60
            new_centers = [[pos[i, 0], pos[i, 1], pos[i, 2]] for i in range(n)]
            current_score, forces = energy_smooth(new_centers, outer_center, outer_angle)
            no_improve_count = 0
            step_size = max(min_step, step_size * 1.5)
        
        # Periodically recenter
        if it % 500 == 0 and it > 0:
            cx = np.mean(pos[:, 0])
            cy = np.mean(pos[:, 1])
            pos[:, 0] -= cx
            pos[:, 1] -= cy
            outer_center[0] -= cx
            outer_center[1] -= cy
    
    # Final validation
    final_centers = [[best_pos[i, 0], best_pos[i, 1], best_pos[i, 2]] for i in range(n)]
    
    # Check validity
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            sep, _ = sat_min_separation(final_centers[i], final_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Compute final outer parameters
    all_verts = []
    for c in final_centers:
        v = hexagon_vertices((c[0], c[1]), 1.0, c[2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
    outer_center_final = find_outer_center(all_verts, outer_angle_rad)
    
    return (final_centers, list(outer_center_final), outer_side, np.degrees(outer_angle_rad))

def simulated_annealing(init_centers, max_iter=30000, T_start=2.0, T_end=0.005,
                        cooling_rate=0.99997, time_limit=100.0):
    """
    Run simulated annealing to optimize the packing.
    """
    start_time = time.time()
    n = len(init_centers)
    
    centers = [list(c) for c in init_centers]
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    current_score, _ = energy_smooth(centers, outer_center, outer_angle)
    
    best_centers = [list(c) for c in centers]
    best_outer_center = list(outer_center)
    best_outer_angle = outer_angle
    best_score = current_score
    
    T = T_start
    no_improve = 0
    max_no_improve = 5000
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        new_centers = [list(c) for c in centers]
        new_outer_center = list(outer_center)
        new_outer_angle = outer_angle
        
        # Move 1-2 hexagons for better exploration
        num_moves = 1 if np.random.random() < 0.7 else 2
        for _ in range(num_moves):
            i = np.random.randint(n)
            step_size = T * 0.4
            new_centers[i][0] += np.random.normal(0, step_size)
            new_centers[i][1] += np.random.normal(0, step_size)
            new_centers[i][2] += np.random.normal(0, step_size * 25)
            new_centers[i][2] = new_centers[i][2] % 60
        
        if np.random.random() < 0.08:
            new_outer_center[0] += np.random.normal(0, T * 0.2)
            new_outer_center[1] += np.random.normal(0, T * 0.2)
        if np.random.random() < 0.04:
            new_outer_angle += np.random.normal(0, T * 8)
            new_outer_angle = new_outer_angle % 360
        
        new_score, _ = energy_smooth(new_centers, new_outer_center, new_outer_angle)
        
        delta = new_score - current_score
        if delta < 0 or (T > T_end and np.random.random() < np.exp(-delta / T)):
            centers = new_centers
            outer_center = new_outer_center
            outer_angle = new_outer_angle
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_centers = [list(c) for c in centers]
                best_outer_center = list(outer_center)
                best_outer_angle = outer_angle
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T = max(T_end, T * cooling_rate)
        
        # Restart if stuck
        if no_improve > max_no_improve and T > T_end:
            # Perturb current best
            centers = [list(c) for c in best_centers]
            outer_center = list(best_outer_center)
            outer_angle = best_outer_angle
            for c in centers:
                c[0] += np.random.normal(0, 0.5)
                c[1] += np.random.normal(0, 0.5)
                c[2] = (c[2] + np.random.normal(0, 10)) % 60
            current_score, _ = energy_smooth(centers, outer_center, outer_angle)
            T = max(T, T_start * 0.3)
            no_improve = 0
        
        if it % 1000 == 0 and it > 0:
            cx = np.mean([c[0] for c in centers])
            cy = np.mean([c[1] for c in centers])
            for c in centers:
                c[0] -= cx
                c[1] -= cy
            outer_center[0] -= cx
            outer_center[1] -= cy
    
    # Final validation
    valid = True
    for i in range(len(best_centers)):
        for j in range(i + 1, len(best_centers)):
            sep, _ = sat_min_separation(best_centers[i], best_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Compute final outer parameters
    all_verts = []
    for c in best_centers:
        v = hexagon_vertices((c[0], c[1]), 1.0, c[2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
    outer_center_final = find_outer_center(all_verts, outer_angle_rad)
    
    return (best_centers, list(outer_center_final), outer_side, np.degrees(outer_angle_rad))

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        ("honeycomb", initial_config_honeycomb()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("random3", initial_config_random(999)),
    ]
    
    # Phase 1: Simulated annealing for each config
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 25:
            break
        
        try:
            time_limit = min(18.0, DEADLINE - elapsed - 5)
            if time_limit < 2:
                break
            
            result = simulated_annealing(
                config,
                max_iter=25000,
                T_start=2.5,
                T_end=0.01,
                cooling_rate=0.99995,
                time_limit=time_limit
            )
            
            if result is not None:
                centers, outer_center, outer_side, outer_angle = result
                
                if check_validity(centers, tuple(outer_center), outer_side, outer_angle):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (centers, tuple(outer_center), outer_side, outer_angle)
        except Exception:
            pass
    
    # Phase 2: Force-based local search on best result with more iterations
    elapsed = time.time() - start_time
    if best_result is not None and elapsed < DEADLINE - 15:
        try:
            centers, outer_center, outer_side, outer_angle = best_result
            time_limit = min(35.0, DEADLINE - elapsed - 5)
            
            result = force_based_local_search(
                centers,
                outer_center,
                outer_angle,
                max_iter=50000,
                step_scale=0.25,
                time_limit=time_limit
            )
            
            if result is not None:
                c2, oc2, os2, oa2 = result
                if check_validity(c2, tuple(oc2), os2, oa2):
                    if os2 < best_score:
                        best_score = os2
                        best_result = (c2, tuple(oc2), os2, oa2)
        except Exception:
            pass
    
    # Phase 3: Try force-based local search from other configs
    for name, config in configs[:3]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        try:
            time_limit = min(12.0, DEADLINE - elapsed - 5)
            if time_limit < 2:
                break
            
            result = force_based_local_search(
                config,
                [0.0, 0.0],
                0.0,
                max_iter=30000,
                step_scale=0.3,
                time_limit=time_limit
            )
            
            if result is not None:
                c2, oc2, os2, oa2 = result
                if check_validity(c2, tuple(oc2), os2, oa2):
                    if os2 < best_score:
                        best_score = os2
                        best_result = (c2, tuple(oc2), os2, oa2)
        except Exception:
            pass
    
    # Phase 4: Final polish with very small steps
    elapsed = time.time() - start_time
    if best_result is not None and elapsed < DEADLINE - 5:
        try:
            centers, outer_center, outer_side, outer_angle = best_result
            time_limit = min(10.0, DEADLINE - elapsed - 2)
            
            result = force_based_local_search(
                centers,
                outer_center,
                outer_angle,
                max_iter=50000,
                step_scale=0.05,
                time_limit=time_limit
            )
            
            if result is not None:
                c2, oc2, os2, oa2 = result
                if check_validity(c2, tuple(oc2), os2, oa2):
                    if os2 < best_score:
                        best_score = os2
                        best_result = (c2, tuple(oc2), os2, oa2)
        except Exception:
            pass
    
    # Fallback
    if best_result is not None:
        return best_result
    
    # Default valid packing
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0
