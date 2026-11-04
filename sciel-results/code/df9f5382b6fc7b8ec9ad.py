import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    """Generate vertices of a unit hexagon centered at (x,y) with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Normals of outer hexagon (centered at 0,0, rotation 0)
    # Faces are perpendicular to vectors at angles 0, 60, ...
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Unit hexagon vertices centered at origin, angle 0
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    def get_hex_verts(params, idx):
        """Get vertices of hexagon at index idx from params array."""
        x, y, ang = params[idx*3], params[idx*3+1], params[idx*3+2]
        angle_rad = np.deg2rad(ang)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array([x, y])
        return verts

    def minkowski_separation(verts_a, verts_b):
        """Check if two convex polygons overlap using separating axis theorem.
        Returns 0 if separated, or the minimum penetration depth if overlapping."""
        n_a = len(verts_a)
        n_b = len(verts_b)
        
        min_sep = float('inf')
        
        # Check edge normals from polygon A
        for i in range(n_a):
            e = verts_a[(i+1) % n_a] - verts_a[i]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            n = n / norm
            
            proj_a = np.dot(verts_a, n)
            proj_b = np.dot(verts_b, n)
            
            min_a, max_a = proj_a.min(), proj_a.max()
            min_b, max_b = proj_b.min(), proj_b.max()
            
            if max_a <= min_b or max_b <= min_a:
                return 0.0  # Separated
            
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < min_sep:
                min_sep = overlap
        
        # Check edge normals from polygon B
        for i in range(n_b):
            e = verts_b[(i+1) % n_b] - verts_b[i]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            n = n / norm
            
            proj_a = np.dot(verts_a, n)
            proj_b = np.dot(verts_b, n)
            
            min_a, max_a = proj_a.min(), proj_a.max()
            min_b, max_b = proj_b.min(), proj_b.max()
            
            if max_a <= min_b or max_b <= min_a:
                return 0.0  # Separated
            
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < min_sep:
                min_sep = overlap
        
        return min_sep

    def calc_side_and_penalty(params):
        """Calculate outer hexagon side length and overlap penalty."""
        # 1. Calculate required outer hexagon side
        max_apothem = 0.0
        verts_list = []
        for k in range(N):
            verts = get_hex_verts(params, k)
            verts_list.append(verts)
            # Project all vertices onto all normals
            for v in verts:
                for n in normals:
                    proj = np.dot(v, n)
                    if proj > max_apothem:
                        max_apothem = proj
        
        # Side S = 2 * apothem / sqrt(3)
        side = 2.0 * max_apothem / np.sqrt(3)
        
        # 2. Calculate overlap penalty using SAT
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = minkowski_separation(verts_list[i], verts_list[j])
                if sep < 1e-6:
                    penalty += 1.0
        
        return side, penalty

    def objective(params):
        side, penalty = calc_side_and_penalty(params)
        return side + penalty * 10.0

    # Generate multiple initial configurations
    d = np.sqrt(3)
    
    def make_config_1():
        """Hexagonal cluster of 7 + 4 more"""
        centers = [
            (0, 0),
            (d, 0), (d/2, 1.5), (-d/2, 1.5),
            (-d, 0), (-d/2, -1.5), (d/2, -1.5),
            (0, 3.0), (2.6, 1.5), (2.6, -1.5), (0, -3.0)
        ]
        params = []
        for x, y in centers:
            params.extend([x, y, 0.0])
        return np.array(params)

    def make_config_2():
        """Linear arrangement"""
        params = []
        for i in range(N):
            params.extend([i * d, 0.0, 0.0])
        return np.array(params)

    def make_config_3():
        """Two rows"""
        params = []
        for i in range(N):
            if i < 6:
                params.extend([i * d, 0.0, 0.0])
            else:
                params.extend([(i-6) * d + d/2, 1.5, 0.0])
        return np.array(params)

    def make_config_4():
        """Random scatter in a box"""
        np.random.seed(42)
        params = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            ang = np.random.uniform(-15, 15)
            params.extend([x, y, ang])
        return np.array(params)

    def make_config_5():
        """Tighter hexagonal arrangement with rotations"""
        # Try a 3-4-3 or similar pattern
        centers = [
            (0, 0),
            (d, 0), (d/2, 1.5), (-d/2, 1.5),
            (-d, 0), (-d/2, -1.5), (d/2, -1.5),
            (0, 3.0), (-2.6, 1.5), (2.6, 1.5), (0, -3.0)
        ]
        params = []
        for x, y in centers:
            params.extend([x, y, 0.0])
        return np.array(params)

    initial_configs = [make_config_1(), make_config_2(), make_config_3(), make_config_4(), make_config_5()]

    best_side = float('inf')
    best_params = None
    best_penalty = float('inf')

    np.random.seed(42)

    # Local search with adaptive step size
    def local_search(params, max_iters=1000, step=0.1):
        """Simple local search with adaptive step size."""
        current = params.copy()
        current_obj = objective(current)
        best_local = current.copy()
        best_obj = current_obj
        step_size = step
        no_improve = 0

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break

            # Generate random perturbation
            perturbation = np.random.normal(0, step_size, len(current))
            # Keep angles small
            perturbation[2::3] *= 0.3
            candidate = current + perturbation

            cand_obj = objective(candidate)

            if cand_obj < current_obj:
                current = candidate
                current_obj = cand_obj
                no_improve = 0
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_local = current.copy()
            else:
                no_improve += 1
                if no_improve > 50:
                    step_size *= 0.9
                    if step_size < 0.001:
                        break

        return best_local, best_obj

    # Run local search from each initial configuration
    for config_idx, init_config in enumerate(initial_configs):
        if time.time() > deadline - 5:
            break

        # Run local search
        try:
            result_params, result_obj = local_search(init_config, max_iters=800, step=0.15)
            side, penalty = calc_side_and_penalty(result_params)

            if penalty < 1e-3 and side < best_side:
                best_side = side
                best_params = result_params.copy()
                best_penalty = penalty
        except Exception:
            pass

    # If local search didn't find a valid packing, try Nelder-Mead
    if best_side == float('inf'):
        for config_idx, init_config in enumerate(initial_configs[:3]):
            if time.time() > deadline - 5:
                break
            try:
                res = scipy.optimize.minimize(
                    objective,
                    init_config,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                side, penalty = calc_side_and_penalty(res.x)
                if penalty < 1e-3 and side < best_side:
                    best_side = side
                    best_params = res.x.copy()
                    best_penalty = penalty
            except Exception:
                pass

    # If still no valid packing, return a safe fallback
    if best_side == float('inf') or best_params is None:
        # Return a widely spaced configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct output
    inner = []
    for i in range(N):
        x, y, ang = best_params[i*3], best_params[i*3+1], best_params[i*3+2]
        inner.append([float(x), float(y), float(ang)])

    return inner, (0.0, 0.0), float(best_side), 0.0
