import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth_fast(v1, v2, normals):
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        if np.min(overlaps) <= 1e-9:
            return 0.0
        return np.min(overlaps)

    def inside_violation_fast(inner_verts, outer_verts, outer_normals):
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        p_inner_max = p_inner.max(axis=0)
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def compute_penalty(positions, alphas, s, theta):
        outer_verts = hex_vertices(0, 0, s, theta)
        outer_normals = get_normals(outer_verts)
        total_penalty = 0.0
        inner_verts_list = []
        
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alphas[i])
            inner_verts_list.append(verts)
            
            viol = inside_violation_fast(verts, outer_verts, outer_normals)
            total_penalty += viol * 10.0
            
        for i in range(N):
            for j in range(i + 1, N):
                normals1 = get_normals(inner_verts_list[i])
                normals2 = get_normals(inner_verts_list[j])
                normals = np.vstack([normals1, normals2])
                pd = penetration_depth_fast(inner_verts_list[i], inner_verts_list[j], normals)
                total_penalty += pd * 10.0
                
        return total_penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        positions = params[:2 * N].reshape(N, 2)
        alphas = params[2 * N : 2 * N + N]
        s = params[2 * N + N]
        theta = params[2 * N + N + 1]
        
        if s < 1.5:
            return 1e6
            
        penalty = compute_penalty(positions, alphas, s, theta)
        return s + penalty

    def make_honeycomb(rows, cols, flat_top):
        pos = []
        if flat_top:
            # Flat-top: angle 30 deg.
            # Horizontal spacing: sqrt(3)
            # Vertical spacing: 1.5
            # Shift: sqrt(3)/2
            dx = SQRT3
            dy = 1.5
            shift = SQRT3 / 2
            for r in range(rows):
                for c in range(cols):
                    x = c * dx
                    if r % 2 == 1:
                        x += shift
                    y = r * dy
                    pos.append([x, y])
        else:
            # Pointy-top: angle 0 deg.
            # Vertical spacing: sqrt(3)
            # Horizontal spacing: 1.5
            # Shift: 1.5/2 = 0.75
            dx = 1.5
            dy = SQRT3
            shift = 0.75
            for r in range(rows):
                for c in range(cols):
                    y = c * dy
                    if r % 2 == 1:
                        y += shift
                    x = r * dx
                    pos.append([x, y])
        
        pos = np.array(pos)
        center = pos.mean(axis=0)
        pos = pos - center
        return pos

    def init_hex_cluster():
        d1 = SQRT3
        d2 = 2 * SQRT3
        pos = [[0.0, 0.0]]
        alphas = [0.0]
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
            alphas.append(0.0)
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
            alphas.append(0.0)
        return np.array(pos), np.array(alphas)

    # Generate initial configurations
    # 1. Honeycomb 3x4 (Flat Top)
    pos_34 = make_honeycomb(3, 4, True)
    alphas_34 = np.full(N, np.radians(30))
    
    # 2. Honeycomb 4x3 (Pointy Top)
    pos_43 = make_honeycomb(4, 3, False)
    alphas_43 = np.full(N, 0.0)
    
    # 3. Hex Cluster
    pos_hc, alphas_hc = init_hex_cluster()
    
    inits = [
        (pos_34, alphas_34, np.radians(0)),   # Outer 0 deg
        (pos_34, alphas_34, np.radians(30)),  # Outer 30 deg
        (pos_43, alphas_43, np.radians(0)),   # Outer 0 deg
        (pos_43, alphas_43, np.radians(30)),  # Outer 30 deg
        (pos_hc, alphas_hc, np.radians(0)),   # Outer 0 deg
        (pos_hc, alphas_hc, np.radians(30)),  # Outer 30 deg
    ]

    best_s = np.inf
    best_params = None

    # Multi-start optimization
    max_runs = 10
    for run in range(max_runs):
        if time.time() > deadline - 15:
            break
            
        # Select initialization
        idx = run % len(inits)
        init_pos, init_alphas, init_theta = inits[idx]
        
        # Add slight randomness for diversity in later runs
        if run >= len(inits):
            init_pos = init_pos + np.random.normal(0, 0.1, init_pos.shape)
            init_alphas = init_alphas + np.random.uniform(-0.1, 0.1, N)
            
        init_s = 4.0
        
        x0 = np.concatenate([
            init_pos.flatten(), 
            init_alphas, 
            [init_s], 
            [init_theta]
        ])
        
        try:
            # Use L-BFGS-B
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if res.fun < best_s:
                # Verify validity
                pos = res.x[:2 * N].reshape(N, 2)
                alphas = res.x[2 * N : 2 * N + N]
                s = res.x[2 * N + N]
                theta = res.x[2 * N + N + 1]
                
                penalty = compute_penalty(pos, alphas, s, theta)
                
                if penalty < 1e-5:
                    best_s = s
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass

    # If no valid solution found via optimization, fallback to a known valid configuration
    if best_params is None:
        pos, alphas = inits[0][0], inits[0][1]
        theta = inits[0][2]
        s = 5.0
        best_params = np.concatenate([pos.flatten(), alphas, [s], [theta]])
        best_s = s

    # Extract final positions
    final_pos = best_params[:2 * N].reshape(N, 2)
    final_alphas = best_params[2 * N : 2 * N + N]
    final_s = best_params[2 * N + N]
    final_theta = best_params[2 * N + N + 1]
    
    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_alphas[i])] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(final_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
