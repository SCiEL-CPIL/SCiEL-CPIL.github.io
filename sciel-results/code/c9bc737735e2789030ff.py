import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    angles = angle + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _min_separation(angle_diff_deg):
    phi = np.radians(angle_diff_deg)
    angles_A = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    verts_A = np.column_stack([np.cos(angles_A), np.sin(angles_A)])
    angles_B = angles_A + phi
    verts_B = np.column_stack([np.cos(angles_B), np.sin(angles_B)])
    
    thetas = np.linspace(0, 2 * np.pi, 360)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    h_A = np.max(verts_A @ np.column_stack([cos_t, sin_t]), axis=0)
    h_B = np.max(verts_B @ np.column_stack([cos_t, sin_t]), axis=0)
    
    return float(np.min(h_A + h_B))

def construct():
    sp = np.sqrt(3)
    
    # Hexagonal cluster initial config
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    positions[0] = [0.0, 0.0]
    for k in range(1, 7):
        a = (k - 1) * np.pi / 3
        positions[k] = [sp * np.cos(a), sp * np.sin(a)]
    
    # 4 more hexagons at distance 2*sp in some directions
    for k in range(4):
        a = k * np.pi / 3 + np.pi / 6
        positions[6 + k] = [2 * sp * np.cos(a), 2 * sp * np.sin(a)]
    
    def unpack(params):
        x = params[0:N]
        y = params[N:2 * N]
        ang = params[2 * N:3 * N]
        s = params[3 * N]
        return x, y, ang, s

    def penalized_objective(params, mu=50.0):
        x, y, ang, s = unpack(params)
        penalty = 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                ang_diff = ang[i] - ang[j]
                min_sep = _min_separation(ang_diff)
                viol = max(0.0, min_sep - dist)
                penalty += viol * viol

        for i in range(N):
            verts = _hexagon_vertices(x[i], y[i], 1.0, ang[i])
            for v in verts:
                for k in range(6):
                    theta = (k * 60 + 30) * np.pi / 180
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    d = v[0] * nx + v[1] * ny
                    limit = s * np.sqrt(3) / 2
                    viol = max(0.0, d - limit)
                    penalty += viol * viol

        return s + mu * penalty

    best_s = float('inf')
    best_params = None

    # Multiple initial configurations
    configs = []
    
    # Hexagonal cluster
    configs.append((positions.copy(), angles.copy(), 2.5))
    
    # Line
    pos_line = np.array([[k * sp, 0.0] for k in range(-N // 2, N - N // 2)])
    ang_line = np.zeros(N)
    configs.append((pos_line, ang_line, 4.0))
    
    # Random-ish spread
    rng = np.random.RandomState(42)
    pos_rand = rng.randn(N, 2) * 2.0
    ang_rand = rng.randn(N) * 15.0
    configs.append((pos_rand, ang_rand, 5.0))
    
    for pos_init, ang_init, s_init in configs:
        x0 = np.concatenate([pos_init[:, 0], pos_init[:, 1], ang_init, [s_init]])
        bounds = [(None, None)] * (3 * N) + [(1.0, None)]
        
        result = minimize(penalized_objective, x0, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12})
        
        s_val = result.x[3 * N]
        if s_val < best_s:
            best_s = s_val
            best_params = result.x.copy()
    
    if best_params is None:
        x, y, ang, s = unpack(np.concatenate([positions[:, 0], positions[:, 1], angles, [2.5]]))
    else:
        x, y, ang, s = unpack(best_params)
    
    inner = [[float(x[i]), float(y[i]), float(ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(s), 0.0
# EVOLVE-BLOCK-END
