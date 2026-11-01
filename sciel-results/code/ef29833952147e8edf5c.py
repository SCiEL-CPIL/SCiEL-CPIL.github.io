import numpy as np
import time
import sys

N = 11
SQRT3 = np.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    # Initial guess
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (0, -2), (-2, 0), (0, 2)
    ]
    
    initial_pos = []
    for q, r in axial_coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        initial_pos.append([x, y])
        
    initial_pos = np.array(initial_pos)
    initial_angles = np.zeros(N)
    
    # Variables: x_0, y_0, theta_0, ..., x_10, y_10, theta_10
    p0 = np.concatenate([initial_pos.flatten(), initial_angles])
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
            
        pos = params[:2*N].reshape(-1, 2)
        angles = params[2*N:]
        
        # 1. Check Overlaps
        min_dist = SQRT3 - 1e-6
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        
        # 2. Compute Required Outer Side Length S
        # Base vertices
        base_verts = np.array([
            [1.0, 0.0],
            [0.5, SQRT3 / 2.0],
            [-0.5, SQRT3 / 2.0],
            [-1.0, 0.0],
            [-0.5, -SQRT3 / 2.0],
            [0.5, -SQRT3 / 2.0]
        ])
        
        # Outer Hexagon Normals
        normals = np.array([
            [1.0, 0.0],
            [0.5, SQRT3 / 2.0],
            [-0.5, SQRT3 / 2.0],
            [-1.0, 0.0],
            [-0.5, -SQRT3 / 2.0],
            [0.5, -SQRT3 / 2.0]
        ])
        
        max_proj = 0.0
        for i in range(N):
            # Rotate vertices
            c = np.cos(angles[i])
            s = np.sin(angles[i])
            rot = np.array([[c, -s], [s, c]])
            v_i = pos[i] + base_verts @ rot.T
            projs = v_i @ normals.T
            m_i = np.max(projs)
            if m_i > max_proj:
                max_proj = m_i
        
        S = (2.0 / SQRT3) * max_proj
        cost = S + 10.0 * penalty
        return cost

    best_cost = 1e9
    best_params = p0
    
    try:
        from scipy.optimize import minimize
        
        # Run 1
        res = minimize(objective, p0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9})
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
            
        # Run 2: Perturbed
        if time.time() < deadline - 20:
            p0_perturbed = p0.copy()
            p0_perturbed[2*N:] += np.pi / 6 # Try 30 deg rotation
            res2 = minimize(objective, p0_perturbed, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9})
            if res2.fun < best_cost:
                best_cost = res2.fun
                best_params = res2.x
                
    except Exception:
        pass

    pos = best_params[:2*N].reshape(-1, 2)
    angles = best_params[2*N:]
    
    # Calculate final S
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])
    normals = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])
    
    max_proj = 0.0
    for i in range(N):
        c = np.cos(angles[i])
        s = np.sin(angles[i])
        rot = np.array([[c, -s], [s, c]])
        v_i = pos[i] + base_verts @ rot.T
        projs = v_i @ normals.T
        m_i = np.max(projs)
        if m_i > max_proj:
            max_proj = m_i
            
    S_final = (2.0 / SQRT3) * max_proj
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(angles[i] * 180.0 / np.pi)])
        
    return inner, (0.0, 0.0), float(S_final), 0.0
# EVOLVE-BLOCK-END
