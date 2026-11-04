import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Build a hexagonal lattice adjacency graph for 12 nodes
    n_rows, n_cols = 3, 4
    A = np.zeros((N, N))
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            if c + 1 < n_cols:
                j = r * n_cols + (c + 1)
                A[i, j] = A[j, i] = 1
            if r + 1 < n_rows:
                if c + 1 < n_cols:
                    j = (r + 1) * n_cols + (c + 1)
                    A[i, j] = A[j, i] = 1
                if c - 1 >= 0:
                    j = (r + 1) * n_cols + (c - 1)
                    A[i, j] = A[j, i] = 1
            if r - 1 >= 0:
                if c + 1 < n_cols:
                    j = (r - 1) * n_cols + (c + 1)
                    A[i, j] = A[j, i] = 1
                if c - 1 >= 0:
                    j = (r - 1) * n_cols + (c - 1)
                    A[i, j] = A[j, i] = 1
    
    degree = np.sum(A, axis=1)
    L = np.diag(degree) - A
    eigenvalues, eigenvectors = eigh(L)
    x_init = eigenvectors[:, 1] * 2.0
    y_init = eigenvectors[:, 2] * 2.0
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = np.radians(angle_deg)
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle_rad
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])
    
    def required_side(positions, cont_center, cont_angle):
        max_s = 0.0
        angle_rad = np.radians(cont_angle)
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + angle_rad
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        for i in range(N):
            cx, cy = positions[i]
            verts = get_hex_vertices(cx, cy, 0.0, 1.0)
            for vx, vy in verts:
                d = np.array([vx - cont_center[0], vy - cont_center[1]])
                projs = normals @ d
                for proj in projs:
                    if proj > 0:
                        s_needed = proj / (sqrt3 / 2)
                        max_s = max(max_s, s_needed)
        return max_s
    
    def objective(params):
        positions = params[:2 * N].reshape(N, 2)
        cont_center = params[2 * N:2 * N + 2]
        cont_angle = params[2 * N + 2]
        
        s = required_side(positions, cont_center, cont_angle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
                if dist < 1.99:
                    penalty += (1.99 - dist) ** 2 * 50
        
        return s + penalty
    
    x0 = np.concatenate([
        np.column_stack([x_init, y_init]).flatten(),
        [0.0, 0.0, 0.0]
    ])
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5})
    
    opt_params = result.x
    positions = opt_params[:2 * N].reshape(N, 2)
    cont_center = opt_params[2 * N:2 * N + 2]
    cont_angle = opt_params[2 * N + 2]
    
    side_length = required_side(positions, cont_center, cont_angle) + 0.01
    
    inner = [[positions[i, 0], positions[i, 1], 0.0] for i in range(N)]
    
    return inner, (cont_center[0], cont_center[1]), side_length, cont_angle
# EVOLVE-BLOCK-END
