import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
S3 = np.sqrt(3.0)

UNIT_VERTS = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0]
])

def get_rotated_verts(angle_deg):
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    return UNIT_VERTS @ R.T

def outer_side_length(C, A, outer_angle_deg=0.0):
    outer_rad = np.deg2rad(outer_angle_deg)
    c_o, s_o = np.cos(outer_rad), np.sin(outer_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    base_normals = np.array([
        [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
        [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
        [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
        [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
        [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
        [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
    ])
    normals = base_normals @ R_o.T
    
    max_proj = 0.0
    for k in range(6):
        n = normals[k]
        for i in range(N):
            verts = get_rotated_verts(A[i]) + C[i]
            proj = verts @ n
            m = np.max(proj)
            if m > max_proj:
                max_proj = m
    return max_proj * 2.0 / S3

def check_validity(C, A, S, eps=1e-7):
    base_normals = np.array([
        [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
        [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
        [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
        [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
        [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
        [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
    ])
    apothem = S * S3 / 2.0
    for k in range(6):
        n = base_normals[k]
        for i in range(N):
            verts = get_rotated_verts(A[i]) + C[i]
            proj = verts @ n
            if np.max(proj) > apothem + eps:
                return False
            
    for i in range(N):
        for j in range(i + 1, N):
            verts_i = get_rotated_verts(A[i])
            verts_j = get_rotated_verts(A[j])
            
            edges_i = np.roll(verts_i, -1, axis=0) - verts_i
            normals_i = np.array([[-e[1], e[0]] for e in edges_i])
            norms_i = np.linalg.norm(normals_i, axis=1, keepdims=True)
            normals_i = normals_i / norms_i
            
            edges_j = np.roll(verts_j, -1, axis=0) - verts_j
            normals_j = np.array([[-e[1], e[0]] for e in edges_j])
            norms_j = np.linalg.norm(normals_j, axis=1, keepdims=True)
            normals_j = normals_j / norms_j
            
            all_normals = np.vstack([normals_i, normals_j])
            
            d = C[i] - C[j]
            separated = False
            for n in all_normals:
                proj_i = (verts_i + C[i]) @ n
                proj_j = (verts_j + C[j]) @ n
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - eps or max_j < min_i - eps:
                    separated = True
                    break
            if not separated:
                return False
    return True

def generate_cluster_config(set_id):
    centers = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        centers.append([S3 * np.cos(a), S3 * np.sin(a)])
    
    if set_id == 0:
        angles = [0, 90, 180, 270]
    elif set_id == 1:
        angles = [30, 120, 210, 300]
    elif set_id == 2:
        angles = [0, 60, 180, 240]
    elif set_id == 3:
        angles = [30, 90, 210, 270]
    else:
        angles = [0, 90, 180, 270]
        
    for a_deg in angles:
        a = np.deg2rad(a_deg)
        centers.append([3.0 * np.cos(a), 3.0 * np.sin(a)])
        
    return np.array(centers)

def optimize_rotation(C, A, deadline):
    def obj(theta_deg):
        if time.time() > deadline:
            return 1e9
        rad = np.deg2rad(theta_deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        C_rot = C @ R.T
        A_rot = A + theta_deg
        return outer_side_length(C_rot, A_rot, 0.0)
        
    res = minimize_scalar(obj, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
    return res.x, res.fun

def construct():
    start = time.time()
    deadline = start + 140.0
    
    best_side = float('inf')
    best_config = None
    
    for set_id in range(4):
        if time.time() > deadline - 10:
            break
            
        C = generate_cluster_config(set_id)
        A = np.zeros(N)
        
        theta_best, side_best = optimize_rotation(C, A, deadline)
        
        rad = np.deg2rad(theta_best)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        C_rot = C @ R.T
        A_rot = A + theta_best
        
        if side_best < best_side:
            if check_validity(C_rot, A_rot, side_best):
                best_side = side_best
                best_config = (C_rot, A_rot, side_best)
                
    if best_config is None:
        C = np.zeros((N, 2))
        for i in range(N):
            C[i] = [i * S3, 0.0]
        A = np.zeros(N)
        S = outer_side_length(C, A, 0.0)
        best_config = (C, A, S)
        
    C, A, S = best_config
    inner = [[float(C[i, 0]), float(C[i, 1]), float(A[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
