import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
# Precompute normals for the 6 directions of the outer hexagon
angles_outer = np.deg2rad(np.arange(0, 360, 60))
normals = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))

def hexagon_vertices(cx, cy, angle_deg, size=1.0):
    theta = np.deg2rad(angle_deg)
    a = np.deg2rad(np.arange(0, 360, 60)) + theta
    x = size * np.cos(a)
    y = size * np.sin(a)
    return np.column_stack((x + cx, y + cy))

def support_function(cx, cy, angle_deg, normal):
    theta = np.deg2rad(angle_deg)
    a = np.deg2rad(np.arange(0, 360, 60)) + theta
    v = np.column_stack((np.cos(a), np.sin(a)))
    dots = v.dot(normal)
    max_dot = np.max(dots)
    c = np.array([cx, cy])
    return c.dot(normal) + max_dot

def overlap_penalty(v1, v2):
    e1 = np.roll(v1, -1, axis=0) - v1
    e2 = np.roll(v2, -1, axis=0) - v2
    n1 = np.column_stack((-e1[:, 1], e1[:, 0]))
    n2 = np.column_stack((-e2[:, 1], e2[:, 0]))
    axes = np.vstack((n1, n2))
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / norms
    p1 = v1.dot(axes.T)
    p2 = v2.dot(axes.T)
    min1 = np.min(p1, axis=0)
    max1 = np.max(p1, axis=0)
    min2 = np.min(p2, axis=0)
    max2 = np.max(p2, axis=0)
    overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
    overlap = np.maximum(0, overlap)
    return np.sum(overlap ** 2)

def objective(params):
    xs = params[0::3]
    ys = params[1::3]
    as_ = params[2::3]
    
    R = np.zeros(6)
    for k in range(6):
        n_k = normals[k]
        max_support = -np.inf
        for i in range(N):
            s = support_function(xs[i], ys[i], as_[i], n_k)
            if s > max_support:
                max_support = s
        R[k] = max_support
        
    beta = 10.0
    soft_max_R = np.log(np.sum(np.exp(beta * R))) / beta
    
    penalty = 0.0
    vertices = []
    for i in range(N):
        v = hexagon_vertices(xs[i], ys[i], as_[i])
        vertices.append(v)
        
    for i in range(N):
        for j in range(i + 1, N):
            penalty += overlap_penalty(vertices[i], vertices[j])
            
    weight = 100.0
    return soft_max_R + weight * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess: Hex grid
    xs = np.zeros(N)
    ys = np.zeros(N)
    as_ = np.zeros(N)
    
    idx = 0
    for i in range(-2, 2):
        for j in range(-1, 2):
            if idx < N:
                xs[idx] = i * np.sqrt(3) + (j % 2) * np.sqrt(3) / 2
                ys[idx] = j * 1.5
                as_[idx] = 0.0
                idx += 1
                
    x0 = np.concatenate((xs, ys, as_))
    
    best_params = x0
    best_score = np.inf
    
    np.random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts == 1:
            current_x0 = x0
        else:
            current_x0 = x0 + np.random.normal(0, 1.0, x0.shape)
            
        try:
            res = minimize(objective, current_x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-5})
            
            if res.success or res.fun < best_score:
                params = res.x
                xs = params[0::3]
                ys = params[1::3]
                as_ = params[2::3]
                
                R = np.zeros(6)
                for k in range(6):
                    n_k = normals[k]
                    max_support = -np.inf
                    for i in range(N):
                        s = support_function(xs[i], ys[i], as_[i], n_k)
                        if s > max_support:
                            max_support = s
                    R[k] = max_support
                    
                penalty = 0.0
                vertices = []
                for i in range(N):
                    v = hexagon_vertices(xs[i], ys[i], as_[i])
                    vertices.append(v)
                for i in range(N):
                    for j in range(i + 1, N):
                        penalty += overlap_penalty(vertices[i], vertices[j])
                        
                if penalty < 1e-4:
                    S = (2.0 / np.sqrt(3.0)) * np.max(R)
                    if S < best_score:
                        best_score = S
                        best_params = params.copy()
                        
        except Exception:
            pass
            
        if time.time() > deadline - 10:
            break
            
    if best_params is not None:
        xs = best_params[0::3]
        ys = best_params[1::3]
        as_ = best_params[2::3]
        
        inner = []
        for i in range(N):
            inner.append([xs[i], ys[i], as_[i]])
            
        R = np.zeros(6)
        for k in range(6):
            n_k = normals[k]
            max_support = -np.inf
            for i in range(N):
                s = support_function(xs[i], ys[i], as_[i], n_k)
                if s > max_support:
                    max_support = s
            R[k] = max
            
        S = (2.0 / np.sqrt(3.0)) * np.max(R)
        
        return inner, (0.0, 0.0), S, 0.0
    else:
        inner = []
        for i in range(N):
            inner.append([xs[i], ys[i], as_[i]])
        R = np.zeros(6)
        for k in range(6):
            n_k = normals[k]
            max_support = -np.inf
            for i in range(N):
                s = support_function(xs[i], ys[i], as_[i], n_k)
                if s > max_support:
                    max_support = s
            R[k] = max_support
        S = (2.0 / np.sqrt(3.0)) * np.max(R)
        return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
