import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # 10s margin

def hex_face_normals(theta):
    base = np.array([np.pi/6 + i*np.pi/3 for i in range(6)])
    return np.stack([np.cos(base + theta), np.sin(base + theta)], axis=1)

def enclosing_hex_side(points, cx, cy, theta):
    normals = hex_face_normals(theta)
    rel = points - np.array([cx, cy])
    proj = rel @ normals.T
    max_proj = proj.max(axis=0)
    s = max_proj.max() / (np.sqrt(3)/2)
    return s

def objective(params, points):
    cx, cy, theta = params
    return enclosing_hex_side(points, cx, cy, theta)

def fit_hexagon(points, max_time=20.0):
    cx0, cy0 = points.mean(axis=0)
    best = None
    best_val = np.inf
    theta_inits = [0.0, np.pi/6, np.pi/3, np.pi/4, np.pi/2, np.pi/3*2]
    for theta0 in theta_inits:
        if time.time() > DEADLINE - 15:
            break
        res = minimize(objective, [cx0, cy0, theta0], args=(points,),
                       method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
        if res.fun < best_val:
            best_val = res.fun
            best = res.x
    if best is None:
        best = np.array([cx0, cy0, 0.0])
    cx, cy, theta = best
    s = enclosing_hex_side(points, cx, cy, theta)
    return cx, cy, s, np.degrees(theta)

# Hex lattice
sqrt3 = np.sqrt(3)
v1 = np.array([2.0, 0.0])
v2 = np.array([1.0, sqrt3])

def lattice_point(i, j):
    return i * v1 + j * v2

# Generate candidate shapes
# Shape 1: 12 closest lattice points to origin
pool = []
for i in range(-3, 4):
    for j in range(-3, 4):
        p = lattice_point(i, j)
        if np.linalg.norm(p) < 5.5:
            pool.append((np.linalg.norm(p), i, j))
pool.sort()
shape1 = np.array([lattice_point(i, j) for _, i, j in pool[:12]])

# Shape 2: 4x3 grid
shape2 = np.array([lattice_point(i, j) for i in range(-1, 3) for j in range(0, 3)])

# Shape 3: hex cluster 1+6+5
shape3_coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                 (2,0), (1,1), (0,2), (-1,2), (-2,1)]
shape3 = np.array([lattice_point(i, j) for i, j in shape3_coords])

candidates = [shape1, shape2, shape3]

best_result = None
best_s = np.inf

for shape in candidates:
    if time.time() > DEADLINE - 10:
        break
    cx, cy, s, theta_deg = fit_hexagon(shape)
    if s < best_s:
        best_s = s
        best_result = (shape, cx, cy, s, theta_deg)

if best_result is None:
    cx, cy, s, theta_deg = fit_hexagon(candidates[0])
    best_result = (candidates[0], cx, cy, s, theta_deg)

shape, cx, cy, s, theta_deg = best_result
inner = [[float(p[0]), float(p[1]), 0.0] for p in shape]
center = (float(cx), float(cy))
side = float(s)
angle = float(theta_deg)

def construct():
    return inner, center, side, angle
