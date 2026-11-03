import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 120.0
START_TIME = time.time()

def hex_vertices(center, side, angle_rad):
    cx, cy = center
    return np.array([[cx + side * np.cos(angle_rad + i * np.pi / 3),
                      cy + side * np.sin(angle_rad + i * np.pi / 3)] for i in range(6)])

def point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p - (a + t * ab))

def min_distance_hexagons(v1, v2):
    min_dist = np.inf
    for i in range(6):
        for j in range(6):
            d = point_to_segment_dist(v1[i], v2[j], v2[(j+1)%6])
            min_dist = min(min_dist, d)
            d = point_to_segment_dist(v2[j], v1[i], v1[(i+1)%6])
            min_dist = min(min_dist, d)
    return min_dist

def compute_objective(params):
    """
    params: [x0,y0,a0, ..., xN-1,yN-1,aN-1, s] where s is outer side length
    """
    s = params[-1]
    if s < 1.0:
        return 1e6
    
    violations = []
    inner_verts = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        verts = hex_vertices((x, y), 1.0, a)
        inner_verts.append(verts)
    
    # Containment
    for i in range(N):
        for v in inner_verts[i]:
            max_viol = 0
            for j in range(6):
                theta = j * np.pi / 3 + np.pi / 6
                nx, ny = np.cos(theta), np.sin(theta)
                d = s * np.sqrt(3) / 2
                max_viol = max(max_viol, v[0]*nx + v[1]*ny - d)
            violations.append(max_viol)
    
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            d = min_distance_hexagons(inner_verts[i], inner_verts[j])
            violations.append(-d)
    
    if not violations:
        return 0.0
    violations = np.array(violations)
    m = np.max(violations)
    if m > 1e5:
        return m
    return m + np.log(np.sum(np.exp(violations - m)))

def construct():
    global START_TIME, DEADLINE
    deadline = START_TIME + DEADLINE - 5.0
    
    # Hexagonal lattice init
    init = np.zeros(3*N + 1)
    pos = [(0,0)]
    for i in range(6):
        a = i * np.pi/3
        pos.append((2*np.cos(a), 2*np.sin(a)))
    for a in [0, np.pi/3, 2*np.pi/3, np.pi]:
        pos.append((4*np.cos(a), 4*np.sin(a)))
    for i in range(N):
        init[3*i], init[3*i+1] = pos[i]
    init[-1] = 8.0
    
    best_p, best_f = init.copy(), np.inf
    
    for r in range(8):
        if time.time() > deadline:
            break
        p = init.copy()
        if r > 0:
            p[:-1] += np.random.normal(0, 0.3, size=p[:-1].shape)
            p[-1] = 7.0 + r * 0.5
        
        res = minimize(compute_objective, p, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < best_f:
            best_f, best_p = res.fun, res.x
    
    inner = [[best_p[3*i], best_p[3*i+1], np.degrees(best_p[3*i+2]) % 360] for i in range(N)]
    s = max(best_p[-1], 2.0)
    return inner, (0.0, 0.0), s, 0.0
