import math
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
APOTHEM = math.sqrt(3) / 2.0
N = 11
EPS = 1e-9

def axial_to_cartesian(axial_list):
    """Convert axial coordinates (q, r) to Cartesian (x, y) for pointy-top hexagons of side 1."""
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def get_hex_vertices(center, orientation=0.0):
    """Get vertices of a unit hexagon centered at 'center' with given orientation."""
    cx, cy = center
    verts = []
    for k in range(6):
        angle = orientation + k * math.pi / 3.0
        x = cx + math.cos(angle)
        y = cy + math.sin(angle)
        verts.append((x, y))
    return np.array(verts)

def objective_and_constraints(x):
    """
    x = [c1x, c1y, ..., c11x, c11y, Cx, Cy, theta]
    Returns (objective, constraints)
    Objective: Apothem of outer hexagon (minimize this)
    Constraints: |ci - cj|^2 >= 3 for all i < j
    """
    centers = x[:2*N].reshape(N, 2)
    C = x[2*N:2*N+2]
    theta = x[2*N+2]
    
    normals = []
    for k in range(6):
        angle = theta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(angle), math.sin(angle)))
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(N):
        verts = get_hex_vertices(centers[i], 0.0)
        diffs = verts - C
        projs = diffs @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    A_out = max_proj
    
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            diff = centers[i] - centers[j]
            dist_sq = np.dot(diff, diff)
            cons.append(dist_sq - 3.0)
    
    return A_out, cons

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = []
    
    # 1. 7+4 Asymmetric
    c1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,2), (-2,0)]
    candidates.append(c1)
    
    # 2. 3x3+2
    c2 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1)]
    candidates.append(c2)
    
    # 3. 4-3-4
    c3 = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (1,2), (2,2), (3,2)]
    candidates.append(c3)
    
    # 4. Star
    c4 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)]
    candidates.append(c4)
    
    # 5. Dense 3x3+2 variant
    c5 = [(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1), (0,-1)]
    candidates.append(c5)

    # Add perturbations
    for base_axial in [c1, c2, c3]:
        base_pts = axial_to_cartesian(base_axial)
        for _ in range(3):
            perturbed = base_pts + np.random.normal(0, 0.2, base_pts.shape)
            candidates.append(perturbed)

    best_S = float('inf')
    best_x = None
    
    for idx, cand in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        if len(cand) == 11 and isinstance(cand[0], tuple):
            pts = axial_to_cartesian(cand)
        else:
            pts = np.array(cand)
            
        C0 = np.mean(pts, axis=0)
        theta0 = 0.0
        x0 = np.concatenate([pts.flatten(), C0, [theta0]])
        
        def make_constraint(i, j):
            def con(x):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                diff = ci - cj
                return np.dot(diff, diff) - 3.0
            return con
        
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
                
        def obj(x):
            A_out, _ = objective_and_constraints(x)
            return A_out
            
        try:
            res = minimize(obj, x0, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success or res.fun < best_S:
                A_out, cons = objective_and_constraints(res.x)
                if np.min(cons) >= -1e-6:
                    S_val = (2.0 / math.sqrt(3.0)) * A_out
                    if S_val < best_S:
                        best_S = S_val
                        best_x = res.x
        except Exception:
            pass

    if best_x is None:
        pts = axial_to_cartesian(candidates[0])
        C = np.mean(pts, axis=0)
        theta = 0.0
        best_x = np.concatenate([pts.flatten(), C, [theta]])
        A_out, _ = objective_and_constraints(best_x)
        best_S = (2.0 / math.sqrt(3.0)) * A_out

    def penalized_obj(x):
        A_out, cons = objective_and_constraints(x)
        penalty = 0.0
        for c in cons:
            if c < 0:
                penalty += c**2 * 1000.0
        return A_out + penalty
        
    try:
        res_polish = minimize(penalized_obj, best_x, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res_polish.fun < best_S:
             A_out, cons = objective_and_constraints(res_polish.x)
             if np.min(cons) >= -1e-5:
                 best_S = (2.0 / math.sqrt(3.0)) * A_out
                 best_x = res_polish.x
    except:
        pass

    centers = best_x[:2*N].reshape(N, 2)
    C = best_x[2*N:2*N+2]
    theta = best_x[2*N+2]
    
    A_out, _ = objective_and_constraints(best_x)
    S = (2.0 / math.sqrt(3.0)) * A_out
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), 0.0])
        
    center = [float(C[0]), float(C[1])]
    side = float(S)
    rotation = float(math.degrees(theta))
    
    return inner, center, side, rotation
