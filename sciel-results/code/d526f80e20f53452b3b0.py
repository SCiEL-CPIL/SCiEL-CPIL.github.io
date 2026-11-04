import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(angle):
    """Return vertices of a unit regular hexagon rotated by angle."""
    # Vertices of unit hexagon (side 1) centered at origin, one vertex at (1,0)
    # Angles: 0, pi/3, ...
    angles = np.arange(6) * np.pi / 3 + angle
    return np.column_stack((np.cos(angles), np.sin(angles)))

def support_function(vertices, u):
    """Compute support function h(u) = max(v . u) for a polygon given vertices."""
    return np.max(vertices @ u)

def overlap_constraint(params, i, j):
    """
    Constraint: min over axes (u . (c_j - c_i) - h_i(u) - h_j(u)) >= 0
    params: flat array [x0, y0, a0, x1, y1, a1, ...]
    """
    xi, yi, ai = params[3*i], params[3*i+1], params[3*i+2]
    xj, yj, aj = params[3*j], params[3*j+1], params[3*j+2]
    
    vi = hex_vertices(ai)
    vj = hex_vertices(aj)
    
    # Axes are normals to edges.
    # For a regular hexagon, edge normals are at angles angle + k*pi/3 + pi/6?
    # Let's check. Vertex at angle 0. Edge between angle -pi/6 and pi/6?
    # My hex_vertices generates vertices at 0, 60, ...
    # The edge between v0(1,0) and v1(0.5, sqrt(3)/2) has normal pointing at 30 deg (pi/6).
    # So normals are at angle + pi/6 + k*pi/3.
    
    axes_i = np.column_stack((np.cos(ai + np.pi/6 + np.arange(6)*np.pi/3), 
                              np.sin(ai + np.pi/6 + np.arange(6)*np.pi/3)))
    axes_j = np.column_stack((np.cos(aj + np.pi/6 + np.arange(6)*np.pi/3), 
                              np.sin(aj + np.pi/6 + np.arange(6)*np.pi/3)))
    
    all_axes = np.vstack((axes_i, axes_j))
    
    dc = np.array([xj - xi, yj - yi])
    
    # h_i(u) for each axis
    hi = np.array([support_function(vi, u) for u in all_axes])
    hj = np.array([support_function(vj, u) for u in all_axes])
    
    # Distance along axis
    dists = all_axes @ dc - hi - hj
    
    # We want min(dists) >= 0
    return np.min(dists)

def inside_constraint(params, i, S):
    """
    Constraint: Inner hex i is inside outer hex (center 0, side S, angle 0).
    Outer hex normals are at k*pi/3 + pi/6?
    Standard hexagon with flat top or pointy top?
    Let's assume standard orientation where one vertex is at (S, 0)?
    If side is S, distance to vertex is S.
    Normals to edges are at pi/6 + k*pi/3.
    The inequality is n . x <= S * sqrt(3)/2.
    """
    xi, yi, ai = params[3*i], params[3*i+1], params[3*i+2]
    vi = hex_vertices(ai)
    
    # Outer hexagon normals
    outer_angles = np.pi/6 + np.arange(6)*np.pi/3
    outer_normals = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    
    # Limit for outer hexagon
    limit = S * np.sqrt(3) / 2
    
    # Check if all vertices of inner hex are inside
    # Equivalent to: n . (c + v) <= limit for all n, v
    # n . c + n . v <= limit
    # n . c + h_inner(n) <= limit
    
    mins = []
    for n in outer_normals:
        h_val = support_function(vi, n)
        val = n[0]*xi + n[1]*yi + h_val
        mins.append(limit - val)
        
    return np.min(mins)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_S = float('inf')
    best_params = None
    
    # Initial configurations
    inits = []
    
    # 1. Hexagonal cluster (1 center, 6 ring, 5 more)
    # Center at 0
    p = [0.0, 0.0, 0.0]
    # Ring 1
    for k in range(6):
        ang = k * np.pi / 3
        x = np.sqrt(3) * np.cos(ang)
        y = np.sqrt(3) * np.sin(ang)
        p.extend([x, y, 0.0])
    # 5 more, placed in gaps or extending
    # Just add 5 more in a rough circle
    for k in range(5):
        ang = (k + 0.5) * 2 * np.pi / 5
        x = 2 * np.sqrt(3) * np.cos(ang)
        y = 2 * np.sqrt(3) * np.sin(ang)
        p.extend([x, y, 0.0])
    inits.append(np.array(p))
    
    # 2. Grid
    p = []
    # 4x3 grid
    for i in range(4):
        for j in range(3):
            x = i * np.sqrt(3)
            y = j * 1.5
            if i % 2 == 1:
                y += np.sqrt(3)/2
            p.extend([x, y, 0.0])
    # Center it
    cx = np.mean([p[i] for i in range(0, len(p), 3)])
    cy = np.mean([p[i+1] for i in range(0, len(p), 3)])
    for i in range(0, len(p), 3):
        p[i] -= cx
        p[i+1] -= cy
    inits.append(np.array(p))
    
    # 3. Random
    np.random.seed(42)
    p = []
    for _ in range(N):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        a = np.random.uniform(0, np.pi)
        p.extend([x, y, a])
    inits.append(np.array(p))
    
    constraints = []
    
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: overlap_constraint(x, i, j)})
            
    # Inside constraints (depend on S, which is the last variable)
    # We need to handle S in the constraint functions.
    # Since S is a variable, we can't easily close over it in a static list of lambdas without passing it.
    # Better: define the constraint functions to take the full vector x, and extract S.
    
    def make_inside_constraint(i):
        def fun(x):
            S = x[-1]
            return inside_constraint(x, i, S)
        return fun
        
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': make_inside_constraint(i)})
        
    # Bounds for variables
    # x, y in [-10, 10]
    # angle in [0, pi] (hexagon symmetry is pi/3, but let's allow [0, 2*pi] or [0, pi])
    # S in [1, 10]
    bounds = []
    for _ in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, np.pi)) # Angles
    bounds.append((1.0, 10.0)) # S
    
    def objective(x):
        return x[-1] # Minimize S
        
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 200, 'ftol': 1e-9})
            
            if res.success and res.fun < best_S:
                # Verify validity
                valid = True
                for c in constraints:
                    if c['fun'](res.x) < -1e-6:
                        valid = False
                        break
                if valid:
                    best_S = res.fun
                    best_params = res.x
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, np.degrees(a)])
        
    S = best_params[-1]
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
