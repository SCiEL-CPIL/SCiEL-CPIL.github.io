import numpy as np
from scipy.optimize import minimize

N = 12
EPS = 1e-6

def tight_grid_vertices(phi):
    # Generate vertices for a tight 3x4 grid of hexagons
    # Lattice vectors for hexagonal tiling (distance sqrt(3) between centers)
    # Let's use basis vectors:
    # u = (1.5, sqrt(3)/2)  -> dist sqrt(3)
    # v = (0, sqrt(3))       -> dist sqrt(3)
    # This creates a rhombic grid.
    # We want 3 rows, 4 cols?
    # Let's place 4 in a row along u, and 3 rows along v.
    verts = []
    for i in range(3):
        for j in range(4):
            # Center position
            cx = j * 1.5 + i * 0.0 # Shift rows?
            cy = i * np.sqrt(3) + j * (np.sqrt(3)/2)
            # Wait, standard hex grid:
            # Row i, Col j
            # x = j * 1.5 + (i % 2) * 0.75 ?
            # Let's just use the lattice vectors.
            # v1 = (1.5, sqrt(3)/2)
            # v2 = (0, sqrt(3))
            # Center = j * v1 + i * v2
            c = np.array([j * 1.5 + i * 0.0, j * (np.sqrt(3)/2) + i * np.sqrt(3)])
            # Rotate center by phi
            c_rot = np.array([np.cos(phi)*c[0]-np.sin(phi)*c[1], np.sin(phi)*c[0]+np.cos(phi)*c[1]])
            # Rotate hexagon by phi
            for k in range(6):
                a = phi + np.pi/3*k
                verts.append([c_rot[0]+np.cos(a), c_rot[1]+np.sin(a)])
    return np.array(verts)

def s_min(params, verts):
    phi, theta, cx, cy = params
    c = np.array([cx, cy])
    maxdot = -np.inf
    for i in range(6):
        a = theta + np.pi/6 + np.pi/3*i
        n = np.array([np.cos(a), np.sin(a)])
        dots = verts @ n - (c @ n)
        maxdot = max(maxdot, dots.max())
    return maxdot / np.cos(np.pi/6)

def construct():
    best = None
    best_s = np.inf
    for phi0 in np.linspace(0, np.pi/3, 10):
        for theta0 in np.linspace(0, np.pi/3, 10):
            verts = tight_grid_vertices(phi0)
            x0 = [phi0, theta0, 0.0, 0.0]
            res = minimize(s_min, x0, args=(verts,), method='Nelder-Mead', options={'maxiter': 2000, 'xatol':1e-6, 'fatol':1e-6})
            if res.fun < best_s:
                best_s = res.fun
                best = res.x
    phi, theta, cx, cy = best
    s = best_s
    inner = []
    # Rebuild centers
    for i in range(3):
        for j in range(4):
            c = np.array([j * 1.5, j * (np.sqrt(3)/2) + i * np.sqrt(3)])
            c_rot = np.array([np.cos(phi)*c[0]-np.sin(phi)*c[1], np.sin(phi)*c[0]+np.cos(phi)*c[1]])
            inner.append([c_rot[0], c_rot[1], np.degrees(phi)])
    return inner, (cx, cy), s, np.degrees(theta)
