import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N_HEX = 11
APOTHEM = math.sqrt(3) / 2.0
S3 = math.sqrt(3)
T_START = time.time()
T_LIMIT = 150.0
T_MARGIN = 8.0

# Hexagon vertices for unit hexagon at origin, angle 0
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, S3/2.0],
    [-0.5, S3/2.0],
    [-1.0, 0.0],
    [-0.5, -S3/2.0],
    [0.5, -S3/2.0]
])

def hex_vertices(cx, cy, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ HEX_VERTS.T + np.array([cx, cy])

def check_overlap(centers, angles, tol=1e-9):
    n = len(centers)
    for i in range(n):
        vi = hex_vertices(centers[i][0], centers[i][1], angles[i])
        for j in range(i+1, n):
            vj = hex_vertices(centers[j][0], centers[j][1], angles[j])
            # Check if any edge of vi intersects any edge of vj
            # Or if a vertex is inside the other hexagon
            for k in range(6):
                if point_in_hex(vi[k,0], vi[k,1], centers[j][0], centers[j][1], angles[j]):
                    return False
            # Edge intersection check
            for k in range(6):
                for l in range(6):
                    if segments_intersect(vi[k], vi[(k+1)%6], vj[l], vj[(l+1)%6]):
                        return False
    return True

def point_in_hex(px, py, cx, cy, angle):
    # Transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    c, s = np.cos(-angle), np.sin(-angle)
    lx = c*dx - s*dy
    ly = s*dx + c*dy
    # Check against 6 half-planes
    for k in range(6):
        ang = math.pi/6 + k*math.pi/3
        nx, ny = math.cos(ang), math.sin(ang)
        if lx*nx + ly*ny > APOTHEM + 1e-10:
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    # Return True if segment p1-p2 intersects segment p3-p4
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def calc_S(centers, angles, beta, cx, cy):
    # Compute the minimum side length of outer hexagon
    normals = np.array([
        [math.cos(beta + math.pi/6 + k*math.pi/3),
         math.sin(beta + math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    max_proj = 0.0
    for i in range(N_HEX):
        vi = hex_vertices(centers[i][0], centers[i][1], angles[i])
        for v in vi:
            pv = v - np.array([cx, cy])
            projs = np.abs(pv @ normals.T)
            max_proj = max(max_proj, np.max(projs))
    D = max_proj
    S = (D + APOTHEM) / APOTHEM
    return S

def generate_seeds():
    seeds = []
    # Seed 1: hexagonal grid arrangement
    S = S3
    centers = []
    # 11 hexagons in a hexagonal pattern
    # Row 0: 3
    for i in range(3):
        centers.append((i*S, 0))
    # Row 1: 5
    for i in range(5):
        centers.append((i*S - S/2, 1.5))
    # Row 2: 3
    for i in range(3):
        centers.append((i*S, 3.0))
    angles = [0.0]*N_HEX
    seeds.append((np.array(centers), np.array(angles)))
    
    # Seed 2: different arrangement
    centers2 = []
    centers2.append((0, 0))
    for k in range(6):
        ang = k*math.pi/3
        centers2.append((S*math.cos(ang), S*math.sin(ang)))
    # 4 more on a ring
    for k in range(4):
        ang = (k + 0.5)*math.pi/3
        centers2.append((2*S*math.cos(ang), 2*S*math.sin(ang)))
    angles2 = [0.0]*N_HEX
    seeds.append((np.array(centers2), np.array(angles2)))
    
    # Seed 3: compact diamond
    centers3 = []
    centers3.append((0, 0))
    centers3.append((S, 0))
    centers3.append((S/2, 1.5))
    centers3.append((-S/2, 1.5))
    centers3.append((S/2, -1.5))
    centers3.append((-S/2, -1.5))
    centers3.append((2*S, 0))
    centers3.append((-2*S, 0))
    centers3.append((3*S/2, 1.5))
    centers3.append((-3*S/2, 1.5))
    centers3.append((3*S/2, -1.5))
    angles3 = [0.0]*N_HEX
    seeds.append((np.array(centers3), np.array(angles3)))
    
    return seeds

def objective(params):
    # params: 2*N_HEX for centers, N_HEX for angles, 1 for beta
    cx, cy = params[2*N_HEX], params[2*N_HEX+1]
    beta = params[2*N_HEX+2]
    centers = params[:2*N_HEX].reshape(N_HEX, 2)
    angles = params[2*N_HEX+2+1:2*N_HEX+2+1+N_HEX]
    
    # Check overlap
    if not check_overlap(centers, angles):
        return 100.0
    
    return calc_S(centers, angles, beta, cx, cy)

def optimize_from_seed(centers_init, angles_init):
    n_params = 2*N_HEX + N_HEX + 3  # centers + angles + cx, cy, beta
    # Bounds
    bounds = [(-10, 10)] * 2*N_HEX  # centers
    bounds += [(0, math.pi)] * N_HEX  # angles
    bounds += [(-10, 10)] * 2  # cx, cy
    bounds += [(0, math.pi/6)]  # beta (only 60 degrees of symmetry)
    
    # Initial point
    cx_init = float(np.mean(centers_init[:, 0]))
    cy_init = float(np.mean(centers_init[:, 1]))
    x0 = np.concatenate([centers_init.ravel(), angles_init, [cx_init, cy_init, 0.0]])
    
    # Use differential evolution for global search
    result = differential_evolution(
        objective, bounds, x0=x0,
        maxiter=200, popsize=15, seed=42,
        tol=1e-8, mut=(0.5, 1.0), reccomb=0.8
    )
    
    if result.fun < 100.0:  # Valid
        params = result.x
        centers_opt = params[:2*N_HEX].reshape(N_HEX, 2)
        angles_opt = params[2*N_HEX:2*N_HEX+N_HEX]
        cx_opt = params[2*N_HEX+N_HEX]
        cy_opt = params[2*N_HEX+N_HEX+1]
        beta_opt = params[2*N_HEX+N_HEX+2]
        S_opt = result.fun
        return centers_opt, angles_opt, cx_opt, cy_opt, beta_opt, S_opt
    return None

def construct():
    if time.time() - T_START > T_LIMIT - T_MARGIN:
        # Emergency return
        centers = np.array([[i*S3, 0] for i in range(3)] + 
                         [[i*S3 - S3/2, 1.5] for i in range(5)] +
                         [[i*S3, 3.0] for i in range(3)])
        angles = np.zeros(N_HEX)
        cx = float(np.mean(centers[:, 0]))
        cy = float(np.mean(centers[:, 1]))
        S = calc_S(centers, angles, 0.0, cx, cy)
        inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N_HEX)]
        return inner, [float(cx), float(cy)], float(S), 0.0
    
    seeds = generate_seeds()
    best_result = None
    best_S = float('inf')
    
    for centers_init, angles_init in seeds:
        if time.time() - T_START > T_LIMIT - T_MARGIN:
            break
        try:
            result = optimize_from_seed(centers_init, angles_init)
            if result is not None:
                centers_opt, angles_opt, cx_opt, cy_opt, beta_opt, S_opt = result
                if S_opt < best_S:
                    best_S = S_opt
                    best_result = (centers_opt, angles_opt, cx_opt, cy_opt, beta_opt)
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to seed 0
        centers_opt, angles_opt = seeds[0]
        cx_opt = float(np.mean(centers_opt[:, 0]))
        cy_opt = float(np.mean(centers_opt[:, 1]))
        beta_opt = 0.0
        best_S = calc_S(centers_opt, angles_opt, beta_opt, cx_opt, cy_opt)
    
    # Final verification
    if not check_overlap(best_result[0], best_result[1]):
        # Scale up slightly
        scale = 1.0 + 1e-6
        cx_opt, cy_opt = best_result[2], best_result[3]
        centers_opt = (best_result[0] - np.array([cx_opt, cy_opt])) * scale + np.array([cx_opt, cy_opt])
        best_S = calc_S(centers_opt, best_result[1], best_result[4], cx_opt, cy_opt)
    
    inner = [[float(centers_opt[i][0]), float(centers_opt[i][1]), float(best_result[1][i])] for i in range(N_HEX)]
    center = [float(best_result[2]), float(best_result[3])]
    side = float(best_S)
    rotation = float(best_result[4])
    
    return inner, center, side, rotation
