import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _hexagon_contains_hexagon(cx, cy, S, rot_deg, px, py, p_rot_deg):
    """Check if unit hexagon at (px,py) with rotation p_rot_deg fits inside
    hexagon centered at (cx,cy) with side S and rotation rot_deg."""
    # Vertices of outer hexagon
    rot = math.radians(rot_deg)
    outer_verts = []
    for k in range(6):
        ang = rot + k * math.pi / 3.0
        outer_verts.append((cx + S * math.cos(ang), cy + S * math.sin(ang)))
    outer_verts = np.array(outer_verts)
    
    # Vertices of inner hexagon
    p_rot = math.radians(p_rot_deg)
    inner_verts = []
    for k in range(6):
        ang = p_rot + k * math.pi / 3.0
        inner_verts.append((px + math.cos(ang), py + math.sin(ang)))
    inner_verts = np.array(inner_verts)
    
    # Check each inner vertex is inside outer hexagon
    # Use cross product test for convex polygon
    for i in range(6):
        vx, vy = inner_verts[i]
        for j in range(6):
            ax, ay = outer_verts[j]
            bx, by = outer_verts[(j + 1) % 6]
            cross = (bx - ax) * (vy - ay) - (by - ay) * (vx - ax)
            if cross < -1e-9:
                return False
    return True

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Initial configuration: hexagonal cluster
    def initial_positions():
        pts = [(0, 0),
               (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
               (1, 1), (-1, 2), (-2, 1), (2, -1)]
        result = []
        for q, r in pts:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            result.append((x, y))
        return np.array(result)
    
    def compute_S(positions, rotations, center, outer_rot):
        """Compute minimum side length S such that all hexagons fit."""
        cx, cy = center
        # For each inner hexagon, compute max distance from center to its vertices
        # projected onto the 6 normal directions of the outer hexagon
        outer_rot_rad = math.radians(outer_rot)
        normals = []
        for k in range(6):
            ang = outer_rot_rad + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for i in range(len(positions)):
            px, py = positions[i]
            p_rot_rad = math.radians(rotations[i])
            # Inner hexagon vertices
            for k in range(6):
                ang = p_rot_rad + k * math.pi / 3.0
                vx = px + math.cos(ang)
                vy = py + math.sin(ang)
                for nx, ny in normals:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    max_proj = max(max_proj, abs(proj))
        
        S = (max_proj + APOTHEM) / APOTHEM
        return S
    
    def objective(params):
        """Objective: minimize S + penalty for overlaps."""
        n = N
        # params: [x0,y0,rot0, x1,y1,rot1, ..., x10,y10,rot10, cx, cy, outer_rot]
        positions = np.array([[params[3*i], params[3*i+1]] for i in range(n)])
        rotations = np.array([params[3*i+2] for i in range(n)])
        cx, cy = params[3*n], params[3*n+1]
        outer_rot = params[3*n+2]
        
        S = compute_S(positions, rotations, (cx, cy), outer_rot)
        
        # Overlap penalty: for each pair, check if hexagons overlap
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Two unit hexagons overlap if distance < 2*apothem (roughly)
                # More precisely, min distance for non-overlap is 2*APOTHEM = sqrt(3)
                min_dist = math.sqrt(3)  # 2 * APOTHEM
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return S + 10.0 * penalty
    
    def optimize_from_start(start_params):
        """Run Nelder-Mead optimization from a starting point."""
        try:
            res = minimize(objective, start_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            return res.x, res.fun
        except Exception:
            return start_params, objective(start_params)
    
    # Generate multiple starting configurations
    def make_params(positions, rotations, center, outer_rot):
        params = []
        for i in range(N):
            params.extend([positions[i, 0], positions[i, 1], rotations[i]])
        params.extend([center[0], center[1], outer_rot])
        return np.array(params)
    
    starts = []
    
    # Start 1: Standard hexagonal cluster
    pos1 = initial_positions()
    rot1 = np.zeros(N)
    c1 = (0.0, 0.0)
    starts.append(make_params(pos1, rot1, c1, 0.0))
    
    # Start 2: Slightly rotated cluster
    pos2 = initial_positions()
    rot2 = np.full(N, 15.0)
    starts.append(make_params(pos2, rot2, c1, 15.0))
    
    # Start 3: Cross shape
    cross_pts = [(0, 0),
                 (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, 0), (-2, 0), (0, 2), (0, -2)]
    pos3 = np.array([[math.sqrt(3)*(q+r/2.0), 1.5*r] for q, r in cross_pts])
    rot3 = np.zeros(N)
    starts.append(make_params(pos3, rot3, c1, 0.0))
    
    # Start 4: 3x3 block
    block_pts = [(q, r) for q in range(3) for r in range(3)] + [(-1, 1), (3, 1)]
    pos4 = np.array([[math.sqrt(3)*(q+r/2.0), 1.5*r] for q, r in block_pts])
    rot4 = np.zeros(N)
    starts.append(make_params(pos4, rot4, c1, 0.0))
    
    # Start 5: Random perturbation of cluster
    pos5 = initial_positions() + np.random.randn(N, 2) * 0.3
    rot5 = np.random.randn(N) * 10.0
    starts.append(make_params(pos5, rot5, c1, 0.0))
    
    # Start 6: Elongated
    elong_pts = [(q, 0) for q in range(5)] + [(q, 1) for q in range(6)]
    pos6 = np.array([[math.sqrt(3)*(q+r/2.0), 1.5*r] for q, r in elong_pts])
    rot6 = np.zeros(N)
    starts.append(make_params(pos6, rot6, c1, 0.0))
    
    # Optimize from each start
    best_params = None
    best_val = float('inf')
    
    for i, start in enumerate(starts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        params, val = optimize_from_start(start)
        if val < best_val:
            best_val = val
            best_params = params
    
    # Extract best solution
    if best_params is None:
        # Fallback
        pos = initial_positions()
        rot = np.zeros(N)
        c = (0.0, 0.0)
        S = compute_S(pos, rot, c, 0.0)
        inner = [[float(pos[i, 0]), float(pos[i, 1]), float(rot[i])] for i in range(N)]
        return inner, [float(c[0]), float(c[1])], float(S), 0.0
    
    n = N
    positions = np.array([[best_params[3*i], best_params[3*i+1]] for i in range(n)])
    rotations = np.array([best_params[3*i+2] for i in range(n)])
    cx, cy = best_params[3*n], best_params[3*n+1]
    outer_rot = best_params[3*n+2]
    
    # Final precise S computation
    S = compute_S(positions, rotations, (cx, cy), outer_rot)
    
    # Verify no overlaps (should be satisfied by penalty)
    # Small safety scale
    SCALE = 1.0 + 1e-10
    positions_scaled = positions * SCALE
    
    inner = []
    for i in range(N):
        inner.append([float(positions_scaled[i, 0]), float(positions_scaled[i, 1]), float(rotations[i])])
    
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(outer_rot)
    
    return inner, center, side, rotation
