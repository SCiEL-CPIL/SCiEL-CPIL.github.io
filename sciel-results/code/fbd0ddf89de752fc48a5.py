import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, s, angle_deg):
        """Return vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            ang = a + k * math.pi / 3.0
            verts.append((cx + s * math.cos(ang), cy + s * math.sin(ang)))
        return verts

    def bounding_hex_size(points, beta):
        """
        Given list of (x,y) centers of unit hexagons, find minimum side length S
        of a regular hexagon (with orientation beta) that contains all of them.
        Returns (S, cx, cy) where (cx,cy) is the center of the bounding hexagon.
        """
        pts = np.array(points)
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals)  # (6, 2)
        
        # For each normal direction, compute min and max projection
        projs = pts @ normals.T  # (n, 6)
        mins = projs.min(axis=0)  # (6,)
        maxs = projs.max(axis=0)  # (6,)
        
        # The bounding hexagon in direction k has half-width = (max_k - min_k)/2
        # and the center offset in that direction = (max_k + min_k)/2
        # For a regular hexagon, all half-widths must be equal to D (distance from center to each side)
        # So we need to find cx, cy such that for all k:
        #   (pts @ n_k).max() - cx*n_k[0] - cy*n_k[1] <= D
        #   -(pts @ n_k).min() + cx*n_k[0] + cy*n_k[1] <= D
        # i.e., |proj_k - c·n_k| <= D for all k, where c = (cx, cy)
        
        # This is: for all k, |c·n_k - mid_k| <= D - half_range_k
        # where mid_k = (max_k + min_k)/2, half_range_k = (max_k - min_k)/2
        # So we need: c·n_k - mid_k <= D - half_range_k  AND  -(c·n_k - mid_k) <= D - half_range_k
        # i.e., |c·n_k - mid_k| <= D - half_range_k
        
        # To minimize D, we can set D = max_k (half_range_k + |c·n_k - mid_k|)
        # and minimize over c.
        
        # Use a smooth approximation for optimization
        def D_of_c(c, temp=0.1):
            c_arr = np.array(c)
            # c·n_k for each k
            cn = normals @ c_arr  # (6,)
            mids = (maxs + mins) / 2.0
            half_ranges = (maxs - mins) / 2.0
            diffs = np.abs(cn - mids)
            vals = half_ranges + diffs
            # Soft max
            max_val = vals.max()
            if max_val > 1000:
                return 1e6
            shifted = (vals - max_val) / temp
            shifted = np.clip(shifted, -50, 50)
            log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temp
            return temp * log_sum
        
        # Optimize over c = (cx, cy)
        # Initial guess: centroid of points
        c0 = [pts[:, 0].mean(), pts[:, 1].mean()]
        
        res = minimize(D_of_c, c0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
        
        if res.fun < 1e5:
            cx, cy = res.x
            # Compute exact D
            cn = normals @ np.array([cx, cy])
            mids = (maxs + mins) / 2.0
            half_ranges = (maxs - mins) / 2.0
            D = np.max(half_ranges + np.abs(cn - mids))
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy)
        else:
            return float('inf'), (c0[0], c0[1])

    def evaluate_configuration(positions, angles, beta):
        """
        Given positions (list of (x,y)) and angles (list of deg), 
        check if hexagons are disjoint and compute bounding hex size.
        Returns (valid, S, center).
        """
        # Check pairwise distance between centers
        # Two unit hexagons with centers distance d apart and relative angle theta
        # are disjoint iff d >= 2 * apothem * cos(min(theta, 60-theta) ... ) 
        # Actually, the minimum distance between two unit hexagons depends on their relative orientation.
        # For simplicity, use a conservative check: distance between centers >= 2 * apothem * 2 = 2*sqrt(3)
        # This is the distance when they share an edge (aligned). For rotated, it could be less.
        # Better: compute the actual Minkowski sum or use a safe bound.
        # The minimum distance between two unit regular hexagons is sqrt(3) (apothem*2) when aligned edge-to-edge.
        # If rotated, the minimum distance can be as low as 1.0 (vertex to vertex in worst case? No).
        # Actually, two unit hexagons: the minimum distance between them is 0 if overlapping.
        # The distance at which they just touch: for aligned, it's 2*apothem = sqrt(3) ≈ 1.732
        # For 30-degree rotation, it's different.
        # Conservative: require center distance >= 2*apothem = sqrt(3)
        
        min_dist = 2 * APOTHEM  # sqrt(3)
        
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_dist - 1e-9:
                    return False, float('inf'), (0, 0)
        
        S, center = bounding_hex_size(positions, beta)
        return True, S, center

    def smooth_objective(params, temperature=0.1):
        """
        Smooth objective for optimization.
        params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, beta]
        where ai is rotation in radians, beta is bounding hex orientation.
        Returns smooth approximation of bounding hex size + penalty for overlaps.
        """
        n = N
        # Extract parameters
        pos = []
        angles_rad = []
        for i in range(n):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            pos.append((x, y))
            angles_rad.append(a)
        beta = params[3 * n]
        
        # Compute bounding hex size (smooth)
        pts = np.array(pos)
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals)
        
        projs = pts @ normals.T
        mins = projs.min(axis=0)
        maxs = projs.max(axis=0)
        mids = (maxs + mins) / 2.0
        half_ranges = (maxs - mins) / 2.0
        
        # We need to find center c that minimizes max_k (half_range_k + |c·n_k - mid_k|)
        # For smoothness, use a fixed center estimate: centroid
        c = np.array([pts[:, 0].mean(), pts[:, 1].mean()])
        cn = normals @ c
        D_vals = half_ranges + np.abs(cn - mids)
        
        # Soft max
        max_val = D_vals.max()
        if max_val > 100:
            return 1e6
        shifted = (D_vals - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        D_smooth = temperature * (np.log(np.sum(np.exp(shifted))) + max_val / temperature)
        
        S = (D_smooth + APOTHEM) / APOTHEM
        
        # Overlap penalty: for each pair, if distance < min_dist, add penalty
        penalty = 0.0
        min_dist = 2 * APOTHEM
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_dist:
                    penalty += (min_dist - d) ** 2 * 100
        
        return S + penalty

    # Generate initial configurations
    def generate_initial():
        """Generate a set of initial configurations to try."""
        inits = []
        
        # 1. Grid-aligned configurations (axial coordinates)
        axial_configs = [
            # Star pattern
            [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (-2, 0), (0, 2), (0, -2)],
            # 3x3 grid minus corners
            [(0, 0), (1, 0), (2, 0),
             (0, 1), (1, 1), (2, 1),
             (0, 2), (1, 2), (2, 2),
             (-1, 1), (3, 1)],
            # Compact cluster
            [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 1), (-2, 2), (-1, 2), (1, -2)],
            # Line + offset
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
             (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
             (1, 2)],
            # Two rows
            [(0, 0), (1, 0), (2, 0), (3, 0),
             (0, 1), (1, 1), (2, 1), (3, 1),
             (1, 2), (2, 2), (3, 2)],
            # Diamond
            [(0, 0),
             (1, 0), (-1, 0), (0, 1), (0, -1),
             (2, 0), (-2, 0), (0, 2), (0, -2),
             (1, 1), (-1, 1)],
        ]
        
        for axial in axial_configs:
            pos = []
            for q, r in axial:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                pos.append((x, y))
            angles = [0.0] * N
            # Center the configuration
            cx = np.mean([p[0] for p in pos])
            cy = np.mean([p[1] for p in pos])
            pos = [(p[0] - cx, p[1] - cy) for p in pos]
            params = []
            for i in range(N):
                params.extend([pos[i][0], pos[i][1], angles[i]])
            params.append(0.0)  # beta
            inits.append(params)
        
        # 2. Random configurations with some structure
        np.random.seed(42)
        for trial in range(20):
            # Start with a random compact arrangement
            pos = []
            # Place first hex at origin
            pos.append((0.0, 0.0))
            # Place remaining around it
            for i in range(1, N):
                # Random angle and distance
                ang = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(1.8, 3.5)
                x = pos[0][0] + dist * math.cos(ang)
                y = pos[0][1] + dist * math.sin(ang)
                pos.append((x, y))
            
            # Try to resolve overlaps by pushing apart
            for iteration in range(50):
                moved = False
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        d = math.sqrt(dx * dx + dy * dy)
                        min_dist = 2 * APOTHEM + 0.1
                        if d < min_dist and d > 1e-9:
                            # Push apart
                            push = (min_dist - d) / 2.0
                            nx = dx / d
                            ny = dy / d
                            pos[i] = (pos[i][0] + push * nx, pos[i][1] + push * ny)
                            pos[j] = (pos[j][0] - push * nx, pos[j][1] - push * ny)
                            moved = True
                        elif d <= 1e-9:
                            # Random nudge
                            ang = np.random.uniform(0, 2 * math.pi)
                            pos[i] = (pos[i][0] + 0.1 * math.cos(ang), pos[i][1] + 0.1 * math.sin(ang))
                            pos[j] = (pos[j][0] - 0.1 * math.cos(ang), pos[j][1] - 0.1 * math.sin(ang))
                            moved = True
                if not moved:
                    break
            
            angles = [np.random.uniform(0, math.pi / 3) for _ in range(N)]
            cx = np.mean([p[0] for p in pos])
            cy = np.mean([p[1] for p in pos])
            pos = [(p[0] - cx, p[1] - cy) for p in pos]
            
            params = []
            for i in range(N):
                params.extend([pos[i][0], pos[i][1], angles[i]])
            params.append(np.random.uniform(0, math.pi / 6))
            inits.append(params)
        
        return inits

    def optimize_from_init(init_params, temperature=0.1):
        """Optimize from an initial configuration."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None
        
        best_S = float('inf')
        best_params = None
        
        # Multi-stage temperature annealing
        temps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        current_params = init_params[:]
        
        for temp in temps:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            def obj(p, t=temp):
                return smooth_objective(p, temperature=t)
            
            res = minimize(obj, current_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < 1e5:
                current_params = res.x
                # Evaluate exact
                n = N
                pos = []
                angles_deg = []
                for i in range(n):
                    x = current_params[3 * i]
                    y = current_params[3 * i + 1]
                    a = math.degrees(current_params[3 * i + 2])
                    pos.append((x, y))
                    angles_deg.append(a)
                beta = current_params[3 * n]
                
                valid, S, center = evaluate_configuration(pos, angles_deg, beta)
                if valid and S < best_S:
                    best_S = S
                    best_params = current_params[:]
        
        return best_S, best_params

    # Try all initial configurations
    inits = generate_initial()
    
    best_global_S = float('inf')
    best_global_params = None
    
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        S, params = optimize_from_init(init)
        if S < best_global_S:
            best_global_S = S
            best_global_params = params
    
    # If no valid solution found, fall back to a simple configuration
    if best_global_params is None or best_global_S == float('inf'):
        # Use the first axial config
        axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, 0), (-2, 0), (0, 2), (0, -2)]
        pos = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pos.append((x, y))
        cx = np.mean([p[0] for p in pos])
        cy = np.mean([p[1] for p in pos])
        pos = [(p[0] - cx, p[1] - cy) for p in pos]
        
        beta = 0.0
        S, center = bounding_hex_size(pos, beta)
        best_global_S = S
        best_global_params = []
        for i in range(N):
            best_global_params.extend([pos[i][0], pos[i][1], 0.0])
        best_global_params.append(0.0)
    
    # Extract final configuration
    n = N
    inner = []
    for i in range(n):
        x = best_global_params[3 * i]
        y = best_global_params[3 * i + 1]
        a_deg = math.degrees(best_global_params[3 * i + 2])
        inner.append([float(x), float(y), float(a_deg)])
    
    beta = best_global_params[3 * n]
    
    # Compute bounding hex center and size
    pos = [(inner[i][0], inner[i][1]) for i in range(n)]
    S, center = bounding_hex_size(pos, beta)
    
    cx, cy = center
    side = float(S)
    rotation = float(math.degrees(beta))
    
    # Apply tiny scale to ensure no numerical overlap issues
    scale = 1.0 + 1e-12
    inner_scaled = []
    for i in range(n):
        x = inner[i][0]
        y = inner[i][1]
        a = inner[i][2]
        sx = cx + (x - cx) * scale
        sy = cy + (y - cy) * scale
        inner_scaled.append([float(sx), float(sy), float(a)])
    
    # Recompute size with scaled positions
    pos_scaled = [(inner_scaled[i][0], inner_scaled[i][1]) for i in range(n)]
    S_scaled, _ = bounding_hex_size(pos_scaled, beta)
    
    return inner_scaled, [float(cx), float(cy)], float(S_scaled), float(rotation)

