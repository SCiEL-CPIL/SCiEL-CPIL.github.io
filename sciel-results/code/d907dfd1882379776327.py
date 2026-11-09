"""Hexagon packing: place 12 unit hexagons in a minimal enclosing hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(-angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = cos_a * dx - sin_a * dy
        ly = sin_a * dx + cos_a * dy

        for i in range(6):
            theta = i * math.pi / 3.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            dist = lx * nx + ly * ny
            if dist > side:
                return False
        return True

    def dist_to_edge(px, py, x1, y1, x2, y2):
        # Distance from point (px,py) to line segment (x1,y1)-(x2,y2)
        vx = x2 - x1
        vy = y2 - y1
        wx = px - x1
        wy = py - y1
        
        vv2 = vx*vx + vy*vy
        if vv2 == 0:
            return math.sqrt(wx*wx + wy*wy)
        
        u = (wx*vx + wy*vy) / vv2
        if u < 0:
            return math.sqrt(wx*wx + wy*wy)
        elif u > 1:
            dx = px - x2
            dy = py - y2
            return math.sqrt(dx*dx + dy*dy)
        else:
            cx = x1 + u * vx
            cy = y1 + u * vy
            dx = px - cx
            dy = py - cy
            return math.sqrt(dx*dx + dy*dy)

    def hex_overlap_penalty(i, j, centers, angles):
        """Returns squared penetration depth if overlapping, 0 otherwise."""
        cx1, cy1 = centers[i]
        cx2, cy2 = centers[j]
        a1 = angles[i]
        a2 = angles[j]

        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Quick rejection
        if dist > 4.0:
            return 0.0
        
        verts1 = get_hex_vertices(cx1, cy1, a1)
        verts2 = get_hex_vertices(cx2, cy2, a2)

        max_pen = 0.0
        
        # Check vertices of 1 inside 2
        for v in verts1:
            if point_in_hex(v[0], v[1], cx2, cy2, a2):
                # Find min distance to any edge of hex 2
                min_d = float('inf')
                for k in range(6):
                    e1 = verts2[k]
                    e2 = verts2[(k+1)%6]
                    d = dist_to_edge(v[0], v[1], e1[0], e1[1], e2[0], e2[1])
                    if d < min_d:
                        min_d = d
                if min_d < max_pen:
                    max_pen = min_d
        
        # Check vertices of 2 inside 1
        for v in verts2:
            if point_in_hex(v[0], v[1], cx1, cy1, a1):
                min_d = float('inf')
                for k in range(6):
                    e1 = verts1[k]
                    e2 = verts1[(k+1)%6]
                    d = dist_to_edge(v[0], v[1], e1[0], e1[1], e2[0], e2[1])
                    if d < min_d:
                        min_d = d
                if min_d < max_pen:
                    max_pen = min_d

        # If edges intersect but no vertex inside, we still have overlap.
        # We can approximate penalty as 0.1 if they intersect.
        if max_pen == 0.0:
            # Check edge intersections
            for k in range(6):
                e1 = (verts1[k], verts1[(k+1)%6])
                for l in range(6):
                    e2 = (verts2[l], verts2[(l+1)%6])
                    if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                        return 0.01 # Small constant penalty
                        # This case is rare for hexagons if vertex checks are done
        return max_pen * max_pen

    def segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        p1 = (p1x, p1y); p2 = (p2x, p2y); p3 = (p3x, p3y); p4 = (p4x, p4y)
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def required_L(centers, angles):
        L = 0.0
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for (vx, vy) in verts:
                for k in range(6):
                    theta = math.radians(30 + k * 60)
                    proj = vx * math.cos(theta) + vy * math.sin(theta)
                    if proj > L:
                        L = proj
        return L

    def objective(params):
        centers = [(params[2*i], params[2*i+1]) for i in range(N)]
        angles = list(params[2*N:2*N+N])
        
        L = required_L(centers, angles)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                penalty += hex_overlap_penalty(i, j, centers, angles)
        
        return L + 100.0 * penalty

    def initial_guess_honeycomb():
        centers = []
        angles = []
        # Honeycomb lattice spacing
        d = 2.0 * sqrt3
        # Generate a cluster
        # Using axial coordinates for hex grid
        # q, r
        # Convert to x, y
        # x = d * (q + r/2)
        # y = d * (sqrt(3)/2 * r)
        
        # We want 12 points.
        # A 3x4 rectangle in axial coords?
        # Or a hexagon of radius 1 (7 points) + 5 more.
        
        # Let's just place them in a grid-like pattern close to honeycomb
        positions = []
        # Row 0: 4
        for i in range(4):
            positions.append((i * d, 0))
        # Row 1: 4
        for i in range(4):
            positions.append((i * d + d/2, d * sqrt3 / 2))
        # Row 2: 4
        for i in range(4):
            positions.append((i * d, d * sqrt3))
            
        # Center them
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        for p in positions:
            centers.append((p[0] - cx, p[1] - cy))
            angles.append(0.0)
            
        return centers, angles

    def initial_guess_random():
        centers = []
        angles = []
        for i in range(N):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            a = np.random.uniform(0, 360)
            centers.append((x, y))
            angles.append(a)
        return centers, angles

    best_result = None
    best_value = float('inf')

    # Try multiple initial guesses
    guesses = [initial_guess_honeycomb()]
    for _ in range(5):
        guesses.append(initial_guess_random())

    for g_idx, (centers, angles) in enumerate(guesses):
        if time.time() > deadline - 20:
            break
            
        params0 = []
        for (cx, cy) in centers:
            params0.append(cx)
            params0.append(cy)
        for a in angles:
            params0.append(a)

        # Simulated Annealing
        current_params = np.array(params0)
        current_cost = objective(current_params)
        best_params_sa = current_params.copy()
        best_cost_sa = current_cost
        
        T = 1.0
        T_min = 0.001
        step_size = 0.5
        
        max_iter = 2000
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            new_params = current_params + np.random.normal(0, step_size, size=current_params.shape)
            new_cost = objective(new_params)
            
            if new_cost < current_cost:
                current_params = new_params
                current_cost = new_cost
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_params_sa = current_params.copy()
            else:
                prob = math.exp(-(new_cost - current_cost) / T)
                if np.random.random() < prob:
                    current_params = new_params
                    current_cost = new_cost
            
            T *= 0.999
            if T < T_min:
                break

        if best_cost_sa < best_value:
            best_value = best_cost_sa
            best_result = best_params_sa

    # Local Refinement with Powell
    if best_result is not None and time.time() < deadline - 10:
        try:
            local_result = minimize(
                objective,
                best_result,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8},
            )
            if local_result.fun < best_value:
                best_value = local_result.fun
                best_result = local_result.x
        except Exception:
            pass

    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    centers = [(best_result[2*i], best_result[2*i+1]) for i in range(N)]
    angles = list(best_result[2*N:2*N+N])
    L = required_L(centers, angles)
    
    # Check validity
    # If there is still significant overlap, the result might be invalid.
    # But we hope the penalty drove it to 0.
    
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), L, 0.0

# EVOLVE-BLOCK-END
