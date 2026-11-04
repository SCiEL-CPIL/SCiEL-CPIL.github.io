import math
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def min_separation(v1, v2):
        min_sep = float('inf')
        # Check all 12 edge normals
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            projs1 = np.dot(v1, [nx, ny])
            projs2 = np.dot(v2, [nx, ny])
            min1, max1 = np.min(projs1), np.max(projs1)
            min2, max2 = np.min(projs2), np.max(projs2)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
                if min_sep < 0:
                    return min_sep
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            projs1 = np.dot(v1, [nx, ny])
            projs2 = np.dot(v2, [nx, ny])
            min1, max1 = np.min(projs1), np.max(projs1)
            min2, max2 = np.min(projs2), np.max(projs2)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
                if min_sep < 0:
                    return min_sep
        return min_sep

    def is_valid(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if min_separation(verts[i], verts[j]) < -1e-9:
                    return False
        return True

    def compute_outer_side(centers, angles, otheta_deg):
        """
        Correctly computes the minimal side length of a regular hexagon
        with rotation otheta_deg that contains all inner hexagons.
        """
        otheta_rad = math.radians(otheta_deg)
        max_width = 0.0
        
        # Precompute inner vertices for speed? 
        # 12 hexagons * 6 verts = 72 points.
        # 6 directions.
        # Total 432 dot products.
        
        all_verts = []
        for i in range(12):
            cx, cy = centers[i]
            a_rad = math.radians(angles[i])
            for vi in range(6):
                v_angle = a_rad + vi * math.pi / 3
                vx = cx + math.cos(v_angle)
                vy = cy + math.sin(v_angle)
                all_verts.append((vx, vy))
        
        all_verts = np.array(all_verts)
        
        for k in range(6):
            # Normal to the k-th edge of the outer hexagon
            # Outer hex vertices are at otheta + i*pi/3
            # Edge midpoints (normals) are at otheta + pi/6 + k*pi/3
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            min_proj = np.min(projs)
            max_proj = np.max(projs)
            width = max_proj - min_proj
            
            if width > max_width:
                max_width = width
        
        # Width = Side * sqrt(3)
        return max_width / math.sqrt(3)

    def get_penalty(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                sep = min_separation(verts[i], verts[j])
                if sep < 0:
                    # Soft penalty
                    k = 50.0
                    penalty += -math.log(1.0 + math.exp(-k * sep)) / k
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e9
        centers = []
        for i in range(12):
            centers.append([params[i * 2], params[i * 2 + 1]])
        angles = [params[24 + i] for i in range(12)]
        otheta = params[36]
        
        side = compute_outer_side(centers, angles, otheta)
        pen = get_penalty(centers, angles)
        return side + 100.0 * pen

    def generate_3x4_grid():
        # 3 rows, 4 columns of hexagons
        # Hex side 1.
        # Distance between centers in hex grid: sqrt(3)
        # Orientation: Let's align "flat" sides horizontally? 
        # Standard vertex at 0 deg: width 2 (x), height sqrt(3) (y).
        # If we want them to tile, we need specific offsets.
        
        # Let's use the standard hex grid coordinates.
        # Row r, Col c.
        # x = c * sqrt(3) + (r % 2) * sqrt(3)/2
        # y = r * 1.5
        
        centers = []
        angles = []
        for r in range(3):
            for c in range(4):
                cx = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
                cy = r * 1.5
                centers.append([cx, cy])
                angles.append(0.0) # Vertex at 0 deg
        
        # Center the grid around 0,0
        avg_x = sum(c[0] for c in centers) / 12
        avg_y = sum(c[1] for c in centers) / 12
        for c in centers:
            c[0] -= avg_x
            c[1] -= avg_y
            
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.append(0.0) # Outer angle
        return params

    def generate_flower():
        # 1 center, 6 around, 5 more
        centers = [[0.0, 0.0]]
        angles = [0.0]
        d = math.sqrt(3)
        for i in range(6):
            a = i * math.pi / 3
            centers.append([d * math.cos(a), d * math.sin(a)])
            angles.append(0.0)
        
        # Add 5 more in ring 2
        # Ring 2 distances: 2*sqrt(3) ? No, depends on packing.
        # Let's just place them roughly
        for i in range(5):
            a = i * 2 * math.pi / 5
            centers.append([2 * d * math.cos(a), 2 * d * math.sin(a)])
            angles.append(0.0)
            
        avg_x = sum(c[0] for c in centers) / 12
        avg_y = sum(c[1] for c in centers) / 12
        for c in centers:
            c[0] -= avg_x
            c[1] -= avg_y
            
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.append(0.0)
        return params

    best_params = None
    best_val = float('inf')
    
    inits = [generate_3x4_grid(), generate_flower()]
    
    # Add some random perturbations of the 3x4 grid
    import random
    random.seed(42)
    base = generate_3x4_grid()
    for _ in range(3):
        p = base.copy()
        for i in range(36): # 12*2 + 12
            if i < 24:
                p[i] += random.gauss(0, 0.05)
            else:
                p[i] += random.gauss(0, 5.0)
        inits.append(p)

    for init in inits:
        if time.time() > deadline - 20:
            break
        try:
            # Nelder-Mead is robust for this
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            angles = [res.x[24+i] for i in range(12)]
            
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = inits[0]
        best_val = objective(best_params)

    # Polish with BFGS
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='BFGS',
                           options={'maxiter': 500, 'gtol': 1e-10})
            centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            angles = [res.x[24+i] for i in range(12)]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    
    # Final angle polish
    best_otheta = best_params[36]
    best_side = best_val
    for da in np.linspace(-10, 10, 101):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    for da in np.linspace(-1, 1, 101):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
