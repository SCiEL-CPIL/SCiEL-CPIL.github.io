"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        return [(cx + side * math.cos(angle_rad + k * math.pi / 3),
                 cy + side * math.sin(angle_rad + k * math.pi / 3)) for k in range(6)]

    def compute_min_side_smooth(positions, outer_angle, beta):
        """
        Compute a smooth approximation of the minimum outer side length.
        Uses log-sum-exp on the projections of all vertices onto the 6 normal axes.
        """
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Normals of the outer hexagon (apothem directions)
        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]

        projs = []
        for ix, iy, iang in positions:
            iv = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in iv:
                rx = vx * cos_a - vy * sin_a
                ry = vx * sin_a + vy * cos_a
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    projs.append(proj)
        
        if not projs:
            return 0.0
            
        max_p = max(projs)
        exp_sum = 0.0
        for p in projs:
            exp_sum += math.exp(beta * (p - max_p))
        
        smooth_side = max_p + (1.0 / beta) * math.log(exp_sum)
        return smooth_side / inradius

    def sat_overlap(p1, p2):
        """Return penetration depth > 0 if overlapping, 0 if disjoint."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d >= 2.0:
            return 0.0
        if d < 1e-9:
            return 2.0

        v1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        v2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        axes = []
        for verts in (v1, v2):
            for i in range(6):
                x1, y1 = verts[i]
                x2, y2 = verts[(i+1) % 6]
                ex, ey = x2-x1, y2-y1
                L = math.sqrt(ex*ex + ey*ey)
                if L > 1e-12:
                    axes.append((-ey/L, ex/L))
        axes.append((dx/d, dy/d))

        min_gap = float('inf')
        for nx, ny in axes:
            p1min = min(vx*nx+vy*ny for vx,vy in v1)
            p1max = max(vx*nx+vy*ny for vx,vy in v1)
            p2min = min(vx*nx+vy*ny for vx,vy in v2)
            p2max = max(vx*nx+vy*ny for vx,vy in v2)
            gap = max(p2min - p1max, p1min - p2max)
            if gap > 0:
                return 0.0
            min_gap = min(min_gap, -gap)
        return min_gap if min_gap < float('inf') else 0.0

    def objective(positions, outer_angle, beta, pen_weight=100.0):
        """
        Objective: Smooth outer side + penalty for overlaps.
        """
        side = compute_min_side_smooth(positions, outer_angle, beta)
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p = sat_overlap(positions[i], positions[j])
                if p > 0:
                    pen += p * p
        return side + pen_weight * pen

    def make_honeycomb():
        positions = []
        positions.append([0.0, 0.0, 0.0])
        ring_dist = sqrt3 + 0.1
        for k in range(6):
            a = math.radians(60*k)
            positions.append([ring_dist*math.cos(a), ring_dist*math.sin(a), 0.0])
        gap_dist = 2*ring_dist*math.cos(math.pi/6) + 0.1
        for ad in [30, 150, 210, 330]:
            a = math.radians(ad)
            positions.append([gap_dist*math.cos(a), gap_dist*math.sin(a), 0.0])
        return positions

    def to_params(positions):
        params = []
        for p in positions:
            params.extend([p[0], p[1], p[2]])
        return np.array(params)

    def from_params(params):
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        return positions

    # --- Simulated Annealing with Rigid Clusters ---
    
    # Initialize
    current_positions = make_honeycomb()
    current_obj = objective(current_positions, 0.0, 1.0)
    
    best_positions = [p[:] for p in current_positions]
    best_obj = current_obj
    
    # Define clusters based on initial honeycomb neighbors
    bonds = []
    for i in range(1, 7):
        bonds.append((0, i))
    for i in range(1, 7):
        bonds.append((i, (i % 6) + 1))
    bonds.append((7, 1))
    bonds.append((7, 2))
    bonds.append((8, 3))
    bonds.append((8, 4))
    bonds.append((9, 4))
    bonds.append((9, 5))
    bonds.append((10, 6))
    bonds.append((10, 1))
    
    neighbors = {i: [] for i in range(N)}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
        
    def get_cluster(start_idx, max_size):
        """Get a random connected cluster of size <= max_size starting from start_idx."""
        cluster = [start_idx]
        visited = {start_idx}
        frontier = neighbors[start_idx][:]
        
        while frontier and len(cluster) < max_size:
            if random.random() < 0.7:
                next_idx = random.choice(frontier)
                if next_idx not in visited:
                    cluster.append(next_idx)
                    visited.add(next_idx)
                    frontier.extend(neighbors[next_idx])
                    frontier = [n for n in frontier if n not in visited]
            else:
                break
        return cluster

    temp = 5.0
    beta = 1.0
    steps = 0
    max_steps = 20000
    
    while steps < max_steps and time.time() < deadline - 20.0:
        steps += 1
        
        temp *= 0.995
        beta = min(50.0, beta + 0.01)
        
        move_type = random.random()
        old_positions = [p[:] for p in current_positions]
        
        if move_type < 0.5:
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            current_positions[idx][0] += dx
            current_positions[idx][1] += dy
            current_positions[idx][2] += random.gauss(0, 5.0)
            
        elif move_type < 0.8:
            idx = random.randint(0, N-1)
            current_positions[idx][2] += random.gauss(0, 5.0)
            
        else:
            start_idx = random.randint(0, N-1)
            cluster = get_cluster(start_idx, 3)
            cx = sum(current_positions[i][0] for i in cluster) / len(cluster)
            cy = sum(current_positions[i][1] for i in cluster) / len(cluster)
            
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            rot = random.gauss(0, 2.0)
            
            for i in cluster:
                current_positions[i][0] += dx
                current_positions[i][1] += dy
                x = current_positions[i][0] - cx
                y = current_positions[i][1] - cy
                r_rad = math.radians(rot)
                cos_r = math.cos(r_rad)
                sin_r = math.sin(r_rad)
                new_x = x * cos_r - y * sin_r + cx + dx
                new_y = x * sin_r + y * cos_r + cy + dy
                current_positions[i][0] = new_x
                current_positions[i][1] = new_y
                current_positions[i][2] += rot

        new_obj = objective(current_positions, 0.0, beta)
        
        if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / temp):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_positions = [p[:] for p in current_positions]
        else:
            current_positions = old_positions
            
    # --- Polish with Nelder-Mead ---
    beta_final = 100.0
    
    def nm_objective(params):
        positions = from_params(params)
        return objective(positions, 0.0, beta_final, pen_weight=1000.0)
        
    params = to_params(best_positions)
    
    try:
        from scipy.optimize import minimize
        res = minimize(nm_objective, params, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
        if res.fun < best_obj:
            best_obj = res.fun
            best_positions = from_params(res.x)
    except Exception:
        pass

    # --- Final Overlap Fix and Side Calculation ---
    positions = best_positions
    
    # Iteratively separate overlaps
    for it in range(100):
        if time.time() > deadline - 5.0:
            break
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                pen = sat_overlap(positions[i], positions[j])
                if pen > 1e-6:
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < 1e-9:
                        dx, dy = 1.0, 0.0
                        d = 1.0
                    push = pen * 0.5 + 0.01
                    positions[i][0] += push * dx/d
                    positions[i][1] += push * dy/d
                    positions[j][0] -= push * dx/d
                    positions[j][1] -= push * dy/d
                    moved = True
        if not moved:
            break
            
    # Calculate exact side length and optimize outer rotation
    def compute_exact_side(positions, outer_angle):
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]
        max_side = 0.0
        for ix, iy, iang in positions:
            iv = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in iv:
                rx = vx * cos_a - vy * sin_a
                ry = vx * sin_a + vy * cos_a
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    req = proj / inradius
                    if req > max_side:
                        max_side = req
        return max_side

    best_side = float('inf')
    best_angle = 0.0
    
    # Coarse search for angle
    for ang in range(0, 31, 2):
        s = compute_exact_side(positions, float(ang))
        if s < best_side:
            best_side = s
            best_angle = float(ang)
            
    # Fine search
    for _ in range(200):
        if time.time() > deadline - 2.0:
            break
        trial = best_angle + random.gauss(0, 0.5)
        s = compute_exact_side(positions, trial)
        if s < best_side:
            best_side = s
            best_angle = trial

    # Final validity check
    final_pen = sum(sat_overlap(positions[i], positions[j]) for i in range(N) for j in range(i+1, N))
    if final_pen > 1e-4:
        safe = make_honeycomb()
        best_side = float('inf')
        best_angle = 0.0
        for ang in range(0, 31, 3):
            s = compute_exact_side(safe, float(ang))
            if s < best_side:
                best_side = s
                best_angle = float(ang)
        positions = safe

    return positions, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
