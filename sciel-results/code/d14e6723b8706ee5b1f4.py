"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        return [(cx + side * math.cos(angle_rad + k * math.pi / 3),
                 cy + side * math.sin(angle_rad + k * math.pi / 3)) for k in range(6)]

    def compute_exact_side(positions, outer_angle):
        """Compute exact minimum outer side length for given positions and outer rotation."""
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

    def total_overlap(positions):
        """Total pairwise overlap penetration."""
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p = sat_overlap(positions[i], positions[j])
                if p > 0:
                    pen += p
        return pen

    def objective(positions, beta, pen_weight=100.0):
        """
        Objective: Smooth outer side (log-sum-exp) + penalty for overlaps.
        """
        # Smooth side computation
        angle_rad = 0.0  # Fixed outer angle during search
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
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
            side = 0.0
        else:
            max_p = max(projs)
            exp_sum = 0.0
            for p in projs:
                exp_sum += math.exp(beta * (p - max_p))
            side = (max_p + (1.0 / beta) * math.log(exp_sum)) / inradius

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
        ring_dist = sqrt3 + 0.05
        for k in range(6):
            a = math.radians(60*k)
            positions.append([ring_dist*math.cos(a), ring_dist*math.sin(a), 0.0])
        # 4 more hexagons in the gaps
        gap_dist = 2*ring_dist*math.cos(math.pi/6) + 0.05
        for ad in [30, 150, 210, 330]:
            a = math.radians(ad)
            positions.append([gap_dist*math.cos(a), gap_dist*math.sin(a), 0.0])
        return positions

    def make_ring():
        positions = []
        positions.append([0.0, 0.0, 0.0])
        r1 = sqrt3 + 0.05
        for k in range(6):
            a = math.radians(60*k)
            positions.append([r1*math.cos(a), r1*math.sin(a), 0.0])
        r2 = 2*r1 + 0.05
        for k in range(4):
            a = math.radians(30 + 120*k)
            positions.append([r2*math.cos(a), r2*math.sin(a), 0.0])
        return positions

    def make_random():
        positions = []
        positions.append([0.0, 0.0, 0.0])
        for k in range(1, N):
            a = random.uniform(0, 2*math.pi)
            r = random.uniform(1.0, 3.5)
            positions.append([r*math.cos(a), r*math.sin(a), random.uniform(0, 60)])
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

    def separate_overlaps(positions, iters=50):
        """Iteratively push apart overlapping hexagons."""
        for it in range(iters):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    pen = sat_overlap(positions[i], positions[j])
                    if pen > 1e-7:
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < 1e-9:
                            dx, dy = 1.0, 0.0
                            d = 1.0
                        push = pen * 0.5 + 0.005
                        positions[i][0] += push * dx/d
                        positions[i][1] += push * dy/d
                        positions[j][0] -= push * dx/d
                        positions[j][1] -= push * dy/d
                        moved = True
            if not moved:
                break
        return positions

    def polish(positions, beta_final=80.0, max_iter=3000):
        """Nelder-Mead polish on the smooth objective."""
        def nm_obj(params):
            pos = from_params(params)
            return objective(pos, beta_final, pen_weight=500.0)

        params = to_params(positions)
        try:
            from scipy.optimize import minimize
            res = minimize(nm_obj, params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < objective(positions, beta_final, pen_weight=500.0):
                return from_params(res.x)
        except Exception:
            pass
        return positions

    def sa_search(positions, time_budget, temp_init=3.0, temp_min=0.05):
        """Simulated annealing with cluster moves."""
        current = [p[:] for p in positions]
        current_obj = objective(current, 1.0)
        best = [p[:] for p in current]
        best_obj = current_obj

        # Define neighbor structure for clusters
        # Use proximity-based neighbors
        def get_neighbors(pos):
            nbrs = {i: [] for i in range(N)}
            for i in range(N):
                for j in range(i+1, N):
                    d = math.sqrt((pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
                    if d < 2.5:
                        nbrs[i].append(j)
                        nbrs[j].append(i)
            return nbrs

        nbrs = get_neighbors(current)

        def get_cluster(start, max_size):
            cluster = [start]
            visited = {start}
            frontier = list(nbrs[start])
            while frontier and len(cluster) < max_size:
                if random.random() < 0.6:
                    nxt = random.choice(frontier)
                    if nxt not in visited:
                        cluster.append(nxt)
                        visited.add(nxt)
                        frontier.extend(nbrs[nxt])
                        frontier = [n for n in frontier if n not in visited]
                else:
                    break
            return cluster

        t = temp_init
        beta = 1.0
        end_time = time.time() + time_budget

        while time.time() < end_time:
            t = max(temp_min, t * 0.997)
            beta = min(50.0, beta + 0.005)

            old = [p[:] for p in current]
            move_type = random.random()

            if move_type < 0.45:
                idx = random.randint(0, N-1)
                current[idx][0] += random.gauss(0, 0.08)
                current[idx][1] += random.gauss(0, 0.08)
                current[idx][2] += random.gauss(0, 4.0)
            elif move_type < 0.75:
                idx = random.randint(0, N-1)
                current[idx][2] += random.gauss(0, 4.0)
            else:
                start_idx = random.randint(0, N-1)
                cluster = get_cluster(start_idx, 3)
                cx = sum(current[i][0] for i in cluster) / len(cluster)
                cy = sum(current[i][1] for i in cluster) / len(cluster)
                dx = random.gauss(0, 0.08)
                dy = random.gauss(0, 0.08)
                rot = random.gauss(0, 2.0)
                r_rad = math.radians(rot)
                cos_r = math.cos(r_rad)
                sin_r = math.sin(r_rad)
                for i in cluster:
                    x = current[i][0] - cx
                    y = current[i][1] - cy
                    current[i][0] = x*cos_r - y*sin_r + cx + dx
                    current[i][1] = x*sin_r + y*cos_r + cy + dy
                    current[i][2] += rot

            new_obj = objective(current, beta)

            if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / t):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best = [p[:] for p in current]
            else:
                current = old

        return best, best_obj

    # Multi-start optimization
    inits = [make_honeycomb(), make_ring()]
    for _ in range(3):
        inits.append(make_random())

    best_positions = None
    best_side = float('inf')

    for init_pos in inits:
        if time.time() > deadline - 30.0:
            break

        # Phase 1: SA search
        sa_time = min(15.0, deadline - time.time() - 30.0)
        if sa_time < 2.0:
            break
        sa_pos, _ = sa_search(init_pos, sa_time)

        # Separate overlaps
        sa_pos = separate_overlaps(sa_pos, iters=80)

        # Phase 2: Nelder-Mead polish
        if time.time() > deadline - 25.0:
            break
        polished = polish(sa_pos, beta_final=80.0, max_iter=2000)
        polished = separate_overlaps(polished, iters=50)

        # Check validity
        pen = total_overlap(polished)
        if pen > 1e-5:
            continue

        # Phase 3: Optimize outer rotation
        best_side_local = float('inf')
        best_angle = 0.0
        for ang in range(0, 31, 2):
            s = compute_exact_side(polished, float(ang))
            if s < best_side_local:
                best_side_local = s
                best_angle = float(ang)

        # Fine angle search
        for _ in range(100):
            if time.time() > deadline - 20.0:
                break
            trial = best_angle + random.gauss(0, 0.4)
            s = compute_exact_side(polished, trial)
            if s < best_side_local:
                best_side_local = s
                best_angle = trial

        if best_side_local < best_side:
            best_side = best_side_local
            best_positions = [p[:] for p in polished]

    # Fallback if nothing valid found
    if best_positions is None:
        best_positions = make_honeycomb()
        best_positions = separate_overlaps(best_positions, iters=100)
        best_side = float('inf')
        best_angle = 0.0
        for ang in range(0, 31, 3):
            s = compute_exact_side(best_positions, float(ang))
            if s < best_side:
                best_side = s
                best_angle = float(ang)
    else:
        best_angle = 0.0
        # Recompute best angle for the final positions
        best_side = float('inf')
        for ang in range(0, 31, 2):
            s = compute_exact_side(best_positions, float(ang))
            if s < best_side:
                best_side = s
                best_angle = float(ang)
        for _ in range(80):
            if time.time() > deadline - 5.0:
                break
            trial = best_angle + random.gauss(0, 0.3)
            s = compute_exact_side(best_positions, trial)
            if s < best_side:
                best_side = s
                best_angle = trial

    # Final validity check
    final_pen = total_overlap(best_positions)
    if final_pen > 1e-4:
        # Last resort: use honeycomb
        best_positions = make_honeycomb()
        best_positions = separate_overlaps(best_positions, iters=100)
        best_side = float('inf')
        best_angle = 0.0
        for ang in range(0, 31, 3):
            s = compute_exact_side(best_positions, float(ang))
            if s < best_side:
                best_side = s
                best_angle = float(ang)

    return best_positions, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
