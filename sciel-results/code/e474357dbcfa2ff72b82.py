"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0  # inradius of unit hexagon

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        return [(cx + side * math.cos(angle_rad + k * math.pi / 3),
                 cy + side * math.sin(angle_rad + k * math.pi / 3)) for k in range(6)]

    def compute_min_side(inner_positions, outer_angle):
        """Compute minimum outer side length to contain all inner hexagons.
        Outer hexagon centered at origin with given rotation."""
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]

        for ix, iy, iang in inner_positions:
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

    def total_overlap_penalty(positions):
        """Total pairwise overlap penalty."""
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen += sat_overlap(positions[i], positions[j])
        return pen

    def objective(params, outer_angle=0.0):
        """Objective: outer side + penalty for overlaps."""
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        side = compute_min_side(positions, outer_angle)
        pen = total_overlap_penalty(positions)
        return side + 1000.0 * pen * pen

    def make_honeycomb():
        """Standard honeycomb: 1 center + 6 ring + 4 outer."""
        positions = []
        positions.append([0.0, 0.0, 0.0])
        ring_dist = sqrt3 + 0.05
        for k in range(6):
            a = math.radians(60*k)
            positions.append([ring_dist*math.cos(a), ring_dist*math.sin(a), 0.0])
        gap_dist = 2*ring_dist*math.cos(math.pi/6) + 0.05
        for ad in [30, 150, 210, 330]:
            a = math.radians(ad)
            positions.append([gap_dist*math.cos(a), gap_dist*math.sin(a), 0.0])
        return positions

    def make_tight_ring():
        """Tighter ring configuration."""
        positions = []
        positions.append([0.0, 0.0, 0.0])
        for k in range(6):
            a = math.radians(60*k)
            positions.append([sqrt3*math.cos(a), sqrt3*math.sin(a), 0.0])
        for k in range(4):
            a = math.radians(30 + 120*k)
            r = 2*sqrt3*math.cos(math.pi/6) - 0.1
            positions.append([r*math.cos(a), r*math.sin(a), 0.0])
        return positions

    def make_random(scale=3.0):
        positions = []
        positions.append([0.0, 0.0, 0.0])
        for i in range(N-1):
            r = random.uniform(0.5, scale)
            a = random.uniform(0, 2*math.pi)
            positions.append([r*math.cos(a), r*math.sin(a), random.uniform(-30, 30)])
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

    def nelder_mead_optimize(init_positions, max_iter=2000, outer_angle=0.0):
        """Run Nelder-Mead on a single initialization."""
        try:
            from scipy.optimize import minimize
            params = to_params(init_positions)
            res = minimize(objective, params, method='Nelder-Mead',
                          args=(outer_angle,),
                          options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-7,
                                   'adaptive': True})
            return from_params(res.x), res.fun
        except Exception:
            return init_positions, objective(to_params(init_positions), outer_angle)

    def simulated_annealing(positions, steps=5000, temp_start=1.0, temp_end=0.01):
        """Simple simulated annealing."""
        current = to_params(positions)
        curr_obj = objective(current)
        best = current.copy()
        best_obj = curr_obj
        temp = temp_start
        cooling = (temp_end / temp_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline - 15.0:
                break
            idx = random.randint(0, N-1)
            old = current[3*idx:3*idx+3].copy()
            current[3*idx] += random.gauss(0, 0.05)
            current[3*idx+1] += random.gauss(0, 0.05)
            current[3*idx+2] += random.gauss(0, 2.0)
            new_obj = objective(current)
            if new_obj < curr_obj or random.random() < math.exp(-(new_obj - curr_obj)/temp):
                curr_obj = new_obj
                if curr_obj < best_obj:
                    best_obj = curr_obj
                    best = current.copy()
            else:
                current[3*idx:3*idx+3] = old
            temp *= cooling
        
        return from_params(best), best_obj

    def fix_overlaps(positions):
        """Iteratively separate overlapping pairs."""
        for it in range(100):
            if time.time() > deadline - 10.0:
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
        return positions

    # Multi-start optimization
    inits = [make_honeycomb(), make_tight_ring()]
    for _ in range(6):
        inits.append(make_random(random.uniform(2.0, 4.5)))

    best_positions = None
    best_obj = float('inf')

    # First pass: Nelder-Mead on each initialization
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        positions, obj = nelder_mead_optimize(init, max_iter=1500)
        if obj < best_obj:
            best_obj = obj
            best_positions = positions

    # Second pass: SA from best and a few other starts
    if best_positions is not None:
        if time.time() < deadline - 30.0:
            sa_pos, sa_obj = simulated_annealing(best_positions, steps=8000)
            if sa_obj < best_obj:
                best_obj = sa_obj
                best_positions = sa_pos

    # Additional SA from random starts if time permits
    for _ in range(3):
        if time.time() > deadline - 25.0:
            break
        rand_pos = make_random(random.uniform(2.0, 4.0))
        sa_pos, sa_obj = simulated_annealing(rand_pos, steps=5000)
        if sa_obj < best_obj:
            best_obj = sa_obj
            best_positions = sa_pos

    # Fix any remaining overlaps
    positions = fix_overlaps(best_positions)

    # Optimize outer rotation
    best_side = float('inf')
    best_angle = 0.0
    for ang in range(0, 31, 1):
        s = compute_min_side(positions, float(ang))
        if s < best_side:
            best_side = s
            best_angle = float(ang)
    
    # Fine-tune angle
    for _ in range(200):
        if time.time() > deadline - 3.0:
            break
        trial = best_angle + random.gauss(0, 0.3)
        s = compute_min_side(positions, trial)
        if s < best_side:
            best_side = s
            best_angle = trial

    # Final overlap check
    final_pen = total_overlap_penalty(positions)
    if final_pen > 1e-4:
        # Safe fallback: well-separated configuration
        safe = [[0.0, 0.0, 0.0]]
        for i in range(1, N):
            ring = (i + 5) // 6
            idx = i % 6
            if ring == 1:
                r = sqrt3 + 0.1
                a = math.radians(60 * idx)
            else:
                r = 2*sqrt3 + 0.3
                a = math.radians(60 * idx + 30)
            safe.append([r*math.cos(a), r*math.sin(a), 0.0])
        positions = safe
        best_side = float('inf')
        best_angle = 0.0
        for ang in range(0, 31, 2):
            s = compute_min_side(positions, float(ang))
            if s < best_side:
                best_side = s
                best_angle = float(ang)

    return positions, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
