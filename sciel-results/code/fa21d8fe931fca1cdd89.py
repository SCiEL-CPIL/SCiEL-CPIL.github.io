"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)
    R = 1.0  # circumradius of unit hexagon
    INR = sqrt3 / 2.0  # inradius

    def hex_verts(cx, cy, ang_deg):
        a = math.radians(ang_deg)
        return [(cx + R * math.cos(a + i * math.pi / 3),
                 cy + R * math.sin(a + i * math.pi / 3)) for i in range(6)]

    def hex_normals(ang_deg):
        a = math.radians(ang_deg)
        return [(math.cos(a + i * math.pi / 3 + math.pi / 6),
                 math.sin(a + i * math.pi / 3 + math.pi / 6)) for i in range(6)]

    def is_disjoint(v1, v2, n1, n2):
        for nx, ny in n1 + n2:
            p1 = [vx * nx + vy * ny for vx, vy in v1]
            p2 = [vx * nx + vy * ny for vx, vy in v2]
            if max(p1) < min(p2) - 1e-12 or max(p2) < min(p1) - 1e-12:
                return True
        return False

    def outer_side(hex_list, oa_deg):
        oa = math.radians(oa_deg)
        max_p = 0.0
        for verts in hex_list:
            for vx, vy in verts:
                for i in range(6):
                    nx = math.cos(oa + i * math.pi / 3 + math.pi / 6)
                    ny = math.sin(oa + i * math.pi / 3 + math.pi / 6)
                    p = abs(vx * nx + vy * ny)
                    if p > max_p:
                        max_p = p
        return 2.0 * max_p / sqrt3

    def axial_to_cart(q, r, s=1.0):
        return (sqrt3 * s * (q + r / 2.0), 1.5 * s * r)

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """
        Compute minimum outer side to contain all inner hexagons.
        The outer hexagon has center (cx, cy), side length S, and rotation outer_angle.
        """
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hex_verts(ix, iy, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    required_side = proj / INR
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def relax(params, max_iter=5000, dt=0.05, damping=0.92):
        """Physics-based repulsion relaxation."""
        p = params.copy()
        for it in range(max_iter):
            if time.time() > deadline - 20.0:
                break

            # Centroid shift
            cx_avg = np.mean(p[0::3])
            cy_avg = np.mean(p[1::3])
            for i in range(N):
                p[i*3] -= cx_avg
                p[i*3+1] -= cy_avg

            forces = np.zeros_like(p)
            verts = []
            norms = []
            for i in range(N):
                cx, cy, a = p[i*3], p[i*3+1], p[i*3+2]
                verts.append(hex_verts(cx, cy, a))
                norms.append(hex_normals(a))

            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(verts[i], verts[j], norms[i], norms[j]):
                        dx = p[i*3] - p[j*3]
                        dy = p[i*3+1] - p[j*3+1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            dx = random.uniform(-0.1, 0.1)
                            dy = random.uniform(-0.1, 0.1)
                            dist = math.sqrt(dx*dx + dy*dy)
                        pen = 2 * R - dist
                        if pen > 0:
                            fx = pen * dx / dist
                            fy = pen * dy / dist
                            forces[i*3] += fx
                            forces[i*3+1] += fy
                            forces[j*3] -= fx
                            forces[j*3+1] -= fy

            # Wall constraint
            hex_list = [hex_verts(p[i*3], p[i*3+1], p[i*3+2]) for i in range(N)]
            side = outer_side(hex_list, 0.0)
            target = side * 0.999
            for i in range(N):
                cx, cy, a = p[i*3], p[i*3+1], p[i*3+2]
                for vx, vy in hex_verts(cx, cy, a):
                    for k in range(6):
                        nx = math.cos(k * math.pi / 3 + math.pi / 6)
                        ny = math.sin(k * math.pi / 3 + math.pi / 6)
                        proj = vx * nx + vy * ny
                        limit = target * INR
                        if proj > limit:
                            forces[i*3] -= (proj - limit) * nx * 0.5
                            forces[i*3+1] -= (proj - limit) * ny * 0.5
                        elif proj < -limit:
                            forces[i*3] += (-limit - proj) * nx * 0.5
                            forces[i*3+1] += (-limit - proj) * ny * 0.5

            # Update
            for i in range(N):
                p[i*3] += dt * forces[i*3] * damping
                p[i*3+1] += dt * forces[i*3+1] * damping
                if it % 50 == 0:
                    p[i*3+2] += random.gauss(0, 1.0)
                p[i*3+2] = p[i*3+2] % 60

        return p

    def refine(params, max_iter=2000):
        """Finer relaxation with smaller dt."""
        p = params.copy()
        dt = 0.01
        damping = 0.95
        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break

            cx_avg = np.mean(p[0::3])
            cy_avg = np.mean(p[1::3])
            for i in range(N):
                p[i*3] -= cx_avg
                p[i*3+1] -= cy_avg

            forces = np.zeros_like(p)
            verts = []
            norms = []
            for i in range(N):
                cx, cy, a = p[i*3], p[i*3+1], p[i*3+2]
                verts.append(hex_verts(cx, cy, a))
                norms.append(hex_normals(a))

            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(verts[i], verts[j], norms[i], norms[j]):
                        dx = p[i*3] - p[j*3]
                        dy = p[i*3+1] - p[j*3+1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            dx = random.uniform(-0.01, 0.01)
                            dy = random.uniform(-0.01, 0.01)
                            dist = math.sqrt(dx*dx + dy*dy)
                        pen = 2 * R - dist
                        if pen > 0:
                            fx = pen * dx / dist
                            fy = pen * dy / dist
                            forces[i*3] += fx
                            forces[i*3+1] += fy
                            forces[j*3] -= fx
                            forces[j*3+1] -= fy

            hex_list = [hex_verts(p[i*3], p[i*3+1], p[i*3+2]) for i in range(N)]
            side = outer_side(hex_list, 0.0)
            target = side * 0.9995
            for i in range(N):
                cx, cy, a = p[i*3], p[i*3+1], p[i*3+2]
                for vx, vy in hex_verts(cx, cy, a):
                    for k in range(6):
                        nx = math.cos(k * math.pi / 3 + math.pi / 6)
                        ny = math.sin(k * math.pi / 3 + math.pi / 6)
                        proj = vx * nx + vy * ny
                        limit = target * INR
                        if proj > limit:
                            forces[i*3] -= (proj - limit) * nx * 0.3
                            forces[i*3+1] -= (proj - limit) * ny * 0.3
                        elif proj < -limit:
                            forces[i*3] += (-limit - proj) * nx * 0.3
                            forces[i*3+1] += (-limit - proj) * ny * 0.3

            for i in range(N):
                p[i*3] += dt * forces[i*3] * damping
                p[i*3+1] += dt * forces[i*3+1] * damping
                p[i*3+2] = p[i*3+2] % 60

        return p

    def params_to_positions(params):
        positions = []
        for i in range(N):
            positions.append([params[i*3], params[i*3+1], params[i*3+2]])
        return positions

    def positions_to_params(positions):
        params = []
        for x, y, a in positions:
            params.extend([x, y, a])
        return np.array(params)

    # Diverse initial topologies
    topologies = [
        # Honeycomb-ish
        [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)],
        # Ring + center
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)],
        # Line + offsets
        [(0,0),(1,0),(-1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(-1,-1),(2,0),(-2,0)],
        # Compact cluster
        [(0,0),(1,0),(-1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(-1,-1),(2,0),(-2,0)],
        # Star pattern
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(2,-1),(-2,1),(-2,-1)],
    ]

    def init_params_from_topology(coords):
        params = []
        for q, r in coords:
            x, y = axial_to_cart(q, r)
            params.extend([x, y, 0.0])
        return np.array(params)

    def random_init():
        params = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            params.extend([x, y, a])
        return np.array(params)

    best_side = float('inf')
    best_positions = None
    best_angle = 0.0

    # Try multiple initial configurations
    all_inits = []
    for coords in topologies:
        all_inits.append(init_params_from_topology(coords))
    for _ in range(5):
        all_inits.append(random_init())

    for init_params in all_inits:
        if time.time() > deadline - 30.0:
            break

        # Coarse relaxation
        p1 = relax(init_params, max_iter=3000, dt=0.05, damping=0.92)
        # Fine refinement
        p2 = refine(p1, max_iter=2000)

        positions = params_to_positions(p2)
        
        # Verify no overlap
        valid = True
        verts = [hex_verts(p[0], p[1], p[2]) for p in positions]
        norms = [hex_normals(p[2]) for p in positions]
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(verts[i], verts[j], norms[i], norms[j]):
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        # Try different outer angles
        for outer_angle in range(0, 60, 5):
            side = compute_min_side(positions, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_positions = [list(p) for p in positions]
                best_angle = outer_angle

    # If no valid packing found, use fallback
    if best_positions is None:
        # Absolute fallback: spread hexagons in a ring
        best_positions = []
        for i in range(N):
            if i == 0:
                best_positions.append([0.0, 0.0, 0.0])
            else:
                r = 2.5 * (i // 6 + 1)
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                best_positions.append([x, y, 0.0])
        best_side = compute_min_side(best_positions, (0.0, 0.0), 0.0)
        best_angle = 0.0

    # Final verification
    if best_positions is not None:
        verts = [hex_verts(p[0], p[1], p[2]) for p in best_positions]
        norms = [hex_normals(p[2]) for p in best_positions]
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(verts[i], verts[j], norms[i], norms[j]):
                    # Fallback to safe configuration
                    best_positions = []
                    for i in range(N):
                        if i == 0:
                            best_positions.append([0.0, 0.0, 0.0])
                        else:
                            r = 2.5 * (i // 6 + 1)
                            angle = math.radians(60 * (i % 6))
                            x = r * math.cos(angle)
                            y = r * math.sin(angle)
                            best_positions.append([x, y, 0.0])
                    best_side = compute_min_side(best_positions, (0.0, 0.0), 0.0)
                    best_angle = 0.0
                    break

    return best_positions, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
