import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 148.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Hexagon geometry constants
    # Inner hexagons are unit side length.
    # We model them as circles of radius R_circ = sqrt(3) (circumradius).
    # If these circles are disjoint, the hexagons are definitely disjoint (no overlap).
    # If the circles are inside the outer hexagon, the hexagons are inside the outer hexagon.
    # This is a conservative (safe) packing strategy that avoids complex polygon intersection checks.
    R_circ = math.sqrt(3)
    apothem_factor = math.sqrt(3) / 2

    def compute_outer_hex(circles, ocx, ocy, oangle):
        """Find minimum S such that all circles fit in outer hexagon."""
        a = math.radians(oangle)
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for cx, cy in circles:
                # The projection of the circle's furthest point in direction (nx, ny) is proj + R
                proj = (cx - ocx) * nx + (cy - ocy) * ny
                if proj + R_circ > max_proj:
                    max_proj = proj + R_circ
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_circles_disjoint(circles):
        """Check if all circles are mutually disjoint."""
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (2 * R_circ - 1e-6) ** 2:
                    return False
        return True

    def local_search(circles, ocx, ocy, oangle, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        rng = random.Random(seed)
        cur_circles = [list(c) for c in circles]
        cur_S = compute_outer_hex(cur_circles, ocx, ocy, oangle)
        
        # Initial validity check
        if not all_circles_disjoint(cur_circles):
            # Try to fix by pushing apart
            for _ in range(100):
                # Find a collision and push
                collided = False
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = cur_circles[i][0] - cur_circles[j][0]
                        dy = cur_circles[i][1] - cur_circles[j][1]
                        dist_sq = dx * dx + dy * dy
                        min_dist = 2 * R_circ
                        if dist_sq < min_dist ** 2:
                            collided = True
                            if dist_sq < 1e-6:
                                dx = rng.gauss(0, 0.1)
                                dy = rng.gauss(0, 0.1)
                            dist = math.sqrt(dist_sq)
                            push = (min_dist - dist) / 2 + 0.01
                            cur_circles[i][0] += dx / dist * push
                            cur_circles[i][1] += dy / dist * push
                            cur_circles[j][0] -= dx / dist * push
                            cur_circles[j][1] -= dy / dist * push
                if not collided:
                    break
            
            if not all_circles_disjoint(cur_circles):
                return

        cur_valid = True
        if cur_valid and cur_S < best_S:
            best_S = cur_S
            best_inner = [[c[0], c[1], 0.0] for c in cur_circles] # Angle 0 is fine for the circle model
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True

        for it in range(3000):
            if time.time() > deadline:
                break
            old = [list(c) for c in cur_circles]
            idx = rng.randrange(N)
            cur_circles[idx][0] += rng.gauss(0, 0.2)
            cur_circles[idx][1] += rng.gauss(0, 0.2)
            
            new_valid = all_circles_disjoint(cur_circles)
            if new_valid:
                new_S = compute_outer_hex(cur_circles, ocx, ocy, oangle)
                if new_S < cur_S:
                    cur_S = new_S
                    if new_S < best_S:
                        best_S = new_S
                        best_inner = [[c[0], c[1], 0.0] for c in cur_circles]
                        best_center = (ocx, ocy)
                        best_angle = oangle
                        best_valid = True
                elif rng.random() < 0.05:
                    cur_S = new_S
            else:
                cur_circles = old

    # Generate initial configurations
    initial_configs = []
    
    # Configuration 1: Hexagonal lattice (central + 6 + 4)
    sp = 2 * R_circ + 0.1  # Slightly larger than touching to allow local search to start
    hex_config = [[0.0, 0.0]]
    for i in range(6):
        theta = math.radians(60 * i)
        hex_config.append([sp * math.cos(theta), sp * math.sin(theta)])
    # Add 4 more to make 11
    # 2nd ring has 12 positions, we pick 4
    sp2 = sp * 2
    for i in [0, 2, 4, 6]:
        if len(hex_config) >= N: break
        theta = math.radians(60 * i + 30)
        hex_config.append([sp2 * math.cos(theta), sp2 * math.sin(theta)])
    initial_configs.append(hex_config[:N])
    
    # Configuration 2: Linear
    lin_config = []
    for k in range(N):
        lin_config.append([(k - N // 2) * sp, 0.0])
    initial_configs.append(lin_config)
    
    # Configuration 3: Random clusters
    for s in range(50):
        rng_init = random.Random(s)
        cfg = []
        for _ in range(N):
            x = rng_init.uniform(-3, 3)
            y = rng_init.uniform(-3, 3)
            cfg.append([x, y])
        initial_configs.append(cfg)

    for i, cfg in enumerate(initial_configs):
        if time.time() > deadline:
            break
        local_search(cfg, 0.0, 0.0, 0.0, i * 1000 + 42)

    if not best_valid or best_inner is None:
        # Fallback: use a simple valid configuration
        sp = 2 * R_circ + 0.1
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
