import time
import random
import math
from collections import Counter

def construct():
    deadline = time.time() + 45.0
    random.seed(42)
    N = 30

    # Precompute all triplets
    triplets = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                triplets.append((i, j, k))

    # Precompute for each index which triplets it participates in
    idx_triplets = [[] for _ in range(N)]
    for t_idx, (i, j, k) in enumerate(triplets):
        idx_triplets[i].append(t_idx)
        idx_triplets[j].append(t_idx)
        idx_triplets[k].append(t_idx)

    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    def full_count(points):
        areas = []
        for i, j, k in triplets:
            a = doubled_area(points[i], points[j], points[k])
            if a > 0:
                areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count

    def count_target(points, target):
        count = 0
        for i, j, k in triplets:
            if doubled_area(points[i], points[j], points[k]) == target:
                count += 1
        return count

    def generate_grid_config(rows, cols, start_x=0, start_y=0):
        pts = []
        for r in range(rows):
            for c in range(cols):
                pts.append((start_x + c, start_y + r))
        return pts

    def generate_hex_config():
        # Approximate hexagonal packing with integer coordinates
        # Row 0: (0,0), (2,0), (4,0), (6,0), (8,0), (10,0)
        # Row 1: (1,1), (3,1), (5,1), (7,1), (9,1), (11,1)
        # This creates many small triangles
        pts = []
        for r in range(6):
            for c in range(5):
                x = c * 2 + (r % 2)
                y = r
                pts.append((x, y))
        # Center it
        min_x = min(p[0] for p in pts)
        min_y = min(p[1] for p in pts)
        centered = [(x - min_x - 4, y - min_y - 2) for x, y in pts]
        return centered

    def generate_triangular_config():
        # Triangular lattice subset
        # (0,0), (1,0), (2,0)...
        # (0,1), (1,1)...
        pts = []
        for r in range(6):
            for c in range(5):
                x = c
                y = r
                pts.append((x, y))
        min_x = min(p[0] for p in pts)
        min_y = min(p[1] for p in pts)
        centered = [(x - min_x - 2, y - min_y - 2) for x, y in pts]
        return centered

    initial_configs = [
        generate_grid_config(6, 5, -2, -2),
        generate_grid_config(5, 6, -2, -3),
        generate_hex_config(),
        generate_triangular_config(),
        # Diamond
        [(x, y) for x in range(-3, 4) for y in range(-3, 4) if abs(x) + abs(y) <= 4][:30],
    ]

    best_initial = None
    best_initial_count = 0

    for config in initial_configs:
        seen = set()
        filtered = []
        for p in config:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                filtered.append(p)
        while len(filtered) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                filtered.append(p)
        filtered = filtered[:30]

        counter, ba, bc = full_count(filtered)
        if bc > best_initial_count:
            best_initial_count = bc
            best_initial = filtered

    coords = list(best_initial)
    counter, target, current_count = full_count(coords)
    best_overall = current_count
    best_points = list(coords)
    best_target = target
    
    # Keep track of top areas for switching
    top_areas = [a for a, c in counter.most_common(3)]

    T = 15.0
    cooling_rate = 0.985

    no_improve = 0
    max_no_improve = 150
    restarts = 0
    max_restarts = 15

    while time.time() < deadline:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break

            # Re-evaluate current state
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
                top_areas = [a for a, c in counter.most_common(3)]

            # Switch target to a random one from top areas
            if top_areas:
                target = random.choice(top_areas)
            else:
                target = ba
            current_count = count_target(coords, target)
            no_improve = 0
            max_no_improve += 50
            T = 10.0

            # Random restart with perturbation
            for _ in range(3):
                idx = random.randint(0, N-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                if new_pt not in coords:
                    coords[idx] = new_pt
                    current_count = count_target(coords, target)
                    break

            if current_count < best_overall:
                coords = list(best_points)
                target = best_target
                current_count = best_overall
            continue

        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]

        candidates = set()
        
        # 1. Local moves
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0: continue
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    candidates.add(new_pt)
        
        # 2. Parallel moves (preserve area for some triangles)
        # Pick 2 other points
        others = [i for i in range(N) if i != idx]
        if len(others) >= 2:
            j, k = random.sample(others, 2)
            dx = coords[j][0] - coords[k][0]
            dy = coords[j][1] - coords[k][1]
            # Normalize direction
            g = math.gcd(abs(dx), abs(dy)) if dx != 0 or dy != 0 else 1
            if g > 0:
                vx = dx // g
                vy = dy // g
                for t in range(-2, 3):
                    if t == 0: continue
                    new_pt = (old_pt[0] + t * vx, old_pt[1] + t * vy)
                    if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                        candidates.add(new_pt)
        
        # 3. Random global moves
        for _ in range(3):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt != old_pt and new_pt not in coords:
                candidates.add(new_pt)

        if not candidates:
            no_improve += 1
            T *= cooling_rate
            continue

        best_delta = -999
        best_new_pt = None

        # Precompute old matches for this idx
        old_matches = 0
        for t_idx in idx_triplets[idx]:
            i, j, k = triplets[t_idx]
            if doubled_area(coords[i], coords[j], coords[k]) == target:
                old_matches += 1

        for new_pt in candidates:
            new_matches = 0
            for t_idx in idx_triplets[idx]:
                i, j, k = triplets[t_idx]
                if i == idx:
                    p1, p2, p3 = new_pt, coords[j], coords[k]
                elif j == idx:
                    p1, p2, p3 = coords[i], new_pt, coords[k]
                else:
                    p1, p2, p3 = coords[i], coords[j], new_pt
                if doubled_area(p1, p2, p3) == target:
                    new_matches += 1

            delta = new_matches - old_matches
            if delta > best_delta:
                best_delta = delta
                best_new_pt = new_pt

        if best_new_pt is not None:
            if best_delta > 0:
                accept = True
            else:
                if T > 0.1:
                    accept_prob = math.exp(best_delta / T)
                    accept = random.random() < accept_prob
                else:
                    accept = False

            if accept:
                coords[idx] = best_new_pt
                current_count += best_delta
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
            else:
                no_improve += 1

        T *= cooling_rate

        # Periodically re-evaluate target
        if no_improve > 50 and time.time() < deadline - 5:
            counter, ba, bc = full_count(coords)
            if bc > current_count:
                target = ba
                current_count = bc
                no_improve = 0
                top_areas = [a for a, c in counter.most_common(3)]

    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]

    # Final polish
    if time.time() < deadline - 2:
        for _ in range(3):
            improved = False
            for idx in range(N):
                old_pt = coords[idx]
                best_delta = 0
                best_new_pt = None
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        if dx == 0 and dy == 0: continue
                        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                        if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                            old_matches = 0
                            for t_idx in idx_triplets[idx]:
                                i, j, k = triplets[t_idx]
                                if doubled_area(coords[i], coords[j], coords[k]) == target:
                                    old_matches += 1
                            new_matches = 0
                            for t_idx in idx_triplets[idx]:
                                i, j, k = triplets[t_idx]
                                if i == idx:
                                    p1, p2, p3 = new_pt, coords[j], coords[k]
                                elif j == idx:
                                    p1, p2, p3 = coords[i], new_pt, coords[k]
                                else:
                                    p1, p2, p3 = coords[i], coords[j], new_pt
                                if doubled_area(p1, p2, p3) == target:
                                    new_matches += 1
                            delta = new_matches - old_matches
                            if delta > best_delta:
                                best_delta = delta
                                best_new_pt = new_pt
                if best_new_pt is not None and best_delta > 0:
                    coords[idx] = best_new_pt
                    current_count += best_delta
                    improved = True
                    if current_count > best_overall:
                        best_overall = current_count
                        best_points = list(coords)
            if not improved:
                break

    return best_points
