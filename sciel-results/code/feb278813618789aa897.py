import time
import random
import sys
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    N = 20
    COORD_MIN = -100
    COORD_MAX = 100

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return (0, 0, 0)
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def build_line_map(points):
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def is_valid(points):
        line_map = build_line_map(points)
        for line_pts in line_map.values():
            if len(line_pts) >= 4:
                return False
        return True

    def get_collinear_triples(points):
        line_map = build_line_map(points)
        triples = []
        for line_pts in line_map.values():
            if len(line_pts) >= 3:
                pts = sorted(line_pts)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        for k in range(j + 1, len(pts)):
                            triples.append((pts[i], pts[j], pts[k]))
        return triples

    def compute_exact_score(points):
        """
        Optimized Branch and Bound for max subset with no 3 collinear.
        Uses the insight that adding v removes k if (v, u, k) is a triple for any u in current_set.
        """
        triples = get_collinear_triples(points)
        if not triples:
            return N

        # triples_mask[i][j] is a bitmask of k such that (i, j, k) is a triple.
        # Since triples are symmetric, we can just store for i < j, but for speed,
        # let's make a 20x20 matrix of bitmasks.
        triples_mask = [[0] * N for _ in range(N)]
        for i, j, k in triples:
            triples_mask[i][j] |= (1 << k)
            triples_mask[j][i] |= (1 << k)
            triples_mask[i][k] |= (1 << j)
            triples_mask[k][i] |= (1 << j)
            triples_mask[j][k] |= (1 << i)
            triples_mask[k][j] |= (1 << i)

        best = [0]
        
        # available_mask: points that can still be added
        # current_set_mask: points already in the subset
        def backtrack(available_mask, current_set_mask, current_size):
            # Pruning
            if current_size + available_mask.bit_count() <= best[0]:
                return
            
            if available_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return

            # Heuristic: Pick a vertex v from available_mask.
            # To maximize pruning, pick v that has the most "conflicts" with current_set?
            # Or just pick the first one?
            # Let's pick the one that removes the most points from available if added.
            # This is expensive to calculate for all v.
            # Let's just pick the lowest set bit for simplicity, or a random one?
            # A good heuristic for independent set is to pick the vertex with highest degree.
            # Here, "degree" is the number of points in available that would be removed if v is added.
            # Removed = union over u in current_set of triples_mask[v][u] & available_mask.
            
            # To keep it fast, let's just pick the first available vertex.
            # Or, better: pick a vertex involved in a triple with two available vertices?
            # If we pick v, and there is a triple (v, u, w) with u, w in available,
            # then we can't pick all three.
            
            # Let's stick to a simple heuristic: pick the vertex with the most conflicts in the *current* available set.
            # Conflict of v: { k in available : exists u in current_set such that (v,u,k) triple }
            # This depends on current_set.
            
            # Alternative: Just iterate.
            temp_mask = available_mask
            best_v = -1
            max_removal = -1
            best_v_bit = 0
            
            # To save time, only evaluate a subset of candidates if N is large? N=20 is small.
            while temp_mask:
                v_bit = temp_mask & -temp_mask
                v = v_bit.bit_length() - 1
                
                # Calculate removal mask if we add v
                removal = 0
                cm = current_set_mask
                while cm:
                    u_bit = cm & -cm
                    u = u_bit.bit_length() - 1
                    removal |= triples_mask[v][u]
                    cm ^= u_bit
                
                # We remove v itself too
                removal |= v_bit
                removal &= available_mask
                
                cnt = removal.bit_count()
                if cnt > max_removal:
                    max_removal = cnt
                    best_v = v
                    best_v_bit = v_bit
                    best_removal_mask = removal
                
                temp_mask ^= v_bit

            if best_v == -1:
                return

            # Option 1: Exclude best_v
            backtrack(available_mask ^ best_v_bit, current_set_mask, current_size)
            
            # Option 2: Include best_v
            new_available = available_mask & ~best_removal_mask
            backtrack(new_available, current_set_mask | best_v_bit, current_size + 1)

        backtrack((1 << N) - 1, 0, 0)
        return best[0]

    def generate_random_points():
        pts = []
        while len(pts) < N:
            x = random.randint(COORD_MIN, COORD_MAX)
            y = random.randint(COORD_MIN, COORD_MAX)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    def generate_structured_points():
        """
        Generate points with structure to encourage collinearities (triples).
        We want to minimize the "no-3-collinear" subset size, so we want MANY triples.
        """
        pts = []
        method = random.randint(0, 2)
        if method == 0:
            # Points on a few lines
            num_lines = random.randint(3, 5)
            lines = []
            for _ in range(num_lines):
                m = random.randint(-5, 5)
                c = random.randint(-50, 50)
                lines.append((m, c))
            
            for _ in range(N):
                if random.random() < 0.8:
                    m, c = random.choice(lines)
                    x = random.randint(COORD_MIN, COORD_MAX)
                    y = m * x + c
                    if -100 <= y <= 100:
                        p = [x, y]
                        if p not in pts:
                            pts.append(p)
                while len(pts) < N:
                    x = random.randint(COORD_MIN, COORD_MAX)
                    y = random.randint(COORD_MIN, COORD_MAX)
                    p = [x, y]
                    if p not in pts:
                        pts.append(p)
                        break
        elif method == 1:
            # Grid-like
            step = random.randint(10, 20)
            start_x = random.randint(-80, 80)
            start_y = random.randint(-80, 80)
            for i in range(5):
                for j in range(4):
                    x = start_x + i * step + random.randint(-3, 3)
                    y = start_y + j * step + random.randint(-3, 3)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        p = [x, y]
                        if p not in pts:
                            pts.append(p)
            while len(pts) < N:
                x = random.randint(COORD_MIN, COORD_MAX)
                y = random.randint(COORD_MIN, COORD_MAX)
                p = [x, y]
                if p not in pts:
                    pts.append(p)
        else:
            return generate_random_points()
        return pts[:N]

    best_config = None
    best_score = N + 1

    # Initial search
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        if random.random() < 0.5:
            pts = generate_random_points()
        else:
            pts = generate_structured_points()
        if is_valid(pts):
            score = compute_exact_score(pts)
            if score < best_score:
                best_score = score
                best_config = [p[:] for p in pts]
            if best_score <= 5: # Very good
                break

    if best_config is None:
        return [[x, x * x] for x in range(-10, 10)]

    current_pts = [p[:] for p in best_config]
    current_score = best_score

    # Local Search
    temp = 5.0
    cooling = 0.995
    no_improve_steps = 0
    max_no_improve = 500

    while time.time() < deadline - 5:
        idx = random.randint(0, N - 1)
        old_x, old_y = current_pts[idx]
        new_x = random.randint(COORD_MIN, COORD_MAX)
        new_y = random.randint(COORD_MIN, COORD_MAX)

        if new_x == old_x and new_y == old_y:
            continue
        if any(current_pts[i] == [new_x, new_y] for i in range(N) if i != idx):
            continue

        current_pts[idx] = [new_x, new_y]
        if not is_valid(current_pts):
            current_pts[idx] = [old_x, old_y]
            continue

        new_score = compute_exact_score(current_pts)
        delta = new_score - current_score

        if new_score < best_score:
            best_score = new_score
            best_config = [p[:] for p in current_pts]
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            if no_improve_steps > max_no_improve:
                # Restart
                for _ in range(20):
                    if time.time() > deadline - 10: break
                    if random.random() < 0.5:
                        new_pts = generate_random_points()
                    else:
                        new_pts = generate_structured_points()
                    if is_valid(new_pts):
                        current_pts = [p[:] for p in new_pts]
                        current_score = compute_exact_score(current_pts)
                        break
                no_improve_steps = 0
                temp = 5.0
                continue

        if delta < 0:
            current_score = new_score
        else:
            if random.random() < pow(2.718, -delta / temp):
                current_score = new_score
            else:
                current_pts[idx] = [old_x, old_y]
                continue

        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return best_config

if __name__ == "__main__":
    res = construct()
    print(res)
