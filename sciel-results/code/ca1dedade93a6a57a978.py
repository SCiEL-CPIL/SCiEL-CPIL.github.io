import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 30
    bounds = 30

    def compute_all_areas(pts):
        areas = []
        for i in range(N):
            x1, y1 = pts[i]
            for j in range(i+1, N):
                x2, y2 = pts[j]
                for k in range(j+1, N):
                    x3, y3 = pts[k]
                    d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if d > 0:
                        areas.append(d)
        if not areas:
            return Counter()
        return Counter(areas)

    def get_max_count(counts):
        if not counts:
            return 0
        return max(counts.values())

    def get_best_area(counts):
        if not counts:
            return 0
        return counts.most_common(1)[0][0]

    def compute_delta(old_pt, new_pt, pts, idx):
        removed = Counter()
        added = Counter()
        x_old, y_old = old_pt
        x_new, y_new = new_pt
        others = [i for i in range(N) if i != idx]
        for a in range(len(others)):
            i = others[a]
            x1, y1 = pts[i]
            for b in range(a+1, len(others)):
                j = others[b]
                x2, y2 = pts[j]
                d_old = abs(x_old*(y1-y2) + x1*(y2-y_old) + x2*(y_old-y1))
                if d_old > 0:
                    removed[d_old] += 1
                d_new = abs(x_new*(y1-y2) + x1*(y2-y_new) + x2*(y_new-y1))
                if d_new > 0:
                    added[d_new] += 1
        return removed, added

    pts = [(i, j) for i in range(6) for j in range(5)]
    counts = compute_all_areas(pts)
    current_max_count = get_max_count(counts)
    best_max_count = current_max_count
    best_pts = pts[:]
    current_best_area = get_best_area(counts)
    best_area = current_best_area

    stagnation_count = 0
    max_stagnation = 3000

    while time.time() < deadline:
        if stagnation_count > max_stagnation:
            pts = best_pts[:]
            for _ in range(8):
                idx = random.randint(0, N-1)
                x, y = pts[idx]
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx, ny = x + dx, y + dy
                if -bounds <= nx <= bounds and -bounds <= ny <= bounds:
                    new_pt = (nx, ny)
                    if new_pt not in pts:
                        pts[idx] = new_pt
            counts = compute_all_areas(pts)
            current_max_count = get_max_count(counts)
            current_best_area = get_best_area(counts)
            if current_max_count > best_max_count:
                best_max_count = current_max_count
                best_pts = pts[:]
                best_area = current_best_area
            stagnation_count = 0
            continue

        idx = random.randint(0, N-1)
        x, y = pts[idx]
        dx = random.choice([-2, -1, 0, 1, 2])
        dy = random.choice([-2, -1, 0, 1, 2])
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
            stagnation_count += 1
            continue
        new_pt = (nx, ny)
        if new_pt in pts:
            stagnation_count += 1
            continue

        old_pt = pts[idx]
        removed, added = compute_delta(old_pt, new_pt, pts, idx)

        for area, cnt in removed.items():
            counts[area] -= cnt
            if counts[area] <= 0:
                del counts[area]
        for area, cnt in added.items():
            counts[area] += cnt

        new_max_count = get_max_count(counts)

        if new_max_count > current_max_count:
            pts[idx] = new_pt
            current_max_count = new_max_count
            current_best_area = get_best_area(counts)
            stagnation_count = 0
            if current_max_count > best_max_count:
                best_max_count = current_max_count
                best_pts = pts[:]
                best_area = current_best_area
        else:
            for area, cnt in added.items():
                counts[area] -= cnt
                if counts[area] <= 0:
                    del counts[area]
            for area, cnt in removed.items():
                counts[area] += cnt
            stagnation_count += 1

    return best_pts

# EVOLVE-BLOCK-START
# (removed - the construct function above is the only one)
# EVOLVE-BLOCK-END
