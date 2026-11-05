import time
import random
import itertools
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def _calculate_max_multiplicity(points):
        n = len(points)
        if n < 3:
            return 0, 0
        areas = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        max_area = [a for a, m in areas.items() if m == max_mult][0]
        return max_mult, max_area

    def _calc_delta(points, idx, old_pos, new_pos, target_area):
        delta = 0
        x1, y1 = new_pos
        ox, oy = old_pos
        n = len(points)
        for j in range(n):
            if j == idx: continue
            x2, y2 = points[j]
            for k in range(j + 1, n):
                if k == idx: continue
                x3, y3 = points[k]
                det_new = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det_new == target_area:
                    delta += 1
                det_old = abs((x2 - ox) * (y3 - oy) - (y2 - oy) * (x3 - ox))
                if det_old == target_area:
                    delta -= 1
        return delta

    # Generate diverse initial candidates
    candidates = []
    random.seed(42)

    # A. Rectangular Grids
    for w in range(1, 31):
        for h in range(1, 31):
            if w * h != 30: continue
            pts = [(x, y) for x in range(w) for y in range(h)]
            if len(pts) == 30 and len(set(pts)) == 30:
                cx, cy = w // 2, h // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)

    # B. Sheared Grids
    for w in range(1, 31):
        for h in range(1, 31):
            if w * h != 30: continue
            for k in [-3, -2, -1, 0, 1, 2, 3]:
                pts = []
                valid = True
                for x in range(w):
                    for y in range(h):
                        px = x + k * y
                        py = y
                        pts.append((px, py))
                if len(set(pts)) != 30:
                    valid = False
                else:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    cx = (min(xs) + max(xs)) // 2
                    cy = (min(ys) + max(ys)) // 2
                    pts_c = [(x - cx, y - cy) for x, y in pts]
                    if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts_c):
                        candidates.append(pts_c)
                    else:
                        valid = False

    # C. Staggered patterns
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10)]:
        pts = []
        for y in range(h):
            off = y % 2
            for x in range(w):
                px = x + off
                py = y
                pts.append((px, py))
        if len(set(pts)) == 30:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            pts_c = [(x - cx, y - cy) for x, y in pts]
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts_c):
                candidates.append(pts_c)

    # D. Random points
    for _ in range(80):
        pts = set()
        while len(pts) < 30:
            pts.add((random.randint(-15, 15), random.randint(-15, 15)))
        candidates.append(list(pts))

    # Evaluate all candidates
    scored_candidates = []
    for pts in candidates:
        score, area = _calculate_max_multiplicity(pts)
        scored_candidates.append((score, area, pts))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    current_best_score = scored_candidates[0][0] if scored_candidates else 0
    current_best_points = scored_candidates[0][2] if scored_candidates else [(0,0)]
    current_best_area = scored_candidates[0][1] if scored_candidates else 1

    # Local search with multiple restarts and adaptive neighborhood
    for start_score, start_area, start_pts in scored_candidates[:8]:
        if time.time() > deadline - 5:
            break

        current_points = list(start_pts)
        current_area = start_area
        current_score = start_score

        no_improve_count = 0
        max_no_improve = 300

        while time.time() < deadline - 2:
            if no_improve_count > max_no_improve:
                break

            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]

            best_move = None
            best_delta = 0

            # Adaptive neighborhood size
            neighborhood = 3 if no_improve_count < 50 else 2
            for dx in range(-neighborhood, neighborhood + 1):
                for dy in range(-neighborhood, neighborhood + 1):
                    if dx == 0 and dy == 0: continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_pos = (new_x, new_y)
                    if new_pos in current_points:
                        continue
                    delta = _calc_delta(current_points, idx, (old_x, old_y), new_pos, current_area)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = new_pos

            if best_move:
                current_points[idx] = best_move
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count % 50 == 0:
                    score, area = _calculate_max_multiplicity(current_points)
                    if score > current_score:
                        current_score = score
                        current_area = area
                        no_improve_count = 0

        final_score, final_area = _calculate_max_multiplicity(current_points)
        if final_score > current_best_score:
            current_best_score = final_score
            current_best_points = current_points
            current_best_area = final_area

    return current_best_points
