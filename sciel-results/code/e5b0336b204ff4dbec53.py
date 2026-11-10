# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
import random

_DEADLINE_MARGIN = 30.0


def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidents(points, lines):
    if len(points) == 0 or len(lines) == 0:
        return 0
    x = points[:, 0]
    y = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    val = np.outer(A, x) + np.outer(B, y) + C[:, None]
    return int(np.count_nonzero(val == 0))


def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points.astype(float))):
        return False
    if not np.all(np.isfinite(lines.astype(float))):
        return False
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    for i in range(27):
        if lines[i, 0] == 0 and lines[i, 1] == 0:
            return False
    keys = set()
    for i in range(27):
        k = _normalize_line(int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2]))
        if k in keys:
            return False
        keys.add(k)
    return True


def _generate_concentrated_config(seed):
    """Generate a configuration with points concentrated on a few dense lines."""
    rng = random.Random(seed)
    N, M = 54, 27
    
    # Choose 3-5 base lines with small integer coefficients
    num_base = rng.randint(3, 5)
    base_lines = []
    used_norms = set()
    
    for _ in range(50):
        A = rng.randint(-3, 3)
        B = rng.randint(-3, 3)
        C = rng.randint(-5, 5)
        if A == 0 and B == 0:
            continue
        norm = _normalize_line(A, B, C)
        if norm in used_norms:
            continue
        used_norms.add(norm)
        base_lines.append(norm)
        if len(base_lines) == num_base:
            break
    
    if len(base_lines) < 3:
        return None, None, 0
    
    # Place points on base lines, preferring intersection points
    points = []
    used_positions = set()
    
    # Find intersection points of base lines
    intersections = []
    for i in range(len(base_lines)):
        for j in range(i + 1, len(base_lines)):
            A1, B1, C1 = base_lines[i]
            A2, B2, C2 = base_lines[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            x_num = -C1 * B2 + C2 * B1
            y_num = A1 * C2 - A2 * C1
            if x_num % det == 0 and y_num % det == 0:
                x, y = x_num // det, y_num // det
                if abs(x) <= 20 and abs(y) <= 20:
                    intersections.append((x, y))
    
    # Add intersection points first (they're on 2+ lines)
    for (x, y) in intersections:
        if (x, y) not in used_positions and len(points) < N:
            points.append((x, y))
            used_positions.add((x, y))
    
    # Distribute remaining points among base lines
    remaining = N - len(points)
    if remaining > 0:
        points_per_line = [remaining // num_base] * num_base
        for i in range(remaining % num_base):
            points_per_line[i] += 1
        
        for li, (A, B, C) in enumerate(base_lines):
            count = points_per_line[li]
            if count <= 0:
                continue
            # Find integer points on this line
            candidates = []
            for x in range(-20, 21):
                for y in range(-20, 21):
                    if A * x + B * y + C == 0 and (x, y) not in used_positions:
                        candidates.append((x, y))
            rng.shuffle(candidates)
            for p in candidates[:count]:
                if len(points) < N:
                    points.append(p)
                    used_positions.add(p)
    
    # Fill any remaining with random points
    while len(points) < N:
        x = rng.randint(-10, 10)
        y = rng.randint(-10, 10)
        if (x, y) not in used_positions:
            points.append((x, y))
            used_positions.add((x, y))
    
    points = np.array(points[:N], dtype=np.int64)
    
    # Select best M lines
    all_candidates = list(base_lines)
    
    # Add lines through pairs of points
    seen = set(all_candidates)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in seen:
                seen.add(norm)
                all_candidates.append(norm)
    
    # Score and select top M
    cand_arr = np.array(all_candidates, dtype=np.int64)
    x_pts = points[:, 0]
    y_pts = points[:, 1]
    scores = np.zeros(len(cand_arr))
    for i in range(len(cand_arr)):
        val = cand_arr[i, 0] * x_pts + cand_arr[i, 1] * y_pts + cand_arr[i, 2]
        scores[i] = np.count_nonzero(val == 0)
    
    order = np.argsort(-scores)
    selected = order[:M]
    lines = cand_arr[selected]
    
    score = _count_incidents(points, lines)
    return points, lines, score


def _generate_grid_config(cols, rows, seed):
    """Generate a grid configuration with horizontal, vertical, and diagonal lines."""
    points = []
    for x in range(1, cols + 1):
        for y in range(1, rows + 1):
            points.append((x, y))
    points = np.array(points[:54], dtype=np.int64)
    
    if points.shape[0] != 54:
        return None, None, 0
    
    candidates = []
    # Horizontal lines
    for y in range(1, rows + 1):
        candidates.append((0, 1, -y))
    # Vertical lines
    for x in range(1, cols + 1):
        candidates.append((1, 0, -x))
    # Diagonals with various slopes
    for slope in [1, -1, 2, -2, 3, -3]:
        for intercept in range(-15, 20):
            count = 0
            for (x, y) in points:
                if slope * x - y + intercept == 0:
                    count += 1
            if count >= 2:
                candidates.append((slope, -1, intercept))
    
    # Deduplicate
    seen = set()
    deduped = []
    for l in candidates:
        k = _normalize_line(l[0], l[1], l[2])
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    
    # Score and select top 27
    cand_arr = np.array(deduped, dtype=np.int64)
    x_pts = points[:, 0]
    y_pts = points[:, 1]
    scores = np.zeros(len(cand_arr))
    for i in range(len(cand_arr)):
        val = cand_arr[i, 0] * x_pts + cand_arr[i, 1] * y_pts + cand_arr[i, 2]
        scores[i] = np.count_nonzero(val == 0)
    
    order = np.argsort(-scores)
    if len(order) < 27:
        return None, None, 0
    
    lines = cand_arr[order[:27]]
    score = _count_incidents(points, lines)
    return points, lines, score


def _optimize_lines_given_points(points, num_lines, time_limit):
    """Select best lines for given points using greedy + local search."""
    start = time.time()
    n = len(points)
    
    # Generate all candidate lines through pairs
    candidates = {}
    for i in range(n):
        x1, y1 = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in candidates:
                candidates[norm] = norm
    
    cand_list = list(candidates.values())
    if len(cand_list) < num_lines:
        for x in range(-10, 11):
            cand_list.append(_normalize_line(1, 0, -x))
        for y in range(-10, 11):
            cand_list.append(_normalize_line(0, 1, -y))
        seen = set()
        deduped = []
        for c in cand_list:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        cand_list = deduped
    
    cand_arr = np.array(cand_list, dtype=np.int64)
    num_cand = len(cand_arr)
    
    # Compute incidence matrix
    x_pts = points[:, 0]
    y_pts = points[:, 1]
    incidence = np.zeros((num_cand, n), dtype=np.int8)
    chunk = 500
    for s in range(0, num_cand, chunk):
        e = min(s + chunk, num_cand)
        A = cand_arr[s:e, 0]
        B = cand_arr[s:e, 1]
        C = cand_arr[s:e, 2]
        val = np.outer(A, x_pts) + np.outer(B, y_pts) + C[:, None]
        incidence[s:e] = (val == 0).astype(np.int8)
    
    scores = incidence.sum(axis=1)
    
    # Greedy initial selection
    selected_mask = np.zeros(num_cand, dtype=bool)
    order = np.argsort(-scores)
    selected = []
    for idx in order:
        if len(selected) == num_lines:
            break
        if not selected_mask[idx]:
            selected.append(idx)
            selected_mask[idx] = True
    
    # Local search
    best_total = incidence[selected].sum()
    improved = True
    max_iters = 100
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        if time.time() - start > time_limit:
            break
        for i in range(len(selected)):
            if time.time() - start > time_limit:
                break
            cur = selected[i]
            cur_score = scores[cur]
            best_alt = -1
            best_alt_score = -1
            for j in range(num_cand):
                if selected_mask[j] or scores[j] <= cur_score:
                    continue
                if scores[j] > best_alt_score:
                    best_alt_score = scores[j]
                    best_alt = j
            if best_alt >= 0:
                selected_mask[cur] = False
                selected_mask[best_alt] = True
                selected[i] = best_alt
                improved = True
    
    return cand_arr[selected]


def _optimize_points_given_lines(lines, points, time_limit):
    """Move points to intersection points of lines to maximize incidences."""
    start = time.time()
    n, m = len(points), len(lines)
    best_points = points.copy()
    best_score = _count_incidents(best_points, lines)
    
    # Find all integer intersection points of line pairs
    intersections = set()
    for i in range(m):
        A1, B1, C1 = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        for j in range(i + 1, m):
            A2, B2, C2 = int(lines[j, 0]), int(lines[j, 1]), int(lines[j, 2])
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            x_num = -C1 * B2 + C2 * B1
            y_num = A1 * C2 - A2 * C1
            if x_num % det == 0 and y_num % det == 0:
                x, y = x_num // det, y_num // det
                if abs(x) <= 200 and abs(y) <= 200:
                    intersections.add((x, y))
    
    for p in points:
        intersections.add((int(p[0]), int(p[1])))
    
    # Score each position
    pos_scores = {}
    for (x, y) in intersections:
        cnt = 0
        for i in range(m):
            if int(lines[i, 0]) * x + int(lines[i, 1]) * y + int(lines[i, 2]) == 0:
                cnt += 1
        pos_scores[(x, y)] = cnt
    
    # Greedy point movement
    occupied = set(map(tuple, best_points.tolist()))
    improved = True
    max_iters = 30
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        if time.time() - start > time_limit:
            break
        for k in range(n):
            if time.time() - start > time_limit:
                break
            old = (int(best_points[k, 0]), int(best_points[k, 1]))
            old_score = pos_scores.get(old, 0)
            best_pos = None
            best_pos_score = old_score
            for (x, y), sc in pos_scores.items():
                if (x, y) == old or (x, y) in occupied:
                    continue
                if sc > best_pos_score:
                    best_pos_score = sc
                    best_pos = (x, y)
            if best_pos is not None:
                occupied.discard(old)
                occupied.add(best_pos)
                best_points[k, 0] = best_pos[0]
                best_points[k, 1] = best_pos[1]
                improved = True
    
    new_score = _count_incidents(best_points, lines)
    return best_points, new_score


def _simulated_annealing(points, lines, deadline):
    """Simulated annealing to refine the configuration."""
    N, M = len(points), len(lines)
    curr_pts = points.copy()
    curr_lines = lines.copy()
    curr_score = _count_incidents(curr_pts, curr_lines)
    
    best_pts = curr_pts.copy()
    best_lines = curr_lines.copy()
    best_score = curr_score
    
    T = 2.0
    T_min = 0.01
    cooling = 0.9999
    
    point_set = set(map(tuple, curr_pts))
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        if random.random() < 0.5:
            # Move a point
            i = random.randint(0, N - 1)
            old = tuple(curr_pts[i])
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_pos = (old[0] + dx, old[1] + dy)
            if new_pos in point_set:
                continue
            
            # Compute delta
            old_line_hits = np.sum(curr_lines[:, 0] * old[0] + curr_lines[:, 1] * old[1] + curr_lines[:, 2] == 0)
            new_line_hits = np.sum(curr_lines[:, 0] * new_pos[0] + curr_lines[:, 1] * new_pos[1] + curr_lines[:, 2] == 0)
            delta = int(new_line_hits - old_line_hits)
            
            if delta >= 0 or random.random() < np.exp(delta / T):
                point_set.discard(old)
                point_set.add(new_pos)
                curr_pts[i, 0] = new_pos[0]
                curr_pts[i, 1] = new_pos[1]
                curr_score += delta
                if curr_score > best_score:
                    best_score = curr_score
                    best_pts = curr_pts.copy()
                    best_lines = curr_lines.copy()
        else:
            # Replace a line
            i = random.randint(0, M - 1)
            old_norm = _normalize_line(int(curr_lines[i, 0]), int(curr_lines[i, 1]), int(curr_lines[i, 2]))
            
            # Try a new line through two random points
            a = random.randint(0, N - 1)
            b = random.randint(0, N - 1)
            if a == b:
                continue
            x1, y1 = int(curr_pts[a, 0]), int(curr_pts[a, 1])
            x2, y2 = int(curr_pts[b, 0]), int(curr_pts[b, 1])
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            if A == 0 and B == 0:
                continue
            new_norm = _normalize_line(A, B, C)
            if new_norm == old_norm:
                continue
            
            # Check if new_norm already exists
            exists = False
            for j in range(M):
                if j != i and _normalize_line(int(curr_lines[j, 0]), int(curr_lines[j, 1]), int(curr_lines[j, 2])) == new_norm:
                    exists = True
                    break
            if exists:
                continue
            
            # Compute delta
            old_hits = np.sum(A * curr_pts[:, 0] + B * curr_pts[:, 1] + C == 0)
            old_A, old_B, old_C = int(curr_lines[i, 0]), int(curr_lines[i, 1]), int(curr_lines[i, 2])
            old_old_hits = np.sum(old_A * curr_pts[:, 0] + old_B * curr_pts[:, 1] + old_C == 0)
            delta = int(old_hits - old_old_hits)
            
            if delta >= 0 or random.random() < np.exp(delta / T):
                curr_lines[i, 0] = new_norm[0]
                curr_lines[i, 1] = new_norm[1]
                curr_lines[i, 2] = new_norm[2]
                curr_score += delta
                if curr_score > best_score:
                    best_score = curr_score
                    best_pts = curr_pts.copy()
                    best_lines = curr_lines.copy()
        
        T *= cooling
    
    return best_pts, best_lines, best_score


def szemeredi_trotter_incidences():
    wall_start = time.time()
    
    def time_remaining():
        return 630.0 - (time.time() - wall_start) - _DEADLINE_MARGIN
    
    # Generate multiple initial configurations
    configs = []
    
    # Grid configs
    for cols, rows in [(6, 9), (9, 6), (3, 18), (18, 3)]:
        p, l, s = _generate_grid_config(cols, rows, 42)
        if p is not None and l is not None:
            configs.append((p, l, s))
    
    # Concentrated configs
    for seed in range(10):
        p, l, s = _generate_concentrated_config(seed)
        if p is not None and l is not None:
            configs.append((p, l, s))
    
    # Select best initial config
    best_pts = None
    best_lines = None
    best_score = -1
    for p, l, s in configs:
        if not _is_valid(p, l):
            continue
        actual_s = _count_incidents(p, l)
        if actual_s > best_score:
            best_score = actual_s
            best_pts = p.copy()
            best_lines = l.copy()
    
    if best_pts is None:
        # Fallback
        pts = np.array([(x, y) for x in range(1, 7) for y in range(1, 10)], dtype=np.int64)
        lines_list = []
        for y in range(1, 10):
            lines_list.append((0, 1, -y))
        for x in range(1, 7):
            lines_list.append((1, 0, -x))
        for slope in [1, -1]:
            for intercept in range(-5, 15):
                cnt = sum(1 for (x, y) in pts if slope * x - y + intercept == 0)
                if cnt >= 2:
                    lines_list.append((slope, -1, intercept))
        seen = set()
        deduped = []
        for l in lines_list:
            k = _normalize_line(l[0], l[1], l[2])
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        lines = np.array(deduped[:27], dtype=np.int64)
        best_pts = pts
        best_lines = lines
        best_score = _count_incidents(best_pts, best_lines)
    
    # Alternating optimization
    remaining = time_remaining()
    if remaining < 10:
        return best_pts.astype(np.int64), best_lines.astype(np.int64)
    
    num_rounds = min(8, int(remaining // 40))
    per_round = remaining / max(1, num_rounds)
    
    for rnd in range(num_rounds):
        if time_remaining() < 15:
            break
        
        # Optimize lines
        line_time = min(per_round * 0.4, 40.0)
        new_lines = _optimize_lines_given_points(best_pts, 27, line_time)
        if len(new_lines) == 27 and _is_valid(best_pts, new_lines):
            new_score = _count_incidents(best_pts, new_lines)
            if new_score > best_score:
                best_lines = new_lines
                best_score = new_score
        
        # Optimize points
        point_time = min(per_round * 0.3, 30.0)
        new_pts, new_score = _optimize_points_given_lines(best_lines, best_pts, point_time)
        if _is_valid(new_pts, best_lines):
            actual = _count_incidents(new_pts, best_lines)
            if actual > best_score:
                best_pts = new_pts
                best_score = actual
        
        # Simulated annealing if time permits
        if time_remaining() > 30:
            sa_time = min(per_round * 0.3, time_remaining() - 20)
            if sa_time > 5:
                sa_pts, sa_lines, sa_score = _simulated_annealing(best_pts, best_lines, time.time() + sa_time)
                if _is_valid(sa_pts, sa_lines) and sa_score > best_score:
                    best_pts = sa_pts
                    best_lines = sa_lines
                    best_score = sa_score
    
    if not _is_valid(best_pts, best_lines):
        pts = np.array([(x, y) for x in range(1, 7) for y in range(1, 10)], dtype=np.int64)
        lines_list = [(0, 1, -y) for y in range(1, 10)] + [(1, 0, -x) for x in range(1, 7)]
        seen = set()
        deduped = []
        for l in lines_list:
            k = _normalize_line(l[0], l[1], l[2])
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        best_pts = pts
        best_lines = np.array(deduped[:27], dtype=np.int64)
    
    return best_pts.astype(np.int64), best_lines.astype(np.int64)


# EVOLVE-BLOCK-END
