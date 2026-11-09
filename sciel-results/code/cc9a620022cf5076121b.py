import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit

    def elapsed():
        return time.time() - t0

    def compute_max_area_count(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        Uses vectorized cross-product computation over all point pairs.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]  # shape (M, 2), M = n*(n-1)/2
        dx = d[:, 0].astype(np.int64)
        dy = d[:, 1].astype(np.int64)
        M = len(dx)

        # For each base vector a (index u), compute cross products with all later vectors v
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)

        if not area_counts:
            return 0
        return max(area_counts.values())

    def score_of(pts):
        return compute_max_area_count(pts)

    def targeted_search(pts, target_area, max_iters=2000):
        """
        Constraint-satisfaction approach: try to maximize the count of triples
        with doubled area == target_area by moving points.
        """
        n = len(pts)
        current = list(pts)
        current_score = score_of(current)
        best = list(current)
        best_score = current_score

        # Count how many triples currently have the target area
        def count_target_area(pts):
            P = np.array(pts, dtype=np.int64)
            i_idx, j_idx = np.triu_indices(n, k=1)
            d = P[j_idx] - P[i_idx]
            dx = d[:, 0]
            dy = d[:, 1]
            M = len(dx)
            count = 0
            for u in range(M - 1):
                cu = dx[u]
                cv = dy[u]
                if cu == 0 and cv == 0:
                    continue
                cross = cu * dy[u+1:] - cv * dx[u+1:]
                count += int(np.sum(np.abs(cross) == target_area))
            return count

        target_count = count_target_area(current)

        for it in range(max_iters):
            if elapsed() > DEADLINE - 2:
                break

            idx = random.randint(0, 29)
            old_x, old_y = current[idx]

            # Generate candidate new position
            r = random.random()
            if r < 0.05:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            else:
                step = random.randint(1, 8)
                new_x = old_x + random.randint(-step, step)
                new_y = old_y + random.randint(-step, step)

            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue

            new_pt = (new_x, new_y)
            if new_pt == (old_x, old_y):
                continue

            # Check distinctness
            is_valid = True
            for j in range(30):
                if j == idx:
                    continue
                if current[j] == new_pt:
                    is_valid = False
                    break

            if not is_valid:
                continue

            # Try the move
            current[idx] = new_pt
            new_target_count = count_target_area(current)

            # Accept if it improves target count, or with some probability if neutral
            if new_target_count > target_count or (new_target_count == target_count and random.random() < 0.3):
                target_count = new_target_count
                # Also check overall score
                new_score = score_of(current)
                if new_score > best_score:
                    best_score = new_score
                    best = list(current)
            else:
                current[idx] = (old_x, old_y)

        return best, best_score

    # 1. Initialize with strong structured candidates
    candidates = []

    # Two parallel lines
    candidates.append([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])

    # Grid 6x5
    candidates.append([(x, y) for x in range(6) for y in range(5)])

    # Three lines
    candidates.append([(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)])

    # Square grid 5x6 centered
    candidates.append([(x, y) for x in range(-2, 3) for y in range(-3, 3)])

    # Diagonal lines: y=x and y=x+1, 15 points each
    candidates.append([(x, x) for x in range(15)] + [(x, x+1) for x in range(15)])

    # Rectangle 10x3
    candidates.append([(x, y) for x in range(10) for y in range(3)])

    # Two perpendicular lines with offset
    candidates.append([(x, 0) for x in range(15)] + [(0, y) for y in range(1, 16)])

    # Random
    random.seed(42)
    cand4 = []
    used = set()
    while len(cand4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            cand4.append((x, y))
    candidates.append(cand4)

    # Evaluate all candidates
    best_points = None
    best_score = 0
    best_area = 1

    for cand in candidates:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            # Find the dominant area for targeted search
            P = np.array(cand, dtype=np.int64)
            i_idx, j_idx = np.triu_indices(30, k=1)
            d = P[j_idx] - P[i_idx]
            dx = d[:, 0]
            dy = d[:, 1]
            M = len(dx)
            area_counts = defaultdict(int)
            for u in range(M - 1):
                cu = dx[u]
                cv = dy[u]
                if cu == 0 and cv == 0:
                    continue
                cross = cu * dy[u+1:] - cv * dx[u+1:]
                nz = cross[cross != 0]
                if len(nz) == 0:
                    continue
                nz_abs = np.abs(nz)
                vals, cnts = np.unique(nz_abs, return_counts=True)
                for v, c in zip(vals, cnts):
                    area_counts[int(v)] += int(c)
            if area_counts:
                best_area = max(area_counts, key=area_counts.get)

    # 2. Targeted search on the best candidate
    if best_points and elapsed() < DEADLINE - 5:
        best_points, best_score = targeted_search(best_points, best_area, max_iters=3000)

    # 3. General local search to refine
    current_points = list(best_points)
    current_score = best_score
    step_size = 3
    last_improvement_iter = 0
    iter_count = 0

    while elapsed() < DEADLINE - 2:
        iter_count += 1

        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]

        r = random.random()
        if r < 0.03:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.15:
            new_x = old_x + random.randint(-6, 6)
            new_y = old_y + random.randint(-6, 6)
        else:
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)

        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue

        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y):
            continue

        is_valid = True
        for j in range(30):
            if j == idx:
                continue
            if current_points[j] == new_pt:
                is_valid = False
                break

        if not is_valid:
            continue

        current_points[idx] = new_pt
        new_score = score_of(current_points)

        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                last_improvement_iter = iter_count
                if step_size > 1 and random.random() < 0.4:
                    step_size = max(1, step_size - 1)
        else:
            current_points[idx] = (old_x, old_y)

        if iter_count - last_improvement_iter > 2000:
            current_points = list(best_points)
            current_score = best_score
            last_improvement_iter = iter_count
            step_size = random.randint(4, 10)

        if random.random() < 0.005:
            step_size = random.randint(1, 6)

    # 4. Final targeted search with updated best area
    if best_points and elapsed() < DEADLINE - 3:
        # Re-identify dominant area
        P = np.array(best_points, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(30, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)
        if area_counts:
            target = max(area_counts, key=area_counts.get)
            best_points, best_score = targeted_search(best_points, target, max_iters=2000)

    return best_points
