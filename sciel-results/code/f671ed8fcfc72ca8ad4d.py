import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - t0

    def compute_all_areas(pts):
        """
        Compute all non-zero doubled areas for every triple.
        Returns (area_counts dict, max_count, dominant_area)
        """
        n = len(pts)
        if n < 3:
            return {}, 0, 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]  # shape (M, 2), M = n*(n-1)/2
        dx = d[:, 0].astype(np.int64)
        dy = d[:, 1].astype(np.int64)
        M = len(dx)

        # For each pair of edges sharing a vertex, compute cross product
        # Edge (i,j) and edge (i,k) share vertex i -> area = cross(P[j]-P[i], P[k]-P[i])
        # Edge (i,j) and edge (j,k) share vertex j -> area = cross(P[i]-P[j], P[k]-P[j])
        # Edge (i,k) and edge (j,k) share vertex k -> area = cross(P[i]-P[k], P[j]-P[k])
        
        # More efficient: for each vertex i, take all edges from i, compute cross products
        area_counts = defaultdict(int)
        
        for i in range(n):
            # Find all edges starting from vertex i
            edges_from_i = np.where(i_idx == i)[0]
            if len(edges_from_i) < 2:
                continue
            
            dx_i = dx[edges_from_i]
            dy_i = dy[edges_from_i]
            m_i = len(dx_i)
            
            # Cross product of all pairs of edges from i
            # cross(a,b) = a_x * b_y - a_y * b_x
            for a in range(m_i - 1):
                cu = dx_i[a]
                cv = dy_i[a]
                if cu == 0 and cv == 0:
                    continue
                cross = cu * dy_i[a+1:] - cv * dx_i[a+1:]
                nz = cross[cross != 0]
                if len(nz) == 0:
                    continue
                nz_abs = np.abs(nz)
                vals, cnts = np.unique(nz_abs, return_counts=True)
                for v, c in zip(vals, cnts):
                    area_counts[int(v)] += int(c)
        
        if not area_counts:
            return area_counts, 0, 0
        max_count = max(area_counts.values())
        dominant_area = max(area_counts, key=area_counts.get)
        return area_counts, max_count, dominant_area

    def score_of(pts):
        """Compute the maximum multiplicity of any non-zero doubled area."""
        _, max_count, _ = compute_all_areas(pts)
        return max_count

    def incremental_score_update(pts, moved_idx, old_pt, new_pt):
        """
        Update score after moving one point. Only recompute triples involving moved_idx.
        Returns new max area count.
        """
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        
        # Compute all areas first to get baseline
        area_counts, _, _ = compute_all_areas(pts)
        
        # This is still O(n^3) but we can optimize by only checking affected triples
        # For now, just return the full score (this function is a placeholder for optimization)
        return max(area_counts.values()) if area_counts else 0

    def targeted_search(pts, target_area, max_iters=5000):
        """
        Constraint-satisfaction approach: try to maximize the count of triples
        with doubled area == target_area by moving points.
        """
        n = len(pts)
        current = list(pts)
        
        # Count how many triples currently have the target area
        def count_target_area(pts):
            P = np.array(pts, dtype=np.int64)
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        x1, y1 = P[i]
                        x2, y2 = P[j]
                        x3, y3 = P[k]
                        area = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                        if area == target_area:
                            count += 1
            return count

        current_target_count = count_target_area(current)
        current_score = score_of(current)
        best = list(current)
        best_score = current_score

        for it in range(max_iters):
            if elapsed() > DEADLINE - 3:
                break

            idx = random.randint(0, 29)
            old_x, old_y = current[idx]

            # Generate candidate new position
            r = random.random()
            if r < 0.08:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            else:
                step = random.randint(1, 6)
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
            if new_target_count > current_target_count or (new_target_count == current_target_count and random.random() < 0.2):
                current_target_count = new_target_count
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

    # 4x8 grid
    candidates.append([(x, y) for x in range(4) for y in range(8)])

    # 5x6 grid
    candidates.append([(x, y) for x in range(5) for y in range(6)])

    # Lattice: points (2i, j) for i in 0..7, j in 0..3
    candidates.append([(2*i, j) for i in range(8) for j in range(4)])

    # Lattice: points (i, 2j) for i in 0..14, j in 0..2
    candidates.append([(i, 2*j) for i in range(15) for j in range(2)])

    # Hexagonal-like: alternating rows offset
    hex_pts = []
    for row in range(6):
        offset = row % 2
        for col in range(5):
            x = col + offset
            y = row
            hex_pts.append((x, y))
    candidates.append(hex_pts[:30])

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
            area_counts, _, dominant = compute_all_areas(cand)
            if area_counts:
                best_area = dominant

    # 2. Targeted search on the best candidate
    if best_points and elapsed() < DEADLINE - 10:
        best_points, best_score = targeted_search(best_points, best_area, max_iters=8000)

    # 3. General local search to refine
    current_points = list(best_points)
    current_score = best_score
    step_size = 3
    last_improvement_iter = 0
    iter_count = 0

    while elapsed() < DEADLINE - 5:
        iter_count += 1

        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]

        r = random.random()
        if r < 0.03:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.15:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
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

        if iter_count - last_improvement_iter > 3000:
            current_points = list(best_points)
            current_score = best_score
            last_improvement_iter = iter_count
            step_size = random.randint(4, 10)

        if random.random() < 0.005:
            step_size = random.randint(1, 6)

    # 4. Final targeted search with updated best area
    if best_points and elapsed() < DEADLINE - 3:
        # Re-identify dominant area
        area_counts, _, target = compute_all_areas(best_points)
        if area_counts:
            best_points, best_score = targeted_search(best_points, target, max_iters=5000)

    return best_points
