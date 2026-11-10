# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys
from collections import defaultdict, Counter

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for n=54 points and m=27 lines on integer lattice.
    Uses a structural approach: generate candidate lines from point pairs, then select
    27 lines that maximize total incidences using greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 585  # safety margin before 630s kill

    def normalize_line(A, B, C):
        g = math.gcd(abs(int(A)), math.gcd(abs(int(B)), abs(int(C))))
        if g == 0:
            return (0, 0, 0)
        A, B, C = int(A) // g, int(B) // g, int(C) // g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)

    def count_incidents(pts, lns):
        """Vectorized count of point-line incidences."""
        vals = lns[:, 0:1].T * pts[:, 0:1] + lns[:, 1:2].T * pts[:, 1:2] + lns[:, 2:3].T
        return int(np.sum(vals == 0))

    def has_duplicate_points(pts):
        seen = set()
        for p in pts:
            key = (int(p[0]), int(p[1]))
            if key in seen:
                return True
            seen.add(key)
        return False

    def has_duplicate_lines(lns):
        seen = set()
        for i in range(len(lns)):
            n = normalize_line(lns[i, 0], lns[i, 1], lns[i, 2])
            if n in seen:
                return True
            seen.add(n)
        return False

    # Generate a diverse set of candidate points
    # Use a grid plus some random perturbations
    candidate_points = set()
    # Grid from 1..10 x 1..30
    for x in range(1, 11):
        for y in range(1, 31):
            candidate_points.add((x, y))
    # Add some more spread out points
    for x in range(1, 15):
        for y in range(1, 5):
            candidate_points.add((x, y))
    for x in range(1, 5):
        for y in range(1, 40):
            candidate_points.add((x, y))

    np.random.seed(42)
    # Add random points
    for _ in range(200):
        x = np.random.randint(1, 20)
        y = np.random.randint(1, 50)
        candidate_points.add((x, y))

    candidate_points = list(candidate_points)

    # Strategy: 
    # 1. Start with a good initial configuration (grid-based)
    # 2. Generate all candidate lines from pairs of candidate points
    # 3. Use greedy selection to pick 27 lines that maximize incidences
    # 4. Then do local search: try swapping individual lines or points

    # Initial configuration: k=3 grid
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    best_lines = np.array(rows, dtype=np.int64)

    best_score = count_incidents(best_points, best_lines)

    # Generate candidate lines from pairs of points
    # For each pair of points, compute the line through them
    candidate_lines_dict = {}  # normalized line -> set of point indices on it

    pts_list = [(int(p[0]), int(p[1])) for p in best_points]

    # Add all lines through pairs of the 54 points
    for i in range(54):
        for j in range(i + 1, 54):
            x1, y1 = pts_list[i]
            x2, y2 = pts_list[j]
            # Line through (x1,y1) and (x2,y2): (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            if A == 0 and B == 0:
                continue
            n = normalize_line(A, B, C)
            if n not in candidate_lines_dict:
                candidate_lines_dict[n] = set()
            candidate_lines_dict[n].add(i)
            candidate_lines_dict[n].add(j)

    # Also add lines from candidate points that aren't in our set
    # For each candidate point not in best_points, find lines through it and existing points
    existing_pt_set = set(pts_list)
    for cp in candidate_points:
        if cp in existing_pt_set:
            continue
        for i in range(54):
            x1, y1 = pts_list[i]
            x2, y2 = cp
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            if A == 0 and B == 0:
                continue
            n = normalize_line(A, B, C)
            if n not in candidate_lines_dict:
                candidate_lines_dict[n] = set()
            candidate_lines_dict[n].add(i)
            # Also check if any other existing points lie on this line
            for j in range(54):
                if j == i:
                    continue
                xj, yj = pts_list[j]
                if A * xj + B * yj + C == 0:
                    candidate_lines_dict[n].add(j)

    # For each candidate line, compute its incidence count with current points
    line_scores = {}
    for n, pt_set in candidate_lines_dict.items():
        A, B, C = n
        count = 0
        for i in range(54):
            x, y = pts_list[i]
            if A * x + B * y + C == 0:
                count += 1
        line_scores[n] = count

    # Sort lines by score descending
    sorted_lines = sorted(line_scores.keys(), key=lambda n: line_scores[n], reverse=True)

    # Greedy selection: pick 27 lines that maximize total incidences
    # A line's contribution depends on which points are "covered" by other selected lines
    # Use a greedy approach: repeatedly pick the line with highest marginal gain

    def compute_total_score(selected_lines, points_list):
        """Compute total incidences for given lines and points."""
        total = 0
        for (A, B, C) in selected_lines:
            for (x, y) in points_list:
                if A * x + B * y + C == 0:
                    total += 1
        return total

    def greedy_line_selection(points_list, num_lines=27, time_limit=120):
        """Greedy selection of num_lines lines to maximize incidences."""
        selected = []
        selected_set = set()
        remaining = list(sorted_lines)

        # Precompute for each line which points it hits
        line_points = {}
        for n in remaining:
            A, B, C = n
            pts_on_line = set()
            for i, (x, y) in enumerate(points_list):
                if A * x + B * y + C == 0:
                    pts_on_line.add(i)
            line_points[n] = pts_on_line

        # Current incidence count per point
        pt_counts = [0] * 54

        while len(selected) < num_lines:
            if time.time() > time_limit:
                break

            best_line = None
            best_gain = 0

            for n in remaining:
                if n in selected_set:
                    continue
                # Marginal gain = number of points on this line that aren't already counted
                # Actually, each (point, line) pair counts separately, so gain is just
                # the number of points on this line
                gain = len(line_points[n])
                if gain > best_gain:
                    best_gain = gain
                    best_line = n

            if best_line is None:
                break

            selected.append(best_line)
            selected_set.add(best_line)
            remaining.remove(best_line)

        return selected

    # Try greedy line selection
    if time.time() < deadline - 30:
        greedy_selected = greedy_line_selection(pts_list, 27, time_limit=60)
        if len(greedy_selected) == 27:
            greedy_lines = np.array(greedy_selected, dtype=np.int64)
            greedy_score = count_incidents(best_points, greedy_lines)
            if greedy_score > best_score:
                best_lines = greedy_lines
                best_score = greedy_score

    # Now do local search: alternate between optimizing lines and points
    improvement_found = True
    max_iterations = 10
    iteration = 0

    while improvement_found and iteration < max_iterations:
        improvement_found = False
        iteration += 1

        if time.time() > deadline - 30:
            break

        # Rebuild line scores with current points
        pts_list = [(int(p[0]), int(p[1])) for p in best_points]
        existing_pt_set = set(pts_list)

        # Generate candidate lines from current points
        candidate_lines_dict = {}
        for i in range(54):
            for j in range(i + 1, 54):
                x1, y1 = pts_list[i]
                x2, y2 = pts_list[j]
                A = y2 - y1
                B = -(x2 - x1)
                C = (x2 - x1) * y1 - (y2 - y1) * x2
                if A == 0 and B == 0:
                    continue
                n = normalize_line(A, B, C)
                if n not in candidate_lines_dict:
                    candidate_lines_dict[n] = set()
                candidate_lines_dict[n].add(i)
                candidate_lines_dict[n].add(j)

        # Compute line scores
        line_scores = {}
        for n, pt_set in candidate_lines_dict.items():
            A, B, C = n
            count = 0
            for i in range(54):
                x, y = pts_list[i]
                if A * x + B * y + C == 0:
                    count += 1
            line_scores[n] = count

        # Try replacing each line with a better one
        current_normalized = [normalize_line(int(best_lines[i, 0]), int(best_lines[i, 1]), int(best_lines[i, 2])) for i in range(27)]
        current_set = set(current_normalized)

        for li in range(27):
            if time.time() > deadline - 30:
                break

            orig_line = current_normalized[li]
            orig_inc = line_scores.get(orig_line, 0)

            # Find best replacement
            best_new_line = None
            best_delta = 0

            for n, score in line_scores.items():
                if n in current_set or n == orig_line:
                    continue
                delta = score - orig_inc
                if delta > best_delta:
                    best_delta = delta
                    best_new_line = n

            if best_new_line is not None:
                new_lines = best_lines.copy()
                new_lines[li] = np.array(best_new_line, dtype=np.int64)
                new_score = count_incidents(best_points, new_lines)
                if new_score > best_score:
                    best_lines = new_lines
                    best_score = new_score
                    current_normalized[li] = best_new_line
                    current_set = set(current_normalized)
                    improvement_found = True

        # Try moving points
        if time.time() > deadline - 30:
            break

        for pi in range(54):
            if time.time() > deadline - 30:
                break

            orig_pt = (int(best_points[pi, 0]), int(best_points[pi, 1]))

            # Count incidences for original point
            orig_pt_inc = 0
            for li in range(27):
                A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                if A * orig_pt[0] + B * orig_pt[1] + C == 0:
                    orig_pt_inc += 1

            # Try candidate points
            best_new_pt = None
            best_delta = 0

            for cp in candidate_points:
                if cp == orig_pt:
                    continue
                if cp in existing_pt_set:
                    continue

                new_pt_inc = 0
                for li in range(27):
                    A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                    if A * cp[0] + B * cp[1] + C == 0:
                        new_pt_inc += 1

                delta = new_pt_inc - orig_pt_inc
                if delta > best_delta:
                    best_delta = delta
                    best_new_pt = cp

            if best_new_pt is not None:
                new_points = best_points.copy()
                new_points[pi] = np.array(best_new_pt, dtype=np.int64)
                new_score = count_incidents(new_points, best_lines)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
                    existing_pt_set = set((int(p[0]), int(p[1])) for p in best_points)
                    improvement_found = True

    # Final verification
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    assert not has_duplicate_points(best_points)
    assert not has_duplicate_lines(best_lines)

    return best_points, best_lines

# EVOLVE-BLOCK-END
