import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)

    def build_line_data(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1

        # For each line, store which points are on it
        line_points = {}
        for key in line_counts:
            line_points[key] = set()

        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_points[key].add(i)
                line_points[key].add(j)

        # Build conflict graph: edge (i,j) exists if there's a third point k collinear with i and j
        # i.e., i and j share a line with at least 3 points
        conflict = [set() for _ in range(n)]
        for key, pts_on_line in line_points.items():
            if len(pts_on_line) >= 3:
                pts_list = list(pts_on_line)
                for a in range(len(pts_list)):
                    for b in range(a + 1, len(pts_list)):
                        conflict[pts_list[a]].add(pts_list[b])
                        conflict[pts_list[b]].add(pts_list[a])

        # Check for 4-collinear
        has_four = any(len(pts) >= 4 for pts in line_points.values())

        return line_counts, line_points, conflict, has_four

    def exact_max_no3_collinear(points):
        """Find the largest subset of points with no three collinear using branch and bound."""
        n = len(points)
        if n <= 2:
            return n

        line_counts, line_points, conflict, has_four = build_line_data(points)

        if has_four:
            return -1000000  # Invalid

        # Greedy coloring upper bound for independent set
        def greedy_color_bound(vertex_set, conflict):
            """Upper bound on independent set size via greedy coloring."""
            # Greedy coloring: assign colors so that adjacent vertices have different colors
            # The max color used is an upper bound on the independent set size
            remaining = list(vertex_set)
            colors = {}
            color_count = 0
            while remaining:
                color_count += 1
                # Greedy: pick a vertex with max degree in remaining
                remaining.sort(key=lambda v: -len(conflict[v] & set(remaining)))
                if not remaining:
                    break
                # Try to assign this color to as many non-adjacent vertices as possible
                color_class = []
                available = list(remaining)
                for v in available:
                    if all(v not in conflict[u] for u in color_class):
                        color_class.append(v)
                for v in color_class:
                    colors[v] = color_count
                    remaining.remove(v)
            return color_count

        # Branch and bound
        best = [0]
        selected = set()
        remaining = set(range(n))

        def branch_and_bound(selected_set, remaining_set, current_size):
            if current_size + len(remaining_set) <= best[0]:
                return

            if not remaining_set:
                if current_size > best[0]:
                    best[0] = current_size
                return

            # Upper bound via greedy coloring
            ub = current_size + greedy_color_bound(remaining_set, conflict)
            if ub <= best[0]:
                return

            # Pick a vertex with high degree for branching
            rem_list = list(remaining_set)
            rem_list.sort(key=lambda v: -len(conflict[v] & remaining_set))
            v = rem_list[0]

            # Branch 1: include v
            new_selected = selected_set | {v}
            new_remaining = remaining_set - {v} - conflict[v]
            branch_and_bound(new_selected, new_remaining, current_size + 1)

            # Branch 2: exclude v
            branch_and_bound(selected_set, remaining_set - {v}, current_size)

        branch_and_bound(set(), set(range(n)), 0)
        return best[0]

    def fast_score(points):
        """Quick score: count triples and check 4-collinear."""
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1

        has_four = False
        triples = 0
        for key, count in line_counts.items():
            if count >= 4:
                has_four = True
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6

        if has_four:
            return -1000000
        return triples

    def line_packing_construction(seed_offset=0):
        """
        Construct 20 points by packing them onto shared lines to create dense triple structures.
        The idea: create several lines each with 3 points, sharing points between lines.
        """
        rng = random.Random(seed_offset)
        points = []
        used = set()

        # Strategy: create a set of lines, each with 3 points, with overlap
        # Start with some base lines
        base_lines = []
        for _ in range(8):
            # Random line with integer points
            x1 = rng.randint(-50, 50)
            y1 = rng.randint(-50, 50)
            dx = rng.randint(-20, 20)
            dy = rng.randint(-20, 20)
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            base_lines.append((x1, y1, dx, dy))

        # Place points on these lines
        for line in base_lines:
            x0, y0, dx, dy = line
            for t in [0, 1, 2]:
                px = x0 + t * dx
                py = y0 + t * dy
                if -100 <= px <= 100 and -100 <= py <= 100:
                    if (px, py) not in used:
                        points.append([px, py])
                        used.add((px, py))
                        if len(points) >= 20:
                            break
            if len(points) >= 20:
                break

        # Fill remaining with random points that create triples
        while len(points) < 20:
            # Try to place a point that creates a new triple
            best_pos = None
            best_gain = -1
            for _ in range(500):
                px = rng.randint(-100, 100)
                py = rng.randint(-100, 100)
                if (px, py) in used:
                    continue
                gain = 0
                creates_four = False
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        p1 = points[i]
                        p2 = points[j]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        if (px - p1[0]) * dy == (py - p1[1]) * dx:
                            # Check if there's already a third point on this line
                            key = line_key(p1, p2)
                            # Count points on this line
                            cnt = 2
                            for k in range(len(points)):
                                if k != i and k != j:
                                    pk = points[k]
                                    if (pk[0] - p1[0]) * dy == (pk[1] - p1[1]) * dx:
                                        cnt += 1
                            if cnt >= 3:
                                creates_four = True
                                break
                            else:
                                gain += 1
                    if creates_four:
                        break
                if creates_four:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_pos = (px, py)
            if best_pos is not None:
                points.append(list(best_pos))
                used.add(best_pos)
            else:
                # Fallback: random
                while True:
                    px = rng.randint(-100, 100)
                    py = rng.randint(-100, 100)
                    if (px, py) not in used:
                        break
                points.append([px, py])
                used.add((px, py))

        # Verify
        _, _, _, has_four = build_line_data(points)
        if has_four:
            return None
        return points

    def local_search(points, time_limit):
        """Local search minimizing the exact max no-3-collinear subset size."""
        current = [p[:] for p in points]
        current_score = exact_max_no3_collinear(current)
        best = [p[:] for p in current]
        best_score = current_score

        no_improve = 0
        while time.time() < time_limit:
            improved = False
            # Try moving each point
            for i in range(20):
                if time.time() > time_limit:
                    break
                # Generate candidate positions
                candidates = set()
                # Directed moves: extend lines
                for j in range(20):
                    if i == j:
                        continue
                    p1 = current[i]
                    p2 = current[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    dx_n = dx // g
                    dy_n = dy // g
                    for t in range(1, 6):
                        for sign in [1, -1]:
                            np = [p2[0] + sign * t * dx_n, p2[1] + sign * t * dy_n]
                            if -100 <= np[0] <= 100 and -100 <= np[1] <= 100:
                                candidates.add((np[0], np[1]))
                            np = [p1[0] - sign * t * dx_n, p1[1] - sign * t * dy_n]
                            if -100 <= np[0] <= 100 and -100 <= np[1] <= 100:
                                candidates.add((np[0], np[1]))

                # Random candidates
                for _ in range(30):
                    px = random.randint(-100, 100)
                    py = random.randint(-100, 100)
                    candidates.add((px, py))

                # Remove current position and existing points
                candidates.discard((current[i][0], current[i][1]))
                for p in current:
                    candidates.discard((p[0], p[1]))

                # Try each candidate
                best_move_score = current_score
                best_move_pos = None
                for pos in list(candidates)[:50]:  # Limit for speed
                    test = current[:]
                    test[i] = [pos[0], pos[1]]
                    s = exact_max_no3_collinear(test)
                    if s == -1000000:
                        continue
                    if s < best_move_score:
                        best_move_score = s
                        best_move_pos = pos

                if best_move_pos is not None:
                    current[i] = [best_move_pos[0], best_move_pos[1]]
                    current_score = best_move_score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [p[:] for p in current]
                    break

            if not improved:
                no_improve += 1
                if no_improve >= 3:
                    # Random jump
                    idx = random.randint(0, 19)
                    current[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
                    s = exact_max_no3_collinear(current)
                    if s == -1000000:
                        current[idx] = best[idx][:]
                        current_score = best_score
                        break
                    current_score = s
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [p[:] for p in current]
                else:
                    break

        return best

    # Main search
    best_overall = None
    best_overall_score = 999999

    # Phase 1: Generate candidates
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        pts = line_packing_construction(seed)
        if pts is None:
            continue
        score = exact_max_no3_collinear(pts)
        if score < best_overall_score:
            best_overall_score = score
            best_overall = pts

        # Refine
        if time.time() < deadline - 10:
            search_time = min(4, deadline - 10 - time.time())
            if search_time > 1.0:
                refined = local_search(pts, time.time() + search_time)
                refined_score = exact_max_no3_collinear(refined)
                if refined_score < best_overall_score:
                    best_overall_score = refined_score
                    best_overall = refined

    # Phase 2: Final refinement
    if best_overall is not None and time.time() < deadline - 5:
        final = local_search(best_overall, time.time() + min(10, deadline - 5 - time.time()))
        final_score = exact_max_no3_collinear(final)
        if final_score < best_overall_score:
            best_overall_score = final_score
            best_overall = final

    if best_overall is None:
        # Fallback: random
        random.seed(42)
        pts = []
        used = set()
        while len(pts) < 20:
            px = random.randint(-100, 100)
            py = random.randint(-100, 100)
            if (px, py) not in used:
                pts.append([px, py])
                used.add((px, py))
        best_overall = pts

    return best_overall
