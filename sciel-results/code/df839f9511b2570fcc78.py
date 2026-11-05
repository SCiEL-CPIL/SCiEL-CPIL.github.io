import random
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    N = 20
    LIMIT = 100

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def is_valid(points):
        """Check if no 4 points are collinear."""
        n = len(points)
        for i in range(n - 3):
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n - 1):
                    p1, p2, p3 = points[i], points[j], points[k]
                    if not are_collinear(p1, p2, p3):
                        continue
                    for l in range(k + 1, n):
                        p4 = points[l]
                        if are_collinear(p1, p2, p4):
                            return False
        return True

    def get_triples(points):
        """Return list of collinear triples (i, j, k) with i<j<k"""
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def calc_exact_score(points):
        """
        Exact maximum independent set in the 3-uniform hypergraph of collinear triples.
        Uses branch and bound with bitmask and greedy coloring upper bound.
        """
        n = len(points)
        triples = get_triples(points)

        if not triples:
            return n

        # Build adjacency: for each point, which other points form a triple with it
        # Actually, we need: for each point i, the set of points that conflict with i
        # in the sense that they appear together in some triple.
        # But the constraint is: no triple can have all 3 selected.
        # So it's a 3-uniform hypergraph independent set problem.

        # Represent each triple as a bitmask
        triple_masks = []
        for i, j, k in triples:
            mask = (1 << i) | (1 << j) | (1 << k)
            triple_masks.append(mask)

        # For each vertex, compute which triple_masks contain it
        vertex_triples = [[] for _ in range(n)]
        for idx, mask in enumerate(triple_masks):
            for v in range(n):
                if mask & (1 << v):
                    vertex_triples[v].append(idx)

        # Branch and bound
        best = [0]
        best_set = [0]

        def greedy_color_bound(remaining_mask):
            """
            Upper bound on independent set size using greedy coloring of the conflict graph.
            Two vertices conflict if they share a triple (i.e., there exists a triple containing both).
            Actually for 3-uniform hypergraph, the "conflict graph" has edges between pairs
            that appear together in at least one triple. The independent set in the hypergraph
            is not the same as independent set in this graph, but we can use it as a bound.
            
            Better approach: use a simple upper bound. For each remaining vertex, if we include it,
            we must exclude all vertices that form a triple with two others already... 
            This is complex. Let's use a simpler bound: 
            The maximum independent set size <= remaining_count - min_triples_to_break
            where min_triples_to_break is a lower bound on how many triples must be broken.
            
            Actually, let's use: upper_bound = popcount(remaining_mask)
            and prune if best[0] >= upper_bound.
            
            For a tighter bound: each triple in the remaining set can "waste" at most 1 point
            (we can keep 2 out of 3). So the number of points we must remove is at least
            the number of disjoint triples we can find greedily.
            """
            # Greedy: find maximum number of disjoint triples in remaining
            used = 0
            count = 0
            for mask in triple_masks:
                m = mask & remaining_mask
                if m == mask and (m & used) == 0:
                    used |= m
                    count += 1
            # Each disjoint triple forces us to drop at least 1 point
            upper = bin(remaining_mask).count('1') - count
            return upper

        def branch(remaining_mask, current_count):
            if time.time() > deadline - 5:
                return

            if remaining_mask == 0:
                if current_count > best[0]:
                    best[0] = current_count
                    best_set[0] = current_count
                return

            # Upper bound
            ub = greedy_color_bound(remaining_mask)
            if current_count + ub <= best[0]:
                return

            # Pick a vertex to branch on - choose one involved in most triples
            # among remaining vertices
            best_v = -1
            best_conflict = -1
            for v in range(n):
                if remaining_mask & (1 << v):
                    # Count how many triples contain v and are fully in remaining
                    cnt = 0
                    for idx in vertex_triples[v]:
                        if triple_masks[idx] & remaining_mask == triple_masks[idx]:
                            cnt += 1
                    if cnt > best_conflict:
                        best_conflict = cnt
                        best_v = v

            if best_v == -1:
                # No more vertices
                if current_count > best[0]:
                    best[0] = current_count
                return

            # Branch 1: include best_v
            # If we include best_v, we need to ensure no triple containing best_v is fully selected.
            # Since we're building the set incrementally, if we include best_v, we must exclude
            # at least one other member of each triple containing best_v.
            # This is complex. Instead, let's use a different formulation.

            # Alternative: think of it as: we want to select a subset S such that for every triple,
            # not all 3 are in S. Equivalently, for every triple, at least one is NOT in S.
            # This is a hitting set / vertex cover in the hypergraph.
            # The answer is n - (minimum vertex cover of the hypergraph).

            # Let me reformulate: find minimum vertex cover in the 3-uniform hypergraph.
            # Then score = n - min_vertex_cover.

            # For minimum vertex cover, branch on a triple: at least one of its 3 vertices must be in the cover.
            # Branch: pick v1, or v2, or v3 to be in the cover.

            # Switch to vertex cover formulation
            pass

        # Let me redo with vertex cover formulation
        best_cover = [n]

        def min_vertex_cover(remaining_mask, cover_count):
            """
            Find minimum vertex cover. remaining_mask: vertices not yet decided.
            cover_count: number of vertices already in the cover.
            """
            if time.time() > deadline - 5:
                return

            if cover_count >= best_cover[0]:
                return

            # Find a triple that is not yet "covered" (i.e., none of its vertices are in the cover)
            # A triple is covered if at least one of its vertices is in the cover.
            # So we need a triple where ALL three vertices are in remaining_mask (not in cover).
            uncovered_triple = None
            for mask in triple_masks:
                # Check if all 3 vertices of this triple are in remaining_mask
                if mask & remaining_mask == mask:
                    uncovered_triple = mask
                    break

            if uncovered_triple is None:
                # All triples are covered
                if cover_count < best_cover[0]:
                    best_cover[0] = cover_count
                return

            # Branch: pick one of the 3 vertices in uncovered_triple to add to cover
            # Choose the vertex that appears in the most uncovered triples
            vertices = []
            for v in range(n):
                if uncovered_triple & (1 << v):
                    vertices.append(v)

            # Score each vertex by how many uncovered triples it hits
            best_v = None
            best_hits = -1
            for v in vertices:
                hits = 0
                v_bit = 1 << v
                for mask in triple_masks:
                    if mask & v_bit and mask & remaining_mask == mask:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_v = v

            if best_v is None:
                return

            # Branch on each of the 3 vertices
            for v in vertices:
                v_bit = 1 << v
                new_remaining = remaining_mask & ~v_bit
                new_cover = cover_count + 1
                min_vertex_cover(new_remaining, new_cover)

        min_vertex_cover((1 << N) - 1, 0)
        return N - best_cover[0]

    def generate_line_pack():
        """
        Generate a point set by placing points on shared lines to create dense triple structures.
        Strategy: use a few lines, place 3 points on each, ensuring no 4 are collinear.
        """
        points = set()
        used_lines = []

        # Generate candidate lines with small integer coefficients
        candidate_lines = []
        for a in range(-3, 4):
            for b in range(-3, 4):
                if a == 0 and b == 0:
                    continue
                # Normalize: ensure gcd(a,b)=1 and a>0 or a==0,b>0
                from math import gcd
                g = gcd(abs(a), abs(b))
                if g > 1:
                    a, b = a // g, b // g
                if a < 0 or (a == 0 and b < 0):
                    a, b = -a, -b
                # Avoid duplicates
                key = (a, b)
                if key not in [l[:2] for l in candidate_lines]:
                    candidate_lines.append((a, b))

        random.shuffle(candidate_lines)

        for a, b in candidate_lines:
            if len(points) >= N:
                break

            # Find integer points on ax + by = c for various c
            # We want to pick 3 points that don't create 4-collinearities with existing points
            for c in range(-150, 151, 7):  # Skip some to avoid dense lines
                # Find integer points on this line
                pts_on_line = []
                if a != 0:
                    for y in range(-100, 101):
                        num = c - b * y
                        if num % a == 0:
                            x = num // a
                            if -100 <= x <= 100:
                                pts_on_line.append((x, y))
                elif b != 0:
                    for x in range(-100, 101):
                        num = c - a * x
                        if num % b == 0:
                            y = num // b
                            if -100 <= y <= 100:
                                pts_on_line.append((x, y))

                if len(pts_on_line) < 3:
                    continue

                # Try to pick 3 points from this line that don't conflict
                random.shuffle(pts_on_line)
                chosen = []
                for p in pts_on_line:
                    if p in points:
                        continue
                    # Check if adding p would create 4 collinear
                    test_pts = list(points) + chosen + [p]
                    if len(test_pts) >= 4:
                        # Check all 4-subsets for collinearity
                        valid = True
                        for combo in itertools.combinations(range(len(test_pts)), 4):
                            if all(are_collinear(test_pts[i], test_pts[j], test_pts[k]) 
                                   for i, j, k in [(combo[0], combo[1], combo[2]),
                                                    (combo[0], combo[1], combo[3]),
                                                    (combo[0], combo[2], combo[3]),
                                                    (combo[1], combo[2], combo[3])]):
                                valid = False
                                break
                        if not valid:
                            continue
                    chosen.append(p)
                    if len(chosen) == 3:
                        break

                if len(chosen) >= 1:
                    points.update(chosen)
                    if len(points) >= N:
                        break

        # Fill remaining with random points
        while len(points) < N:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in points:
                points.add(p)

        if len(points) > N:
            pts_list = list(points)
            random.shuffle(pts_list)
            points = set(pts_list[:N])

        return list(points)

    def fix_validity(points):
        """Fix 4-collinearities by moving points."""
        points = points[:]
        for _ in range(200):
            if is_valid(points):
                return points

            found = False
            for i in range(N - 3):
                for j in range(i + 1, N - 2):
                    for k in range(j + 1, N - 1):
                        p1, p2, p3 = points[i], points[j], points[k]
                        if not are_collinear(p1, p2, p3):
                            continue
                        for l in range(k + 1, N):
                            p4 = points[l]
                            if are_collinear(p1, p2, p4):
                                # Move point l
                                for _ in range(100):
                                    cand = (random.randint(-100, 100), random.randint(-100, 100))
                                    if cand not in points:
                                        points[l] = cand
                                        break
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

        return points

    # --- Main Search ---

    best_points = None
    best_score = N + 1

    # Generate candidates
    candidates = []
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        pts = generate_line_pack()
        pts = fix_validity(pts)
        if is_valid(pts):
            candidates.append(pts)

    if not candidates:
        # Fallback
        while True:
            pts = [tuple(random.randint(-100, 100) for _ in range(2)) for _ in range(N)]
            if len(set(pts)) == N and is_valid(pts):
                candidates.append(pts)
                break

    # Evaluate exact score for candidates
    for pts in candidates:
        if time.time() > deadline - 10:
            break
        e_score = calc_exact_score(pts)
        if e_score < best_score:
            best_score = e_score
            best_points = pts[:]

    if best_points is None:
        best_points = candidates[0][:]
        best_score = calc_exact_score(best_points)

    # Local search
    current_pts = best_points[:]
    current_score = best_score

    while time.time() < deadline - 5:
        idx = random.randint(0, N - 1)
        old_p = current_pts[idx]

        # Try to move point to create more collinear triples
        # Strategy: pick 2 other points and place idx on their line
        if random.random() < 0.7:
            j = random.randint(0, N - 1)
            k = random.randint(0, N - 1)
            if j == idx or k == idx or j == k:
                continue
            p_j, p_k = current_pts[j], current_pts[k]
            dx = p_k[0] - p_j[0]
            dy = p_k[1] - p_j[1]
            if dx == 0 and dy == 0:
                continue
            from math import gcd
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            step_x = dx // g
            step_y = dy // g
            # Try several positions on this line
            found = False
            for t in range(-8, 9):
                if t == 0:
                    continue
                cand_x = p_j[0] + t * step_x
                cand_y = p_j[1] + t * step_y
                if -100 <= cand_x <= 100 and -100 <= cand_y <= 100:
                    new_p = (cand_x, cand_y)
                    if new_p not in current_pts:
                        trial_pts = current_pts[:]
                        trial_pts[idx] = new_p
                        if is_valid(trial_pts):
                            e_score = calc_exact_score(trial_pts)
                            if e_score < current_score:
                                current_score = e_score
                                current_pts = trial_pts[:]
                                if current_score < best_score:
                                    best_score = current_score
                                    best_points = current_pts[:]
                            found = True
                            break
            if found:
                continue

        # Fallback: random move
        for _ in range(50):
            new_p = (random.randint(-100, 100), random.randint(-100, 100))
            if new_p not in current_pts:
                trial_pts = current_pts[:]
                trial_pts[idx] = new_p
                if is_valid(trial_pts):
                    e_score = calc_exact_score(trial_pts)
                    if e_score < current_score:
                        current_score = e_score
                        current_pts = trial_pts[:]
                        if current_score < best_score:
                            best_score = current_score
                            best_points = current_pts[:]
                break

    return [[p[0], p[1]] for p in best_points]

if __name__ == "__main__":
    res = construct()
    print(res)
