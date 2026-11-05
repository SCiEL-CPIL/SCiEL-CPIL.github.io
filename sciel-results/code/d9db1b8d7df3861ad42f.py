import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def are_three_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])

    def is_valid(points):
        # No 4 collinear
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_three_collinear(points[i], points[j], points[k]):
                        for m in range(k+1, n):
                            if are_three_collinear(points[i], points[j], points[m]):
                                return False
        return True

    def calc_score(points):
        n = len(points)
        # Precompute for each vertex v, the set of pairs (u,w) that form a triple with v
        # Use bitmasks: for vertex v, conflict_mask[v] is not quite right because constraint is on pairs
        # Better: for each vertex v, store list of (u, w) pairs such that {v,u,w} is collinear
        # To check if v can join chosen: no pair (u,w) in chosen with both u,w in chosen
        
        # Precompute pairs_for_vertex[v] = list of (u, w) where {v,u,w} collinear
        pairs_for_vertex = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_three_collinear(points[i], points[j], points[k]):
                        pairs_for_vertex[i].append((j, k))
                        pairs_for_vertex[j].append((i, k))
                        pairs_for_vertex[k].append((i, j))
        
        # Sort by number of pairs (descending) for better pruning
        order = sorted(range(n), key=lambda v: len(pairs_for_vertex[v]), reverse=True)
        
        best_size = 0
        chosen = [False] * n
        
        def backtrack(idx, current_size):
            nonlocal best_size
            if idx == n:
                if current_size > best_size:
                    best_size = current_size
                return
            if current_size + (n - idx) <= best_size:
                return
            
            v = order[idx]
            
            # Try include
            valid = True
            for u, w in pairs_for_vertex[v]:
                if chosen[u] and chosen[w]:
                    valid = False
                    break
            if valid:
                chosen[v] = True
                backtrack(idx + 1, current_size + 1)
                chosen[v] = False
            
            # Try exclude
            backtrack(idx + 1, current_size)
        
        backtrack(0, 0)
        return best_size

    def generate_parabola_based():
        # Place points near a parabola to create structure
        points = []
        # Use a few lines with 3 points each to create many collinear triples
        # But ensure no 4 collinear
        # Strategy: pick several lines, put 3 points on each, ensure no 4 on any line
        # For 20 points: use about 7 lines with 3 points each (21 slots, but some overlap)
        
        # Simpler: random but biased
        attempts = 0
        while len(points) < 20 and attempts < 5000:
            attempts += 1
            # Pick a random line direction and offset
            # Use small integer slopes
            slope = random.choice([0, 1, -1, 2, -2, 3, -3, 1000000000])  # last = vertical
            if slope == 0:
                # horizontal: y = c
                c = random.randint(-50, 50)
                xs = random.sample(range(-100, 101), 3)
                candidates = [(x, c) for x in xs]
            elif slope == 1000000000:
                # vertical: x = c
                c = random.randint(-50, 50)
                ys = random.sample(range(-100, 101), 3)
                candidates = [(c, y) for y in ys]
            else:
                # y = slope * x + c
                c = random.randint(-50, 50)
                xs = random.sample(range(-30, 31), 3)
                candidates = []
                for x in xs:
                    y = slope * x + c
                    if -100 <= y <= 100:
                        candidates.append((x, y))
                if len(candidates) < 3:
                    continue
            
            # Add candidates if they don't create 4-collinear with existing
            new_points = []
            for p in candidates:
                if p in points:
                    continue
                # Check if adding p creates 4-collinear
                ok = True
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        for k in range(j+1, len(points)):
                            if are_three_collinear(points[i], points[j], points[k]):
                                if are_three_collinear(points[i], points[j], p) or \
                                   are_three_collinear(points[i], points[k], p) or \
                                   are_three_collinear(points[j], points[k], p):
                                    ok = False
                                    break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    new_points.append(p)
            
            points.extend(new_points)
        
        # Fill remaining randomly
        while len(points) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in points:
                # Quick check for 4-collinear
                ok = True
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        if are_three_collinear(points[i], points[j], p):
                            for k in range(j+1, len(points)):
                                if are_three_collinear(points[i], points[k], p):
                                    ok = False
                                    break
                            if not ok:
                                break
                    if not ok:
                        break
                if ok:
                    points.append(p)
        
        return points[:20]

    def local_search(points, budget):
        points = list(points)
        best_score = calc_score(points)
        current_score = best_score
        
        while time.time() < budget:
            improved = False
            for i in range(20):
                if time.time() >= budget:
                    break
                old_p = points[i]
                # Try several random replacements
                for attempt in range(20):
                    if time.time() >= budget:
                        break
                    new_p = (random.randint(-100, 100), random.randint(-100, 100))
                    if new_p in points:
                        continue
                    points[i] = new_p
                    if is_valid(points):
                        s = calc_score(points)
                        if s < current_score:
                            current_score = s
                            if s < best_score:
                                best_score = s
                            improved = True
                            break
                    points[i] = old_p
                if improved:
                    break
            if not improved:
                # Try more aggressive: random restart from current
                if time.time() < budget - 5:
                    # Perturb 2-3 points
                    idxs = random.sample(range(20), random.randint(2, 3))
                    for i in idxs:
                        points[i] = (random.randint(-100, 100), random.randint(-100, 100))
                    if is_valid(points):
                        s = calc_score(points)
                        if s < current_score:
                            current_score = s
                            if s < best_score:
                                best_score = s
                            improved = True
        return best_score, points

    best_score = 20
    best_points = None

    # Generate multiple initial candidates
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        points = generate_parabola_based()
        if len(points) == 20 and is_valid(points):
            s = calc_score(points)
            if s < best_score:
                best_score = s
                best_points = list(points)

    # Local search
    if best_points is not None and time.time() < deadline - 10:
        budget = deadline - 10
        best_score, best_points = local_search(best_points, budget)

    # Final: try a few more random starts if time allows
    while time.time() < deadline - 5:
        points = generate_parabola_based()
        if len(points) == 20 and is_valid(points):
            s = calc_score(points)
            if s < best_score:
                best_score = s
                best_points = list(points)

    if best_points is None:
        # Fallback: parabola
        best_points = [(x, x*x) for x in range(-10, 10)]

    return [[p[0], p[1]] for p in best_points]
