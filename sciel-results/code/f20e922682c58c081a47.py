import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 145  # Leave 5s margin before 150s limit
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def count_max_collinear(points):
        n = len(points)
        max_count = 2
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count > max_count:
                    max_count = count
        return max_count
    
    def has_4_collinear(points):
        return count_max_collinear(points) >= 4
    
    def count_triples(points):
        n = len(points)
        triples = 0
        for i, j, k in combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                triples += 1
        return triples
        
    def get_triples_list(points):
        n = len(points)
        triples = []
        for i, j, k in combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                triples.append((i, j, k))
        return triples
    
    def exact_independence_number(points):
        n = len(points)
        triples = get_triples_list(points)
        
        pairs_with = [[] for _ in range(n)]
        for i, j, k in triples:
            pairs_with[i].append((j, k))
            pairs_with[j].append((i, k))
            pairs_with[k].append((i, j))
        
        degree = [len(pairs_with[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -degree[i])
        
        new_pairs_with = [[] for _ in range(n)]
        for i, j, k in triples:
            ni = order.index(i)
            nj = order.index(j)
            nk = order.index(k)
            new_pairs_with[ni].append((nj, nk))
            new_pairs_with[nj].append((ni, nk))
            new_pairs_with[nk].append((ni, nj))
        
        best = 0
        
        def bb(idx, current_mask, current_sum):
            nonlocal best
            if current_sum + (n - idx) <= best:
                return
            if idx == n:
                if current_sum > best:
                    best = current_sum
                return
            valid = True
            for j, k in new_pairs_with[idx]:
                if (current_mask & (1 << j)) and (current_mask & (1 << k)):
                    valid = False
                    break
            if valid:
                bb(idx + 1, current_mask | (1 << idx), current_sum + 1)
            bb(idx + 1, current_mask, current_sum)
        
        bb(0, 0, 0)
        return best

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_perturbed_grid():
        """
        Generate a 4x5 grid with perturbations to break 4-collinearities.
        """
        points = []
        seen = set()
        # Base grid
        for x in range(5):
            for y in range(4):
                # Perturb to break collinearity
                px = x * 20 + random.randint(-5, 5)
                py = y * 20 + random.randint(-5, 5)
                px = max(-100, min(100, px))
                py = max(-100, min(100, py))
                p = [px, py]
                if (p[0], p[1]) not in seen:
                    points.append(p)
                    seen.add((p[0], p[1]))
        
        while len(points) < 20:
            p = random_point()
            if (p[0], p[1]) not in seen:
                points.append(p)
                seen.add((p[0], p[1]))
        
        points = points[:20]
        
        # Fix 4-collinearities by moving points
        for _ in range(100):
            if not has_4_collinear(points):
                break
            # Find a line with 4+ points
            found = False
            for i in range(20):
                for j in range(i+1, 20):
                    line_pts = [i, j]
                    for k in range(20):
                        if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                            line_pts.append(k)
                    if len(line_pts) >= 4:
                        # Move one of the points (not i or j, to keep the line base)
                        idx_to_move = random.choice([p for p in line_pts if p != i and p != j])
                        points[idx_to_move] = random_point()
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            return None
        return points

    best_points = None
    best_score = 100
    
    # Phase 1: Try structured initializations
    for _ in range(50):
        if time.time() - start_time > timeout - 30:
            break
        pts = generate_perturbed_grid()
        if pts:
            triples = count_triples(pts)
            # Use triples as a proxy for low independence number
            # More triples usually means lower independence number
            score = exact_independence_number(pts)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in pts]
            if best_score <= 12: # Good enough, break early if we hit theoretical bounds
                break

    if best_points is None:
        # Fallback: Random
        while True:
            pts = [random_point() for _ in range(20)]
            if len(set(map(tuple, pts))) == 20 and not has_4_collinear(pts):
                best_points = pts
                best_score = exact_independence_number(pts)
                break

    # Phase 2: Local Search to improve
    current_points = [p[:] for p in best_points]
    current_score = best_score
    current_triples = count_triples(current_points)
    
    def try_move(i, new_p):
        nonlocal current_points, current_score, current_triples
        orig = current_points[i]
        current_points[i] = new_p
        if len(set(map(tuple, current_points))) < 20 or has_4_collinear(current_points):
            current_points[i] = orig
            return False
        new_triples = count_triples(current_points)
        # Heuristic: accept if triples increase
        # We verify exact score periodically
        if new_triples > current_triples:
            current_triples = new_triples
            return True
        elif new_triples == current_triples:
            # Sideways move, accept with prob 0.2 to escape plateaus
            if random.random() < 0.2:
                return True
        current_points[i] = orig
        return False

    improved = True
    while improved and time.time() - start_time < timeout - 10:
        improved = False
        # Shuffle indices for randomization
        indices = list(range(20))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > timeout - 10:
                break
            # Try several random moves
            for _ in range(10):
                new_p = random_point()
                if new_p == current_points[i]:
                    continue
                if try_move(i, new_p):
                    improved = True
                    # Check exact score occasionally
                    if random.random() < 0.1:
                        new_score = exact_independence_number(current_points)
                        if new_score < best_score:
                            best_score = new_score
                            best_points = [p[:] for p in current_points]
                    break # Restart loop for this i? Or continue? Let's break to re-evaluate neighbors
    
    # Final verification and update
    final_score = exact_independence_number(current_points)
    if final_score < best_score:
        best_score = final_score
        best_points = [p[:] for p in current_points]

    return best_points
