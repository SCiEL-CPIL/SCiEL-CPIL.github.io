"""
Strategy: Greedy construction to maximize collinear triples (minimizing the
maximum subset with no three collinear), combined with bitmask branch-and-bound
for exact score evaluation and local search refinement.

The key structural change from the seed (parabola, score 20) is to explicitly
build configurations where many triples share points, so that any large subset
must contain a collinear triple. We do this by placing points on a small number
of lines (with no 4 collinear on any line), then perturbing to avoid 4-collinear
violations, and using local search to improve.
"""
import random
import time
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
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
    
    def get_triples(points):
        n = len(points)
        triples = []
        for i, j, k in combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                triples.append((i, j, k))
        return triples
    
    def count_triples(points):
        return len(get_triples(points))
    
    def exact_score(points):
        """Exact branch and bound for max subset with no 3 collinear."""
        n = len(points)
        triples = get_triples(points)
        
        # For each point, store which pairs (j,k) with it form a triple
        conflicts = [[] for _ in range(n)]
        for i, j, k in triples:
            conflicts[i].append((j, k))
            conflicts[j].append((i, k))
            conflicts[k].append((i, j))
        
        # Order by degree (most constrained first)
        order = sorted(range(n), key=lambda i: -len(conflicts[i]))
        
        # Remap
        remap = [0] * n
        for new_idx, old_idx in enumerate(order):
            remap[old_idx] = new_idx
        
        new_conflicts = [[] for _ in range(n)]
        for i in range(n):
            for j, k in conflicts[i]:
                new_conflicts[remap[i]].append((remap[j], remap[k]))
        
        best = [0]
        
        def bb(idx, mask, count):
            if count + (n - idx) <= best[0]:
                return
            if idx == n:
                if count > best[0]:
                    best[0] = count
                return
            
            # Try including point idx
            valid = True
            for j, k in new_conflicts[idx]:
                if (mask & (1 << j)) and (mask & (1 << k)):
                    valid = False
                    break
            
            if valid:
                bb(idx + 1, mask | (1 << idx), count + 1)
            
            # Try excluding
            bb(idx + 1, mask, count)
        
        bb(0, 0, 0)
        return best[0]
    
    def generate_line_based_config():
        """
        Generate points placed on a small number of lines (3 per line to avoid
        4-collinear), then add remaining points. This creates many triples that
        share lines, making it hard to pick a large subset without a triple.
        """
        random.seed(random.randint(0, 10**6))
        points = []
        seen = set()
        
        # Choose 5-7 lines, place 3 points on each (15-21 points)
        num_lines = random.randint(5, 7)
        points_per_line = 3
        
        for _ in range(num_lines):
            # Random line: y = mx + b with small integer slope
            m = random.choice([-2, -1, 0, 1, 2])
            b = random.randint(-60, 60)
            
            # Pick 3 distinct x values
            xs = random.sample(range(-90, 91), points_per_line)
            xs.sort()
            
            for x in xs:
                y = m * x + b
                if -100 <= y <= 100:
                    p = (x, y)
                    if p not in seen:
                        points.append(p)
                        seen.add(p)
        
        # Add remaining points randomly
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in seen:
                points.append(p)
                seen.add(p)
        
        points = points[:20]
        
        # Check validity
        if len(set(points)) < 20:
            return None
        if has_4_collinear(points):
            return None
        
        return [list(p) for p in points]
    
    def generate_grid_perturbed():
        """Generate a perturbed grid that avoids 4-collinear."""
        random.seed(random.randint(0, 10**6))
        points = []
        seen = set()
        
        # 4x5 grid base
        for i in range(5):
            for j in range(4):
                x = i * 20 + random.randint(-4, 4)
                y = j * 20 + random.randint(-4, 4)
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                p = (x, y)
                if p not in seen:
                    points.append(p)
                    seen.add(p)
        
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in seen:
                points.append(p)
                seen.add(p)
        
        points = points[:20]
        
        if len(set(points)) < 20 or has_4_collinear(points):
            return None
        return [list(p) for p in points]
    
    def generate_parabola_perturbed():
        """Points near a parabola with perturbations to create some collinearities."""
        random.seed(random.randint(0, 10**6))
        points = []
        seen = set()
        
        for x in range(-10, 10):
            base_y = x * x
            y = base_y + random.randint(-3, 3)
            y = max(-100, min(100, y))
            p = (x, y)
            if p not in seen:
                points.append(p)
                seen.add(p)
        
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in seen:
                points.append(p)
                seen.add(p)
        
        points = points[:20]
        
        if len(set(points)) < 20 or has_4_collinear(points):
            return None
        return [list(p) for p in points]
    
    def local_search(points, time_budget=5):
        """
        Local search: try swapping points to reduce the exact score.
        A swap replaces one point with a random new point.
        """
        if elapsed() > time_budget + start_time:
            return points
        
        current_points = [list(p) for p in points]
        current_score = exact_score(current_points)
        best_points = [list(p) for p in current_points]
        best_score = current_score
        
        n = len(current_points)
        
        while elapsed() < time_budget + start_time:
            # Try a random swap
            idx = random.randint(0, n - 1)
            old_p = current_points[idx]
            
            # Generate new point
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            # Try a few candidates
            improved = False
            for attempt in range(20):
                cand_x = random.randint(-100, 100)
                cand_y = random.randint(-100, 100)
                cand = [cand_x, cand_y]
                
                # Check distinctness
                if any(cand == p for p in current_points):
                    continue
                
                # Check no 4 collinear
                test_points = current_points[:]
                test_points[idx] = cand
                if has_4_collinear(test_points):
                    continue
                
                test_score = exact_score(test_points)
                if test_score < current_score:
                    current_points = test_points
                    current_score = test_score
                    improved = True
                    break
                elif test_score == current_score and random.random() < 0.3:
                    # Accept equal with some probability (simulated annealing lite)
                    current_points = test_points
                    improved = True
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_points = [list(p) for p in current_points]
            
            if best_score <= 10:  # Good enough
                break
        
        return best_points
    
    def local_search2(points, time_budget=5):
        """
        More aggressive local search: try moving a point along a line to create
        new collinearities, or try replacing pairs of points.
        """
        if elapsed() > time_budget + start_time:
            return points
        
        current_points = [list(p) for p in points]
        current_score = exact_score(current_points)
        best_points = [list(p) for p in current_points]
        best_score = current_score
        
        n = len(current_points)
        deadline_local = start_time + time_budget
        
        while time.time() < deadline_local:
            # Strategy 1: Try to create a new triple by moving one point
            idx = random.randint(0, n - 1)
            p = current_points[idx]
            
            # Find two other points and try to make idx collinear with them
            for j, k in combinations(range(n), 2):
                if j == idx or k == idx:
                    continue
                p_j = current_points[j]
                p_k = current_points[k]
                
                # Line through p_j and p_k
                dx = p_k[0] - p_j[0]
                dy = p_k[1] - p_j[1]
                
                # Find integer points on this line (excluding p_j, p_k)
                # Parametric: (p_j[0] + t*dx/g, p_j[1] + t*dy/g) for integer t
                g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                if g == 0:
                    continue
                step_x = dx // g
                step_y = dy // g
                
                # Try a few t values
                for t in range(-30, 31):
                    if t == 0:
                        continue
                    nx = p_j[0] + t * step_x
                    ny = p_j[1] + t * step_y
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        cand = [nx, ny]
                        if any(cand == q for q in current_points):
                            continue
                        
                        test_points = current_points[:]
                        test_points[idx] = cand
                        if has_4_collinear(test_points):
                            continue
                        
                        test_score = exact_score(test_points)
                        if test_score < current_score:
                            current_points = test_points
                            current_score = test_score
                            break
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
                if best_score <= 10:
                    return best_points
            else:
                continue
            break
        
        return best_points
    
    # Main search loop
    best_config = None
    best_score = 20  # Worst case: no triples at all
    
    generators = [generate_line_based_config, generate_grid_perturbed, generate_parabola_perturbed]
    
    iteration = 0
    while elapsed() < deadline - 20:  # Leave 20s for final evaluation
        # Generate a candidate
        gen = generators[iteration % len(generators)]
        candidate = gen()
        
        if candidate is None:
            iteration += 1
            continue
        
        # Evaluate
        score = exact_score(candidate)
        
        if score < best_score:
            best_score = score
            best_config = candidate
        
        # Local search if we have time
        if elapsed() < deadline - 30:
            improved = local_search(candidate, time_budget=min(5, deadline - elapsed() - 25))
            improved_score = exact_score(improved)
            if improved_score < best_score:
                best_score = improved_score
                best_config = improved
            
            if elapsed() < deadline - 25:
                improved2 = local_search2(candidate, time_budget=min(5, deadline - elapsed() - 20))
                improved2_score = exact_score(improved2)
                if improved2_score < best_score:
                    best_score = improved2_score
                    best_config = improved2
        
        if best_score <= 10:
            break
        
        iteration += 1
    
    # Final fallback: if no valid config found, use a simple valid one
    if best_config is None:
        # Use a known valid configuration: points on a circle (no 3 collinear for generic circle)
        # But we need no 4 collinear. Let's use a simple valid set.
        # Actually, let's just generate one more time deterministically
        random.seed(42)
        for _ in range(100):
            candidate = generate_line_based_config()
            if candidate is not None:
                best_config = candidate
                break
    
    if best_config is None:
        # Last resort: parabola (valid, score 20)
        best_config = [[x, x * x] for x in range(-10, 10)]
    
    # Final validation
    assert len(best_config) == 20
    assert len(set(tuple(p) for p in best_config)) == 20
    assert all(abs(p[0]) <= 100 and abs(p[1]) <= 100 for p in best_config)
    assert not has_4_collinear(best_config)
    
    return best_config
