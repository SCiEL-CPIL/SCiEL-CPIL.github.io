import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for 120s limit

    def get_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        # Sign convention: first non-zero positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def compute_line_counts(points):
        """Compute line counts for all pairs of points."""
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        return line_counts

    def is_valid(points):
        """Check distinctness, bounds, and no 4 collinear."""
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
        
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        for line, count in line_counts.items():
            if count >= 4:
                return False
        return True

    def calc_score(points):
        """
        Compute the largest subset with no three collinear.
        Use a greedy approach: iteratively remove points that are part of the most collinear triples.
        """
        n = len(points)
        
        # Find all collinear triples
        triple_set = set()
        # Precompute lines for efficiency
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                if line not in line_map:
                    line_map[line] = []
                line_map[line].append((i, j))
        
        # For each line with >= 3 points, find the triples
        # Actually, we need to identify which triples are collinear
        # A triple (i,j,k) is collinear if line(i,j) == line(i,k)
        # Let's just check all triples directly but efficiently
        for i in range(n):
            for j in range(i + 1, n):
                li = get_line(points[i], points[j])
                for k in range(j + 1, n):
                    if get_line(points[i], points[k]) == li:
                        triple_set.add((i, j, k))
        
        # Now we want the maximum subset with no triple from triple_set
        # This is equivalent to: n - minimum vertex cover of the hypergraph
        
        # Greedy: repeatedly remove the point that is in the most triples
        remaining = set(range(n))
        while True:
            # Count triples for each remaining point
            triple_count = defaultdict(int)
            for (i, j, k) in triple_set:
                if i in remaining and j in remaining and k in remaining:
                    triple_count[i] += 1
                    triple_count[j] += 1
                    triple_count[k] += 1
            
            if not triple_count:
                break
            
            # Find the point with max triple count
            max_count = max(triple_count.values())
            candidates = [p for p in remaining if triple_count[p] == max_count]
            # Pick one (random or first)
            to_remove = candidates[0]
            remaining.remove(to_remove)
        
        return len(remaining)

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    # Initial configuration: Random
    current_points = None
    for _ in range(100):
        pts = []
        while len(pts) < 20:
            p = random_point()
            if p not in pts:
                pts.append(p)
        if is_valid(pts):
            current_points = pts
            break
    
    if current_points is None:
        # Fallback to parabola if random fails (unlikely)
        current_points = [[x, x * x] for x in range(-10, 10)]
        
    current_score = calc_score(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local Search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        
        improved = False
        
        # Heuristic: Try to form a triple with two existing points.
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                p1 = current_points[i]
                p2 = current_points[j]
                
                # Check count of points on line p1-p2
                line = get_line(p1, p2)
                cnt = 0
                for k in range(20):
                    if k == i or k == j or k == idx:
                        continue
                    if get_line(p1, current_points[k]) == line:
                        cnt += 1
                if cnt == 0:
                    # Line has only p1 and p2 (and possibly the point we're moving).
                    # We can place a new point on this line to form a triple.
                    # Generate a point on the line.
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Try small integer steps
                    found = False
                    for t in range(-15, 16):
                        if t == 0 or t == 1:
                            continue
                        nx = p1[0] + t * dx
                        ny = p1[1] + t * dy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            new_candidate = [nx, ny]
                            if new_candidate not in current_points:
                                test_points = current_points[:]
                                test_points[idx] = new_candidate
                                if is_valid(test_points):
                                    test_score = calc_score(test_points)
                                    if test_score < current_score:
                                        current_points = test_points
                                        current_score = test_score
                                        if current_score < best_score:
                                            best_score = current_score
                                            best_points = [p[:] for p in current_points]
                                        improved = True
                                        break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            # Try random move
            new_p = random_point()
            if new_p not in current_points:
                test_points = current_points[:]
                test_points[idx] = new_p
                if is_valid(test_points):
                    test_score = calc_score(test_points)
                    if test_score < current_score:
                        current_points = test_points
                        current_score = test_score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                    else:
                        # Accept with some probability to escape local optima?
                        # Since we want to minimize, and this is a hard problem,
                        # maybe we should accept equal scores sometimes?
                        # But the greedy score is a lower bound on the true score?
                        # No, the greedy score is an upper bound on the true score (it finds *a* valid subset, not necessarily the largest).
                        # Wait, the greedy algorithm finds a subset with no 3 collinear.
                        # So it returns a value S such that there exists a subset of size S.
                        # The true score is the MAXIMUM such S.
                        # So greedy_score <= true_score.
                        # We want to MINIMIZE the true score.
                        # So if greedy_score increases, true_score might increase.
                        # If greedy_score decreases, true_score might decrease.
                        # Actually, if greedy finds a larger subset, it means the configuration is "worse" (higher score).
                        # So we want to minimize the greedy score as a proxy for the true score.
                        
                        # Let's just stick to strict improvement for now to avoid noise.
                        pass
    
    # Final check
    if is_valid(best_points):
        return best_points
    else:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
