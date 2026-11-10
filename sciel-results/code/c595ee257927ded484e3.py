"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def check_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        return False
        return True

    def check_concyclic(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        a1, b1, c1 = x1, y1, x1 * x1 + y1 * y1
                        a2, b2, c2 = x2, y2, x2 * x2 + y2 * y2
                        a3, b3, c3 = x3, y3, x3 * x3 + y3 * y3
                        a4, b4, c4 = x4, y4, x4 * x4 + y4 * y4
                        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
                               - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
                               + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
                               - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
                        if det == 0:
                            return False
        return True

    def validate(points):
        n = len(points)
        if n < 3:
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if not is_square(dist_sq(points[i], points[j])):
                    return False
        if not check_collinear(points):
            return False
        if not check_concyclic(points):
            return False
        return True

    # Known 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_count = 5

    # Generate all Pythagorean triples up to a limit
    def gen_pyth_triples(max_c=2000):
        triples = []
        for m in range(2, 200):
            for n in range(1, m):
                if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    if c <= max_c:
                        triples.append((a, b, c))
                        triples.append((b, a, c))
        # Remove duplicates
        seen = set()
        unique = []
        for t in triples:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    triples = gen_pyth_triples(1500)

    def generate_candidates(existing, max_coord=1000):
        candidates = set()
        for (px, py) in existing:
            for (a, b, c) in triples:
                for sign_x in [1, -1]:
                    for sign_y in [1, -1]:
                        candidates.add((px + sign_x * a, py + sign_y * b))
                        candidates.add((px + sign_x * b, py + sign_y * a))
        
        existing_set = set(existing)
        filtered = []
        for (x, y) in candidates:
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                if (x, y) not in existing_set:
                    filtered.append((x, y))
        return filtered

    def try_add_point(points, candidate):
        new_points = points + [candidate]
        return validate(new_points)

    # Incremental search
    current = list(best_points)
    
    while time.time() < deadline:
        if len(current) >= 12:
            break
        
        cands = generate_candidates(current, max_coord=800)
        
        # Sort by minimum distance to existing points (prefer farther)
        def candidate_score(cand):
            min_dist = float('inf')
            for p in current:
                d = dist_sq(cand, p)
                if d < min_dist:
                    min_dist = d
            return -min_dist
        
        cands.sort(key=candidate_score)
        
        added = False
        for cand in cands:
            if time.time() > deadline:
                break
            if try_add_point(current, cand):
                current.append(cand)
                added = True
                if len(current) > best_count:
                    best_points = list(current)
                    best_count = len(current)
                break
        
        if not added:
            # Try a different approach: search in a grid around the existing points
            import random
            random.seed(len(current) * 104729)
            
            # Try points in a wider range with random sampling
            improved = False
            for _ in range(500):
                if time.time() > deadline:
                    break
                # Pick a random existing point and generate a candidate nearby
                anchor = random.choice(current)
                dx = random.randint(-200, 200)
                dy = random.randint(-200, 200)
                cand = (anchor[0] + dx, anchor[1] + dy)
                if any(cand == p for p in current):
                    continue
                if try_add_point(current, cand):
                    current.append(cand)
                    improved = True
                    if len(current) > best_count:
                        best_points = list(current)
                        best_count = len(current)
                    break
            
            if not improved:
                break

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
