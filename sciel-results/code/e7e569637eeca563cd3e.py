import time
import random
from collections import defaultdict
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

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
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def is_valid(points):
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
        for count in line_counts.values():
            if count >= 4:
                return False
        return True

    def build_hypergraph(points):
        """Build the 3-uniform hypergraph of collinear triples."""
        n = len(points)
        # Find all lines with >= 3 points
        line_points = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_points[line].append((i, j))
        
        # For each line, find the set of points on it
        triples = set()
        for line, pairs in line_points.items():
            # Collect all points on this line
            pts_on_line = set()
            for i, j in pairs:
                pts_on_line.add(i)
                pts_on_line.add(j)
            pts_on_line = list(pts_on_line)
            if len(pts_on_line) >= 3:
                for triple in combinations(sorted(pts_on_line), 3):
                    triples.add(triple)
        
        return list(triples)

    def exact_max_3_collinear_free(points):
        """
        Exact branch and bound to find the largest subset with no three collinear.
        This is the independence number of the 3-uniform hypergraph.
        """
        n = len(points)
        triples = build_hypergraph(points)
        
        if not triples:
            return n
        
        # Convert to a set of frozensets for fast lookup
        triple_set = set(frozenset(t) for t in triples)
        
        # Precompute for each point, which triples it's in
        point_triples = defaultdict(list)
        for i, t in enumerate(triples):
            for v in t:
                point_triples[v].append(i)
        
        # Branch and bound
        best = [0]
        
        def search(remaining, current_size, excluded_triples):
            if time.time() > deadline - 5:
                return
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current_size + len(remaining)
            if current_size + len(remaining) <= best[0]:
                return
            
            # Pick a vertex to branch on
            # Heuristic: pick the vertex with the most remaining triples
            v = max(remaining, key=lambda x: len([t for t in excluded_triples if x in t]))
            
            # Branch 1: include v
            # Remove all triples containing v
            new_excluded = set()
            for t in excluded_triples:
                if v not in t:
                    new_excluded.add(t)
            new_remaining = set(remaining)
            new_remaining.discard(v)
            search(new_remaining, current_size + 1, new_excluded)
            
            # Branch 2: exclude v
            new_remaining = set(remaining)
            new_remaining.discard(v)
            search(new_remaining, current_size, excluded_triples)
        
        search(set(range(n)), 0, set(triple_set))
        return best[0]

    def greedy_vertex_cover(points):
        """
        Greedy approximation of minimum vertex cover of the 3-uniform hypergraph.
        Returns the size of the cover, so score = n - cover_size.
        """
        n = len(points)
        triples = build_hypergraph(points)
        
        if not triples:
            return n
        
        remaining_triples = [set(t) for t in triples]
        removed = 0
        remaining_vertices = set(range(n))
        
        while remaining_triples:
            counts = defaultdict(int)
            for t in remaining_triples:
                for v in t:
                    if v in remaining_vertices:
                        counts[v] += 1
            
            if not counts:
                break
            
            max_count = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_count]
            v_to_remove = min(candidates)
            
            remaining_vertices.discard(v_to_remove)
            removed += 1
            
            new_remaining = []
            for t in remaining_triples:
                if v_to_remove not in t:
                    new_remaining.append(t)
            remaining_triples = new_remaining
        
        return n - removed

    def calc_score(points):
        """Compute the score: largest subset with no three collinear."""
        # Use greedy for speed during search, exact for final
        return greedy_vertex_cover(points)

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    # Initialize with a structured configuration that creates many triples
    # Strategy: Use a grid-like pattern with deliberate collinearities
    def make_initial_config():
        # Try to create many 3-point lines
        # Use a pattern based on small integer coordinates
        configs = []
        
        # Configuration 1: Modified grid with some collinearities
        # 4x5 grid with some points shifted to create triples
        base = []
        for x in range(-4, 6):
            for y in range(-3, 3):
                base.append([x, y])
        # Shuffle and take 20
        random.shuffle(base)
        configs.append(base[:20])
        
        # Configuration 2: Lines of 3
        # Create 6 lines of 3 points each, sharing some points
        # Line 1: (0,0), (1,0), (2,0)
        # Line 2: (0,1), (1,1), (2,1)
        # Line 3: (0,2), (1,2), (2,2)
        # Line 4: (3,0), (3,1), (3,2)
        # Line 5: (4,0), (4,1), (4,2)
        # Line 6: (0,3), (1,3), (2,3)
        # That's 18 points, need 2 more
        configs.append([
            [0,0], [1,0], [2,0],
            [0,1], [1,1], [2,1],
            [0,2], [1,2], [2,2],
            [3,0], [3,1], [3,2],
            [4,0], [4,1], [4,2],
            [0,3], [1,3], [2,3],
            [5,0], [5,1]
        ])
        
        # Configuration 3: More dense collinear structure
        # Use a 5x4 grid
        configs.append([[x, y] for x in range(-2, 3) for y in range(-2, 2)][:20])
        
        # Filter valid configs
        valid_configs = []
        for cfg in configs:
            if len(cfg) == 20 and is_valid(cfg):
                valid_configs.append(cfg)
        
        if valid_configs:
            return random.choice(valid_configs)
        
        # Fallback: random
        pts = []
        for _ in range(100):
            pts = []
            while len(pts) < 20:
                p = random_point()
                if p not in pts:
                    pts.append(p)
            if is_valid(pts):
                return pts
        
        # Last resort: parabola (score 20, but valid)
        return [[x, x * x] for x in range(-10, 10)]

    current_points = make_initial_config()
    current_score = calc_score(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score

    # Local search with triple-creating moves
    T = 1.0
    cooling_rate = 0.9995

    while time.time() < deadline:
        idx = random.randint(0, 19)
        
        # Strategy: Try to place point on a line with exactly 2 other points
        # to create a new triple
        others = [k for k in range(20) if k != idx]
        random.shuffle(others)
        
        candidate = None
        found = False
        
        # Try to create a triple with two other points
        for i in others:
            if found:
                break
            for j in others:
                if found:
                    break
                if i == j:
                    continue
                
                p1 = current_points[i]
                p2 = current_points[j]
                line = get_line(p1, p2)
                
                # Check how many points are on this line (excluding idx)
                cnt = 0
                for k in range(20):
                    if k == i or k == j or k == idx:
                        continue
                    if get_line(p1, current_points[k]) == line:
                        cnt += 1
                
                # We want cnt == 0 so that adding idx makes a triple (3 points)
                # If cnt == 1, adding idx makes 4 points - invalid
                if cnt == 0:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
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
                                    candidate = test_points
                                    found = True
                                    break
        
        if not found:
            # Try random move
            new_p = random_point()
            if new_p not in current_points:
                test_points = current_points[:]
                test_points[idx] = new_p
                if is_valid(test_points):
                    candidate = test_points
        
        if candidate is None:
            T *= cooling_rate
            continue
        
        candidate_score = calc_score(candidate)
        
        if candidate_score < current_score:
            current_points = candidate
            current_score = candidate_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            delta_e = candidate_score - current_score
            if delta_e == 0 or random.random() < pow(2.718281828, -delta_e / T):
                current_points = candidate
                current_score = candidate_score
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Final: compute exact score for the best configuration
    # But we need to return the points, and the evaluator will compute the score
    # So we just return the best points we found
    
    if is_valid(best_points):
        return best_points
    else:
        return [[x, x * x] for x in range(-10, 10)]
