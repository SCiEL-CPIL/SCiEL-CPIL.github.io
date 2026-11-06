import time
import random
import sys

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin
    
    # Initialize with a random configuration or a structured one
    # Let's try to start with a grid perturbed to avoid 4-collinear
    # Grid 5x4 = 20 points
    # Points: (i, j) for i in 0..3, j in 0..4
    # This has 4-collinear rows and cols.
    # We need to break them.
    
    def get_initial_points():
        points = []
        # Use a "twisted" grid
        # x = i, y = j + i^2 ?
        # Let's just use random points first, then refine
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if [x, y] not in points:
                points.append([x, y])
        return points

    def is_valid(points):
        # Check no 4 collinear
        n = len(points)
        # Use a set of lines to detect 4-collinear
        # Line defined by normalized (A, B, C) for Ax + By + C = 0
        # Or just check all 4-tuples? \binom{20}{4} = 4845.
        # For each 4-tuple, check if any 3 are collinear and the 4th is on the same line.
        # Faster: For each pair, find all points on the line. If count >= 4, invalid.
        # Pairs: \binom{20}{2} = 190.
        # For each pair, check other 18 points.
        # Total 190 * 18 = 3420 ops. Very fast.
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                count = 2 # i and j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        count += 1
                if count >= 4:
                    return False
        return True

    def get_collinear_triples(points):
        n = len(points)
        triples = set()
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Find all k on this line
                line_points = [i, j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        line_points.append(k)
                # If line has exactly 3 points, add the triple
                if len(line_points) == 3:
                    triple = tuple(sorted(line_points))
                    triples.add(triple)
                # If line has > 3, it's invalid, but we check validity separately.
                # If we are in a valid config, max count is 3.
        return triples

    def calc_score(triples):
        # Min vertex cover in 3-uniform hypergraph
        # Vertices 0..19
        # Branch and bound
        best_cover_size = 20
        
        def solve(remaining_edges, covered, current_size):
            nonlocal best_cover_size
            if not remaining_edges:
                if current_size < best_cover_size:
                    best_cover_size = current_size
                return
            
            # Lower bound: current_size + ceil(len(remaining_edges) / 3) ?
            # Each vertex covers at most 3 edges? No, degree can be higher.
            # Simple bound: current_size < best_cover_size
            
            # Pick an edge
            edge = next(iter(remaining_edges))
            v1, v2, v3 = edge
            
            # Option 1: Include v1
            new_edges_1 = set()
            for e in remaining_edges:
                if v1 not in e:
                    new_edges_1.add(e)
            solve(new_edges_1, covered, current_size + 1)
            
            # Option 2: Include v2
            new_edges_2 = set()
            for e in remaining_edges:
                if v2 not in e:
                    new_edges_2.add(e)
            solve(new_edges_2, covered, current_size + 1)
            
            # Option 3: Include v3
            new_edges_3 = set()
            for e in remaining_edges:
                if v3 not in e:
                    new_edges_3.add(e)
            solve(new_edges_3, covered, current_size + 1)

        solve(triples, set(), 0)
        return 20 - best_cover_size

    def evaluate(points):
        if not is_valid(points):
            return 100 # Invalid
        triples = get_collinear_triples(points)
        return calc_score(triples)

    # Search
    best_points = get_initial_points()
    best_score = evaluate(best_points)
    
    # Local Search
    while time.time() - start_time < timeout:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = best_points[idx]
        
        # Try a few random moves
        improved = False
        for _ in range(10):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if new_x == old_x and new_y == old_y:
                continue
            if [new_x, new_y] in best_points:
                continue
            
            # Create new candidate
            candidate = [p[:] for p in best_points]
            candidate[idx] = [new_x, new_y]
            
            # Quick validity check
            if not is_valid(candidate):
                continue
            
            score = evaluate(candidate)
            if score < best_score:
                best_score = score
                best_points = candidate
                improved = True
                break
            elif score == best_score and random.random() < 0.1:
                # Accept equal score sometimes to escape local minima
                best_points = candidate
                improved = True
                break
        
        if not improved:
            # Occasionally jump to a new random config if stuck
            if random.random() < 0.05:
                new_points = get_initial_points()
                new_score = evaluate(new_points)
                if new_score < best_score:
                    best_points = new_points
                    best_score = new_score

    return best_points

if __name__ == "__main__":
    construct()
