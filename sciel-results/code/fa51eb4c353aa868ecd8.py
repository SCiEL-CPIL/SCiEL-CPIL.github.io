import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    best_points = None
    best_score = 21  # Max possible is 20
    
    # Helper to check if 4 points are collinear
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        # Check if any other point is on this line
                        # Line defined by points[i], points[j]
                        # Cross product (p - p_i) x (p_j - p_i) == 0
                        dx = points[j][0] - points[i][0]
                        dy = points[j][1] - points[i][1]
                        for l in range(k+1, n):
                            if (points[l][0] - points[i][0]) * dy - (points[l][1] - points[i][1]) * dx == 0:
                                return True
        return False

    def get_collinear_triples(points):
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def max_no_three_collinear(points):
        n = len(points)
        triples = get_collinear_triples(points)
        if not triples:
            return n
        
        # Branch and Bound
        # State: available points (bitmask), current size
        # We want to maximize size of subset with no collinear triple.
        
        # Precompute for each point, which triples it is in?
        # Or just use the list of triples.
        
        best = 0
        def dfs(available_mask, current_size):
            nonlocal best
            if current_size + bin(available_mask).count('1') <= best:
                return
            
            if available_mask == 0:
                best = max(best, current_size)
                return
            
            # Find a collinear triple in the available set
            triple_found = None
            for (i, j, k) in triples:
                if (available_mask >> i) & 1 and (available_mask >> j) & 1 and (available_mask >> k) & 1:
                    triple_found = (i, j, k)
                    break
            
            if triple_found is None:
                # No more collinear triples in available set
                best = max(best, current_size + bin(available_mask).count('1'))
                return
            
            i, j, k = triple_found
            # Branch: exclude i, or exclude j, or exclude k
            # We can also include one and exclude others?
            # Standard B&B for independent set in hypergraph:
            # Pick a hyperedge H. Any independent set can contain at most 2 vertices of H.
            # Actually, it can contain 0, 1, or 2.
            # To prune, we can say:
            # Option 1: Exclude i.
            # Option 2: Exclude j.
            # Option 3: Exclude k.
            # This covers all cases where i, j, k are not all included.
            
            dfs(available_mask & ~(1 << i), current_size)
            dfs(available_mask & ~(1 << j), current_size)
            dfs(available_mask & ~(1 << k), current_size)
            
        dfs((1 << n) - 1, 0)
        return best

    def score(points):
        if has_four_collinear(points):
            return 100 # Invalid
        return max_no_three_collinear(points)

    # Initial random restarts
    # Try to find a configuration with score < 20
    
    # Heuristic: Start with a grid-like structure and perturb
    # Or just random
    
    best_found = None
    
    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Generate random points
        points = []
        used = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                points.append([x, y])
        
        # Local search
        current_points = [p[:] for p in points]
        current_score = score(current_points)
        
        if current_score < 20:
            best_found = current_points
            best_score = current_score
            if best_score <= 10: # Good enough?
                break
                
        # Local improvement
        for step in range(100):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 19)
            old_x, old_y = current_points[idx]
            
            # Try a random neighbor
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            
            if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                continue
            
            # Check if point is already used
            if (new_x, new_y) in [(p[0], p[1]) for p in current_points if current_points.index(p) != idx]:
                continue
                
            current_points[idx] = [new_x, new_y]
            
            new_score = score(current_points)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_found = [p[:] for p in current_points]
                    if best_score <= 10:
                        break
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
                
        if time.time() > deadline:
            break

    if best_found:
        return best_found
    
    # Fallback: Return a valid configuration if we found one, else a parabola
    # We should ensure we return a valid configuration
    # Let's just return the parabola if we didn't find a better one, 
    # but we know the parabola has score 20.
    # We want to minimize score.
    # If best_found is None, it means we didn't find any with score < 20?
    # Or we didn't run enough?
    # Let's ensure we at least return a valid one.
    
    # The parabola is valid.
    return [[x, x * x] for x in range(-10, 10)]
