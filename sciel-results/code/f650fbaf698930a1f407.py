import random
import time
import itertools

def construct():
    start_time = time.time()
    timeout = 140  # Leave 10s margin for evaluator overhead
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def count_collinear_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def has_4_collinear(points):
        n = len(points)
        # Check all pairs, count points on line
        # If any line has >= 4 points, return True
        # Optimization: Use a set of lines? Or just count for each pair.
        # There are 190 pairs. For each pair, check other 18 points.
        # 190 * 18 = 3420 checks.
        for i in range(n):
            for j in range(i+1, n):
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 4:
                    return True
        return False

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_valid_config():
        # Generate 20 random points
        while True:
            points = [random_point() for _ in range(20)]
            # Ensure distinct
            if len(set(map(tuple, points))) < 20:
                continue
            # Ensure no 4 collinear
            if not has_4_collinear(points):
                return points

    best_points = None
    best_triples = -1
    
    # Initial random search
    for _ in range(10):
        if time.time() - start_time > timeout:
            break
        pts = generate_valid_config()
        triples = count_collinear_triples(pts)
        if triples > best_triples:
            best_triples = triples
            best_points = pts

    # Local search
    # We will try to improve the best_points found
    if best_points is None:
        best_points = generate_valid_config()
        best_triples = count_collinear_triples(best_points)
        
    current_points = best_points
    current_triples = best_triples
    
    # Hill Climbing with random restarts
    for restart in range(50):
        if time.time() - start_time > timeout:
            break
            
        # If this is not the first restart, perturb the current best
        if restart > 0:
            # Pick a random point and move it
            idx = random.randint(0, 19)
            current_points[idx] = random_point()
            # Ensure validity
            attempts = 0
            while has_4_collinear(current_points) or len(set(map(tuple, current_points))) < 20:
                current_points[idx] = random_point()
                attempts += 1
                if attempts > 100:
                    break
            
        if time.time() - start_time > timeout:
            break
            
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            # Try moving each point
            for i in range(20):
                if time.time() - start_time > timeout:
                    break
                orig_point = current_points[i]
                # Try a few random moves for this point
                for _ in range(10):
                    new_point = random_point()
                    if new_point == orig_point:
                        continue
                    current_points[i] = new_point
                    # Check validity
                    if len(set(map(tuple, current_points))) < 20:
                        current_points[i] = orig_point
                        continue
                    if has_4_collinear(current_points):
                        current_points[i] = orig_point
                        continue
                    
                    new_triples = count_collinear_triples(current_points)
                    if new_triples > current_triples:
                        current_triples = new_triples
                        improved = True
                        if current_triples > best_triples:
                            best_triples = current_triples
                            best_points = [p[:] for p in current_points]
                        break # Restart the inner loop or continue?
                    else:
                        current_points[i] = orig_point
                        break # Don't waste time on this point if first move fails? 
                             # Or try more? Let's try a few.
                # If we improved, we restart the point loop? Or just continue?
                # Standard hill climbing: if improved, restart loop over points.
                if improved:
                    break

    return best_points
