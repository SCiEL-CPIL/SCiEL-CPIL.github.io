import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def get_triples(points):
        triples = []
        for i, j, k in combinations(range(len(points)), 3):
            if is_collinear(points[i], points[j], points[k]):
                triples.append((i, j, k))
        return triples

    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if there are at least 2 other points on the line defined by i, j
                count = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
                        if count >= 2:
                            return True
        return False

    # 1. Generate a valid initial configuration (no 4 collinear)
    # We use a "parabola-like" distribution to minimize collinearities initially,
    # then verify and repair if necessary.
    def generate_initial():
        while True:
            points = []
            # Try to place points with distinct x-coordinates to avoid vertical lines
            # and use a quadratic function y = x^2 + noise to minimize collinearities
            x_coords = random.sample(range(-100, 101), 20)
            for x in x_coords:
                # y = x^2 mod 201 mapped to [-100, 100]
                # Actually, let's just pick random y first, then check.
                # Better: pick random points and check.
                pass
            
            # Let's stick to the robust random generation with checking
            points = []
            attempts = 0
            while len(points) < 20:
                p = [random.randint(-100, 100), random.randint(-100, 100)]
                if p in points:
                    continue
                valid = True
                # Check if adding p creates a 4-collinear set
                # This is equivalent to: is there a line containing p and 3 existing points?
                # Or a line containing p and 2 existing points where a 3rd existing point is also there?
                # No, simply: for any 3 existing points, are they collinear with p?
                # If 3 existing points are collinear, they already violate the condition (if we ensured no 4 collinear previously, 
                # they can't be 3 collinear? No, 3 collinear is allowed, 4 is not).
                # Wait, the constraint is "no four collinear". 
                # So we can have 3 collinear.
                # If we add p, we must ensure we don't create a set of 4.
                # This happens if p lies on a line that already contains 3 points.
                # Or if p lies on a line that contains 2 points, and we add another point later? 
                # No, we check incrementally.
                # If current set has no 4 collinear, adding p creates 4 collinear iff:
                # There exists a line L such that L contains p and at least 3 points from the current set.
                
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        # Count points on line ij
                        count = 2
                        for k in range(len(points)):
                            if k != i and k != j:
                                if is_collinear(points[i], points[j], points[k]):
                                    count += 1
                        # If count >= 3, then line ij has >= 3 points.
                        # If p is on this line, we get >= 4.
                        if count >= 3:
                            if is_collinear(points[i], points[j], p):
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    points.append(p)
                attempts += 1
                if attempts > 50000:
                    return None
            if not has_four_collinear(points):
                return points
    points = generate_initial()
    if points is None:
        # Fallback to simple random if generation fails (unlikely)
        points = []
        while len(points) < 20:
            p = [random.randint(-100, 100), random.randint(-100, 100)]
            if p not in points:
                points.append(p)
        # This fallback might have 4 collinear, but we'll try to fix it. 
        # However, the task requires NO FOUR COLLINEAR as a hard constraint.
        # So we must ensure validity.
        # Let's restart if fallback is invalid.
        while has_four_collinear(points):
             points = generate_initial()
             if points is None:
                 # Extreme fallback: fixed points? No, must be distinct.
                 # Just keep trying
                 points = []
                 for i in range(20):
                     points.append([i*10-100, (i*i)%200-100])
                 # This might have 4 collinear. 
                 # Given the time, the first generator is robust.

    current_triples = get_triples(points)
    best_points = points
    best_triple_count = len(current_triples)

    # 2. Greedy Hitting Set / Set Cover
    # Objective: Minimize number of collinear triples.
    # Move: Change one point.
    # Gain: Number of existing triples broken.
    # Cost: Number of new triples created.
    # We accept a move if Gain > Cost (or Gain >= Cost to allow plateaus, but strictly decreasing triples is safer).
    # Actually, we want to reduce the total count.
    # Delta = New_Count - Old_Count.
    # We want Delta < 0.
    
    while time.time() < deadline:
        made_progress = False
        # Find the best move
        best_move = None
        best_delta = 0 # We want negative delta
        
        for idx in range(20):
            old_p = points[idx]
            # Try a few random new positions? Or just 1?
            # To be deterministic and fast, let's try 5 random candidates per index.
            candidates = []
            for _ in range(5):
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                if new_p != old_p and new_p not in points:
                    candidates.append(new_p)
            
            for new_p in candidates:
                # Check validity: No 4 collinear
                test_points = points.copy()
                test_points[idx] = new_p
                if has_four_collinear(test_points):
                    continue
                
                # Calculate change in triple count
                # Instead of full recount, we can compute delta.
                # But full recount is O(N^3) = 1140 checks. 
                # 20 indices * 5 candidates * 1140 = 114,000 checks per iteration.
                # This is very fast.
                
                test_triples = get_triples(test_points)
                new_count = len(test_triples)
                delta = new_count - len(current_triples)
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = (idx, new_p, new_count)
        
        if best_move is not None:
            idx, new_p, new_count = best_move
            points[idx] = new_p
            current_triples = get_triples(points)
            if new_count < best_triple_count:
                best_points = points.copy()
                best_triple_count = new_count
            made_progress = True
        else:
            # No improving move found. 
            # Try a random restart or perturbation?
            # Since we are minimizing, if we are stuck, we might be in a local minimum.
            # Let's try a random "kick": accept a move with delta <= 2?
            # Or just restart the search from a new random point in the neighborhood?
            # Given the time budget, let's try a few random moves that don't increase triples too much.
            kicked = False
            for _ in range(10):
                idx = random.randint(0, 19)
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                if new_p == points[idx] or new_p in points:
                    continue
                test_points = points.copy()
                test_points[idx] = new_p
                if has_four_collinear(test_points):
                    continue
                test_triples = get_triples(test_points)
                new_count = len(test_triples)
                delta = new_count - len(current_triples)
                if delta <= 1: # Allow small increases to escape
                    points[idx] = new_p
                    current_triples = test_triples
                    if new_count < best_triple_count:
                        best_points = points.copy()
                        best_triple_count = new_count
                    kicked = True
                    break
            
            if not kicked:
                # If we can't even kick, break? Or try harder?
                # Let's just break to avoid wasting time if stuck in a deep hole.
                # Or restart from scratch?
                # Let's try one full restart if we have time.
                if time.time() < deadline - 20:
                    new_points = generate_initial()
                    if new_points:
                        new_triples = get_triples(new_points)
                        if len(new_triples) < best_triple_count:
                            points = new_points
                            current_triples = new_triples
                            best_points = points.copy()
                            best_triple_count = len(current_triples)
                        else:
                            # Keep current points, but maybe shuffle?
                            # Just break to prevent infinite loop of failed kicks
                            pass
                else:
                    break

    return best_points
