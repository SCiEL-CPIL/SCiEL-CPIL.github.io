import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(p1, p2, points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def calculate_score(points):
        """
        Exact branch and bound for the largest subset with no 3 collinear points.
        """
        n = len(points)
        
        # Precompute for each pair (i, j) the set of k such that i, j, k are collinear.
        pair_conflicts = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                pair_conflicts[i][j] = mask
                pair_conflicts[j][i] = mask

        best_score = 0
        full_mask = (1 << n) - 1

        # Precompute bit counts for masks up to 2^20? No, too big. Use bin().count('1')
        # But for speed, we can use a lookup table for bytes?
        # For n=20, bin().count('1') is reasonably fast in Python 3.
        
        def popcount(x):
            return bin(x).count('1')

        def backtrack(available, current_size, selected_mask):
            nonlocal best_score
            
            # Pruning 1: Upper bound on size
            if current_size + popcount(available) <= best_score:
                return
            
            if available == 0:
                if current_size > best_score:
                    best_score = current_size
                return

            # Heuristic: Pick a point from available.
            # Pick the point with the highest "constraint" potential?
            # Let's stick to the first available bit for determinism and speed,
            # but ordering the points initially by degree (number of collinear triples)
            # might help. However, reordering indices complicates pair_conflicts.
            # Let's just pick the lowest index.
            
            i_bit = available & -available
            i = i_bit.bit_length() - 1
            
            # Option 1: Skip i
            backtrack(available ^ i_bit, current_size, selected_mask)
            
            # Option 2: Include i
            new_selected = selected_mask | i_bit
            new_available = available ^ i_bit
            
            # Prune new_available:
            # If we include i, then for any j in new_available, 
            # if there exists k in selected_mask such that i, j, k are collinear,
            # then j cannot be selected.
            
            forbidden = 0
            temp_sel = selected_mask
            while temp_sel:
                k_bit = temp_sel & -temp_sel
                k = k_bit.bit_length() - 1
                forbidden |= pair_conflicts[i][k]
                temp_sel ^= k_bit
            
            new_available &= ~forbidden
            
            backtrack(new_available, current_size + 1, new_selected)

        backtrack(full_mask, 0, 0)
        return best_score

    def generate_random_points():
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    def generate_parabola_points():
        # Parabola y = x^2 has no 4 collinear, but score is 20 (no 3 collinear).
        # We want to minimize score, so this is a bad start, but valid.
        return [[x, x * x] for x in range(-10, 10)]

    best_points = None
    best_score = 21  # Max possible is 20

    # Strategy:
    # 1. Try to find a configuration with low score (many collinear triples) via random search + local search.
    # 2. The score is the size of the largest "no 3 collinear" subset.
    #    Minimizing this means maximizing the number of collinear triples such that they cover the points densely.
    #    Ideally, we want a structure where every point is part of many triples.
    #    A projective plane of order 4 has 21 points, lines of size 5.
    #    If we could embed a subset, the score would be low.
    #    However, we are in Z^2.

    # Let's start with random points and improve via local search.

    # Initial random restarts
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
        points = generate_random_points()
        if not has_four_collinear(points):
            score = calculate_score(points)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]

    if best_points is None:
        best_points = generate_parabola_points()
        best_score = 20

    # Local Search with Simulated Annealing
    current_points = [p[:] for p in best_points]
    current_score = best_score
    T = 10.0
    cooling = 0.995
    
    # Precompute all candidate moves? No, too many.
    # Generate moves on the fly.
    
    while time.time() < deadline - 5:
        if T < 0.1:
            # Restart temperature
            T = 10.0
            # Maybe shuffle points to break symmetry?
            # random.shuffle(current_points)
            # Recalculate score? No, score is invariant to permutation.
            # But we might be stuck. Let's try a few random restarts if stuck.
            pass

        # Pick a random point to move
        i = random.randint(0, 19)
        old_point = current_points[i][:]
        
        # Generate a new position
        # 50% random, 50% on a line defined by two other points
        if random.random() < 0.5:
            new_point = [random.randint(-100, 100), random.randint(-100, 100)]
        else:
            # Pick two other points
            j, k = random.sample([x for x in range(20) if x != i], 2)
            p1 = current_points[j]
            p2 = current_points[k]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Try a few integer points on this line
            # p1 + t * (dx, dy)
            # We need to avoid p1, p2 and other points.
            # And stay within bounds.
            # Let's pick t randomly from a range that keeps us in bounds.
            # t_min, t_max such that |p1 + t*dx| <= 100 and |p1 + t*dy| <= 100
            # For simplicity, try t in [-5, 5] excluding 0, 1
            possible_ts = []
            for t in range(-5, 6):
                if t == 0 or t == 1:
                    continue
                nx = p1[0] + t * dx
                ny = p1[1] + t * dy
                if abs(nx) <= 100 and abs(ny) <= 100:
                    possible_ts.append(t)
            
            if not possible_ts:
                new_point = [random.randint(-100, 100), random.randint(-100, 100)]
            else:
                t = random.choice(possible_ts)
                new_point = [p1[0] + t * dx, p1[1] + t * dy]

        # Check distinctness
        if any(new_point == p for idx, p in enumerate(current_points) if idx != i):
            continue

        test_points = current_points[:]
        test_points[i] = new_point

        if has_four_collinear(test_points):
            # Reject invalid moves
            T *= cooling
            continue

        new_score = calculate_score(test_points)
        
        if new_score < current_score:
            current_points = test_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if random.random() < pow(2.718281828, -delta / T):
                current_points = test_points
                current_score = new_score
        
        T *= cooling

    return best_points
