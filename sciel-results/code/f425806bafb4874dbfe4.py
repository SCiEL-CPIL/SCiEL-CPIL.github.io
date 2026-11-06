import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    def calculate_score(pts):
        """
        Exact branch and bound for largest subset with no three collinear.
        Uses pair-bitmask for O(1) triple validity check.
        """
        n = len(pts)
        if n < 3:
            return n

        # Precompute for each point i, a bitmask of points j such that
        # i and j are part of some collinear triple with another point.
        # Actually, we need: for each point i, the set of pairs (j,k) with j<k
        # such that i,j,k are collinear. But for the B&B, we just need to check
        # if adding i creates a triple with any two already-selected points.
        #
        # Better approach: precompute for each point i, a bitmask conflict_mask[i]
        # where conflict_mask[i] has bit j set if i and j are on a line with
        # at least one other point. Wait, that's not quite right either.
        #
        # The correct approach: when we have selected set S and want to add point i,
        # we need to check if there exist j, k in S such that i, j, k are collinear.
        # This is equivalent to: for each j in S, check if there exists k in S
        # such that i, j, k are collinear.
        #
        # Precompute: for each pair (i, j), a bitmask of all points k such that
        # i, j, k are collinear. Then when adding i with current set S, for each
        # j in S, check if (collinear_with[i][j] & S) has any bits set other than i and j.
        #
        # Actually simpler: precompute triple_mask[i] = bitmask of all points that
        # form a collinear triple with i and at least one other point. No, that's
        # not sufficient.
        #
        # Let's precompute: for each i, and for each j != i, the set of k such that
        # i, j, k are collinear. Store as collinear_pairs[i][j] = bitmask of k.
        # Then to check if adding i to set S is valid: for each j in S, check if
        # collinear_pairs[i][j] & S has any bits. If yes, invalid.

        # Precompute collinear_pairs
        collinear_pairs = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        collinear_pairs[i][j] |= (1 << k)
                        collinear_pairs[j][i] |= (1 << k)

        best_score = 0

        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return

            # Pruning
            if current_size + (n - step) <= best_score:
                return

            idx = step

            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size)

            # Option 2: Include idx
            # Check if adding idx creates a triple with any two points in current_mask
            valid = True
            idx_bit = 1 << idx
            for j in range(n):
                if current_mask & (1 << j):
                    if collinear_pairs[idx][j] & current_mask:
                        valid = False
                        break

            if valid:
                bbb(step + 1, current_mask | idx_bit, current_size + 1)

        bbb(0, 0, 0)
        return best_score

    def generate_initial_points():
        """
        Generate initial points using a structured approach.
        Start with points on a parabola y = x^2 (which has no 3 collinear for distinct x),
        then perturb to get 20 points with no 4 collinear.
        """
        random.seed(42)
        # Use points near a parabola to avoid collinearities
        base_x = list(range(-10, 10))
        pts = []
        for x in base_x:
            y = x * x // 4  # Parabolic curve
            # Add small perturbation
            y += random.randint(-2, 2)
            pts.append((x, y))

        # Ensure no duplicates and within bounds
        seen = set()
        unique_pts = []
        for p in pts:
            if p not in seen and -100 <= p[0] <= 100 and -100 <= p[1] <= 100:
                seen.add(p)
                unique_pts.append(p)

        # Add more points if needed
        while len(unique_pts) < 20:
            p = (random.randint(-50, 50), random.randint(-50, 50))
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)

        # Fix any 4-collinear issues
        for _ in range(200):
            if not has_4_collinear(unique_pts):
                break
            # Find a point that's part of a 4-collinear set and move it
            # Simple approach: randomly move a point
            idx = random.randint(0, 19)
            old_p = unique_pts[idx]
            new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
            if new_p not in unique_pts and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                unique_pts[idx] = new_p

        return unique_pts

    def local_search(pts, max_iterations=3000):
        """
        Simulated annealing local search to minimize the score.
        """
        current_pts = [tuple(p) for p in pts]
        if has_4_collinear(current_pts):
            # Repair
            for _ in range(200):
                if not has_4_collinear(current_pts):
                    break
                idx = random.randint(0, 19)
                old_p = current_pts[idx]
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                if new_p not in current_pts and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    current_pts[idx] = new_p
            if has_4_collinear(current_pts):
                return None, float('inf')

        current_score = calculate_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score

        # Simulated annealing parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.995

        for i in range(max_iterations):
            if elapsed() > deadline - 10:
                break

            idx = random.randint(0, 19)
            old_p = current_pts[idx]

            # Try multiple candidate moves
            found_move = False
            for _ in range(15):
                # Choose move type: local (small) or global (larger)
                if random.random() < 0.8:
                    # Local move
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                else:
                    # Larger move
                    dx = random.randint(-10, 10)
                    dy = random.randint(-10, 10)

                new_p = (old_p[0] + dx, old_p[1] + dy)
                if new_p in current_pts:
                    continue
                if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                    continue

                new_pts = current_pts[:]
                new_pts[idx] = new_p

                if has_4_collinear(new_pts):
                    continue

                new_score = calculate_score(new_pts)

                # Acceptance criterion
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_pts = new_pts
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_pts = new_pts[:]
                    found_move = True
                    break

            if not found_move:
                # Try a random restart of one point
                idx = random.randint(0, 19)
                current_pts[idx] = (random.randint(-100, 100), random.randint(-100, 100))
                if has_4_collinear(current_pts):
                    current_pts[idx] = old_p
                else:
                    current_score = calculate_score(current_pts)
                    if current_score < best_score:
                        best_score = current_score
                        best_pts = current_pts[:]

            T = max(T * cooling, T_min)

        return best_pts, best_score

    best_points = None
    best_val = float('inf')

    # Generate multiple initial candidates
    candidates = []
    for _ in range(8):
        if elapsed() > deadline - 40:
            break
        cand = generate_initial_points()
        if not has_4_collinear(cand):
            candidates.append(cand)

    # Also add some random ones
    for _ in range(4):
        if elapsed() > deadline - 40:
            break
        pts = []
        while len(pts) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in pts:
                pts.append(p)
        if not has_4_collinear(pts):
            candidates.append(pts)

    random.shuffle(candidates)

    for cand in candidates:
        if elapsed() > deadline - 20:
            break

        pts, score = local_search(cand, max_iterations=1500)

        if pts is not None and score < best_val:
            best_val = score
            best_points = pts

    if best_points is None:
        # Fallback
        best_points = generate_initial_points()
        if has_4_collinear(best_points):
            for _ in range(200):
                if not has_4_collinear(best_points):
                    break
                idx = random.randint(0, 19)
                old_p = best_points[idx]
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                if new_p not in best_points and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    best_points[idx] = new_p

    return [[x, y] for x, y in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)
