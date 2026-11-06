"""No-isosceles set: maximize points in [0,N)^2 with no isosceles triangle."""
import random
import time
import math

N = 64
TIME_LIMIT = 280.0


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 5.0

    best_set = []
    best_size = 0

    all_points = [(x, y) for x in range(N) for y in range(N)]
    total_points = N * N

    # Precompute all pairs and their squared distances
    # For speed, we'll work with indices
    # But for simplicity, let's use a smarter greedy with O(K) check using distance groups

    def can_add_fast(point, points_list):
        """Check if adding point to points_list creates an isosceles triangle.
        Returns True if the point can be added (no isosceles created).
        """
        x, y = point
        K = len(points_list)
        if K < 2:
            return True

        # Compute distances from point to all existing points
        dists = []
        for p in points_list:
            dx = x - p[0]
            dy = y - p[1]
            dists.append(dx * dx + dy * dy)

        # Check apex case: two existing points at same distance from new point
        if len(set(dists)) != K:
            return False

        # Check base case: for each pair (i,j), check if dist(new,pi) == dist(pi,pj) or dist(new,pj) == dist(pi,pj)
        # This is O(K^2) but with simple comparisons
        for i in range(K):
            di = dists[i]
            pi = points_list[i]
            for j in range(i + 1, K):
                dj = dists[j]
                pj = points_list[j]
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dij = dx * dx + dy * dy
                if di == dij or dj == dij:
                    return False
        return True

    def greedy_from_order(points_list, order):
        """Greedily add points from the given order to points_list."""
        result = list(points_list)
        for p in order:
            if p in result:
                continue
            if can_add_fast(p, result):
                result.append(p)
        return result

    # Multiple random greedy restarts
    for attempt in range(200):
        if time.time() > deadline:
            break

        order = all_points[:]
        random.shuffle(order)
        current = greedy_from_order([], order)

        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

    # Local search: try to swap points to escape local optima
    if best_size > 0 and time.time() < deadline:
        # Work with the best set
        current_set = list(best_set)
        current_set_set = set(current_set)

        # Try to add any remaining points
        remaining = [p for p in all_points if p not in current_set_set]
        random.shuffle(remaining)
        for p in remaining:
            if time.time() > deadline:
                break
            if can_add_fast(p, current_set):
                current_set.append(p)
                current_set_set.add(p)

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

        # Hill climbing: try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set_list = list(current_set)
            current_set_set = set(current_set_list)

            for i in range(len(current_set_list)):
                if time.time() > deadline:
                    break
                p_remove = current_set_list[i]
                # Try to find a point to replace p_remove with
                # Pick a few random candidates from remaining
                remaining = [p for p in all_points if p not in current_set_set]
                random.shuffle(remaining)
                for candidate in remaining[:20]:  # Try up to 20 random candidates
                    if time.time() > deadline:
                        break
                    # Temporarily remove p_remove and try to add candidate
                    temp_set = [p for j, p in enumerate(current_set_list) if j != i]
                    if can_add_fast(candidate, temp_set):
                        # Accept the swap
                        current_set_list[i] = candidate
                        current_set_set.remove(p_remove)
                        current_set_set.add(candidate)
                        improved = True
                        break

            if len(current_set_list) > best_size:
                best_size = len(current_set_list)
                best_set = list(current_set_list)
                current_set = list(current_set_list)

    return best_set
