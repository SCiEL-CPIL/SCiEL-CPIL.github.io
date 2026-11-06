import time
import random
import numpy as np

N = 64
TIME_LIMIT = 300.0


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 10.0

    # Precompute coordinates as flat arrays
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    total = N * N

    # Precompute pairwise squared distances (total x total)
    # dist[i, j] = (xi-xj)^2 + (yi-yj)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = dx * dx + dy * dy

    def is_isosceles_valid(mask):
        """Check if set of points (bool mask) has no isosceles triangle.
        Returns True if valid (no isosceles triangle).
        """
        if mask.sum() < 3:
            return True
        sub_dist = dist[np.ix_(mask, mask)]
        # For each row i, check if any two columns have same distance (apex at i)
        # Also check base cases: dist(i,j) == dist(j,k) or dist(i,k) == dist(j,k)
        # Actually the apex check covers all cases:
        # Triangle (i,j,k) is isosceles if any two of dist(i,j), dist(i,k), dist(j,k) are equal
        # This is equivalent to: for some vertex v, two edges from v have equal length
        # So checking for each row if any two entries (excluding self) are equal suffices
        for i in range(total):
            if not mask[i]:
                continue
            row = sub_dist[i, mask]
            # Exclude self (distance 0)
            if len(row) <= 2:
                continue
            if len(np.unique(row)) != len(row):
                return False
        return True

    def can_add_point(idx, current_mask, current_indices):
        """Check if adding point idx to current set creates isosceles.
        Returns True if can add (no isosceles created).
        """
        K = len(current_indices)
        if K < 2:
            return True

        # Distances from idx to all current points
        new_dists = dist[idx, current_indices]
        # Apex case: two current points at same distance from idx
        if len(np.unique(new_dists)) != K:
            return False
        # Base cases: for each pair (i,j) in current, check if
        # dist(idx,i)==dist(i,j) or dist(idx,j)==dist(i,j)
        sub_dist = dist[np.ix_(current_indices, current_indices)]
        for i in range(K):
            di = new_dists[i]
            for j in range(i + 1, K):
                dj = new_dists[j]
                dij = sub_dist[i, j]
                if di == dij or dj == dij:
                    return False
        return True

    def greedy_random():
        """Greedily add points in random order."""
        current_mask = np.zeros(total, dtype=bool)
        current_indices = []
        remaining = np.arange(total)
        random.shuffle(remaining)
        for idx in remaining:
            if time.time() > deadline:
                break
            if can_add_point(idx, current_mask, current_indices):
                current_mask[idx] = True
                current_indices.append(idx)
        return list(current_indices)

    def local_search(initial_indices, time_budget):
        """Hill-climbing local search with swaps."""
        current = list(initial_indices)
        current_mask = np.zeros(total, dtype=bool)
        for idx in current:
            current_mask[idx] = True
        current_set = set(current)
        best = list(current)
        best_size = len(current)
        end_time = time.time() + time_budget

        improved = True
        while improved and time.time() < end_time:
            improved = False
            n_iter = 0
            while not improved and n_iter < len(current) and time.time() < end_time:
                n_iter += 1
                i = random.randrange(len(current))
                idx_remove = current[i]
                # Find candidates to swap in
                candidates = [j for j in range(total) if not current_mask[j]]
                if not candidates:
                    break
                # Sample some candidates
                sample_size = min(50, len(candidates))
                sample = random.sample(candidates, sample_size)
                for cand in sample:
                    if time.time() > end_time:
                        break
                    # Try swap: remove idx_remove, add cand
                    temp_mask = current_mask.copy()
                    temp_mask[idx_remove] = False
                    temp_indices = [x for x in current if x != idx_remove]
                    if can_add_point(cand, temp_mask, temp_indices):
                        # Accept swap
                        current_mask[idx_remove] = False
                        current_mask[cand] = True
                        current[i] = cand
                        current_set.discard(idx_remove)
                        current_set.add(cand)
                        # Try to add more points after swap
                        remaining_pts = [j for j in range(total) if not current_mask[j]]
                        random.shuffle(remaining_pts)
                        for add in remaining_pts[:20]:
                            if time.time() > end_time:
                                break
                            if can_add_point(add, current_mask, current):
                                current_mask[add] = True
                                current.append(add)
                        improved = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best = list(current)
                        break
        return best

    # Phase 1: Multiple greedy restarts
    best_indices = []
    best_size = 0
    n_restarts = 0
    while time.time() < deadline - 60:
        n_restarts += 1
        result = greedy_random()
        if len(result) > best_size:
            best_size = len(result)
            best_indices = result

    # Phase 2: Local search on best solution
    if best_size > 0 and time.time() < deadline - 20:
        remaining_time = max(10, deadline - 20 - time.time())
        improved_set = local_search(best_indices, remaining_time)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_indices = improved_set

    # Phase 3: Try to add remaining points
    if time.time() < deadline - 5:
        current_mask = np.zeros(total, dtype=bool)
        for idx in best_indices:
            current_mask[idx] = True
        remaining_pts = [j for j in range(total) if not current_mask[j]]
        random.shuffle(remaining_pts)
        for idx in remaining_pts:
            if time.time() > deadline - 5:
                break
            if can_add_point(idx, current_mask, best_indices):
                current_mask[idx] = True
                best_indices.append(idx)

    # Convert indices to (x, y) tuples
    result = [(xs[idx], ys[idx]) for idx in best_indices]
    return result
