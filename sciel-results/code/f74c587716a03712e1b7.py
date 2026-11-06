"""Snake-in-the-box in the 9-dimensional hypercube - Backtracking with bitmask state."""
N = 9
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute neighbor masks
    # neighbor_mask[v] is a bitmask of all vertices at Hamming distance 1 from v
    neighbor_mask = [0] * (1 << N)
    for v in range(1 << N):
        mask = 0
        for b in range(N):
            nv = v ^ (1 << b)
            mask |= (1 << nv)
        neighbor_mask[v] = mask

    bit_masks = [1 << b for b in range(N)]

    def _backtrack_search(start_vertex, time_budget):
        """Backtracking search with Warnsdorff heuristic using bitmask state."""
        best_path = [start_vertex]
        best_len = 1
        local_deadline = time.time() + time_budget

        # State: path, forbidden_mask
        # forbidden_mask bit i set = vertex i cannot be next
        # Initially: forbidden = (1 << start_vertex)
        # When we add v to path:
        #   new_forbidden = forbidden | (1 << v) | neighbor_mask[cur]
        #   where cur is the previous last vertex (which becomes non-consecutive)

        def _search(path, forbidden_mask, depth):
            nonlocal best_path, best_len

            if time.time() > local_deadline:
                return

            if depth > best_len:
                best_len = depth
                best_path = path[:]

            cur = path[-1]

            # Find candidate next vertices
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden_mask & (1 << v):
                    continue
                # v is a valid candidate
                # Compute onward moves for Warnsdorff
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, b, v))

            if not candidates:
                return

            # Sort by onward moves (Warnsdorff: fewest onward first)
            candidates.sort(key=lambda x: x[0])

            # Try each candidate
            for onward, b, v in candidates:
                if time.time() > local_deadline:
                    return
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                path.append(v)
                _search(path, new_forbidden, depth + 1)
                path.pop()

        _search([start_vertex], (1 << start_vertex), 1)
        return best_path

    def _greedy_extend(path, time_budget):
        """Greedily extend a path using Warnsdorff with bitmask state."""
        if not path:
            return path

        # Reconstruct forbidden mask from path
        forbidden_mask = 0
        for v in path:
            forbidden_mask |= (1 << v)
        for v in path[:-1]:
            forbidden_mask |= neighbor_mask[v]

        cur = path[-1]
        local_deadline = time.time() + time_budget

        while time.time() < local_deadline:
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden_mask & (1 << v):
                    continue
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, b, v))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            _, b, v = candidates[0]

            path.append(v)
            forbidden_mask |= (1 << v) | neighbor_mask[cur]
            cur = v

        return path

    # Phase 1: Backtracking search from multiple starts
    rng = random.Random(12345)

    # Start with deterministic high-value starts
    starts = [0, 1, (1 << N) - 1, 0x155, 0x1AA, 0x0FF, 0x1F0, 0x123, 0x1AB]
    # Add random starts
    for _ in range(10):
        starts.append(rng.randint(0, (1 << N) - 1))

    best_path = None
    best_len = 0

    # Allocate time for backtracking
    # We have ~140s total. Let's use 100s for backtracking, 40s for refinement.
    bt_deadline = start_time + 100.0

    for sv in starts:
        if time.time() > bt_deadline:
            break
        # Give each backtracking search a time budget
        # We have ~29 starts. 100s / 29 ~ 3.4s each.
        # But we should prioritize. Let's just run until deadline.
        path = _backtrack_search(sv, 0.0) # Pass 0 to let it run until local_deadline? No, _backtrack_search takes time_budget.
        # Wait, _backtrack_search uses local_deadline = time.time() + time_budget.
        # If I pass 0, it checks time.time() > time.time() which is False initially, but might be True later?
        # No, I want it to run until the global deadline.
        # Let's modify _backtrack_search to accept a deadline instead of budget, or calculate remaining time.
        remaining = bt_deadline - time.time()
        if remaining <= 0:
            break
        path = _backtrack_search(sv, remaining)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # Phase 2: Refine best_path using greedy extension
    if best_path is None:
        # Fallback to simple greedy if backtracking failed completely
        best_path = _greedy_extend([0], 10.0)

    # Try to extend best_path further
    # Note: best_path from backtracking is a dead end (no valid moves).
    # Greedy extension won't help unless we do some local search / restart.
    # But the inspiration used greedy extension on the result.
    # Actually, if backtracking found a dead end, greedy can't extend it.
    # However, maybe we can try to extend from a slightly shorter prefix of best_path?
    # Or just run greedy from scratch with more time?

    # Let's try a few greedy runs from random starts with the remaining time
    greedy_deadline = deadline - 5.0
    while time.time() < greedy_deadline:
        sv = rng.randint(0, (1 << N) - 1)
        path = _greedy_extend([sv], 1.0) # 1 second budget per greedy run
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    return best_path

# EVOLVE-BLOCK-END
