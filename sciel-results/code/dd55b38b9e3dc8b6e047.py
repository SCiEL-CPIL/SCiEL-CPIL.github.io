"""Snake-in-the-box in Q_9: bitmask DFS backtracking with MRV + Warnsdorff heuristics."""
import time
import random
import sys

N = 9
NUM_VERTICES = 1 << N  # 512

# Precompute neighbor lists
_NEIGHBORS = None
_NEIGHBOR_MASKS = None  # For each vertex, a bitmask of its neighbors

def _init_data():
    global _NEIGHBORS, _NEIGHBOR_MASKS
    if _NEIGHBORS is not None:
        return
    _NEIGHBORS = [[] for _ in range(NUM_VERTICES)]
    _NEIGHBOR_MASKS = [0] * NUM_VERTICES
    for v in range(NUM_VERTICES):
        for b in range(N):
            nb = v ^ (1 << b)
            _NEIGHBORS[v].append(nb)
            _NEIGHBOR_MASKS[v] |= (1 << nb)

def _is_valid_snake(path):
    """Check if path is a valid induced path (snake)."""
    if len(path) <= 1:
        return True
    used = set()
    for i, v in enumerate(path):
        if v in used:
            return False
        used.add(v)
        if i > 0:
            if (v ^ path[i-1]).bit_count() != 1:
                return False
        for j in range(i-2, -1, -1):
            if (v ^ path[j]).bit_count() == 1:
                return False
    return True

def _count_valid_extensions(v, used_mask, path_len):
    """Count how many neighbors of v are valid next steps (not used, not adjacent to non-consecutive path vertices)."""
    count = 0
    for nb in _NEIGHBORS[v]:
        if (used_mask >> nb) & 1:
            continue
        # Check nb is not adjacent to any non-consecutive vertex in the path
        # Non-consecutive means all except the last vertex in path (which is v's predecessor)
        # Actually, nb must not be adjacent to any vertex in the path except possibly the last one
        # The last vertex is the current position; nb is a neighbor of the current position
        # nb must not be adjacent to path[-2] or earlier
        # For efficiency, we check against all used vertices except the immediate predecessor
        # But we don't have the full path here, just the mask. Let's approximate:
        # nb is valid if it's not used and not a neighbor of any used vertex other than current
        # This is conservative but fast enough
        count += 1
    return count

def _dfs_backtrack(deadline, start_vertex=0, max_depth=None):
    """
    DFS backtracking search for longest snake.
    Uses Warnsdorff's rule: at each step, prefer the neighbor with fewest onward options.
    """
    _init_data()
    best_path = [start_vertex]
    best_len = 1
    used_mask = 1 << start_vertex
    path = [start_vertex]
    nodes_explored = 0
    timeout = [False]

    def _onward_options(v, used_mask, path):
        """Count valid onward options from v, given current used_mask and path."""
        count = 0
        for nb in _NEIGHBORS[v]:
            if (used_mask >> nb) & 1:
                continue
            # Check nb not adjacent to any non-consecutive vertex
            # Non-consecutive to nb means: all path vertices except path[-1] (the predecessor of v)
            # Actually nb is being considered as next after v=path[-1]
            # nb must not be adjacent to path[-2], path[-3], ... path[0]
            # For speed, check against all used vertices except v itself
            valid = True
            for i in range(len(path) - 2, -1, -1):
                if ((1 << path[i]) & _NEIGHBOR_MASKS[nb]):
                    valid = False
                    break
            if valid:
                count += 1
        return count

    def _search(depth):
        nonlocal best_path, best_len, nodes_explored, timeout
        if timeout[0]:
            return
        if time.time() > deadline:
            timeout[0] = True
            return
        nodes_explored += 1

        cur = path[-1]
        # Gather valid candidates
        candidates = []
        for nb in _NEIGHBORS[cur]:
            if (used_mask >> nb) & 1:
                continue
            # Check nb not adjacent to any non-consecutive vertex
            valid = True
            for i in range(len(path) - 2, -1, -1):
                if ((1 << path[i]) & _NEIGHBOR_MASKS[nb]):
                    valid = False
                    break
            if valid:
                candidates.append(nb)

        if not candidates:
            # Dead end - update best if longer
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            return

        # Warnsdorff: sort by number of onward options (ascending)
        scored = []
        for nb in candidates:
            # Count onward options from nb
            onward = 0
            for nb2 in _NEIGHBORS[nb]:
                if nb2 == cur:
                    continue
                if (used_mask >> nb2) & 1:
                    continue
                # Check nb2 not adjacent to non-consecutive vertices
                ok = True
                for i in range(len(path) - 2, -1, -1):
                    if ((1 << path[i]) & _NEIGHBOR_MASKS[nb2]):
                        ok = False
                        break
                if ok:
                    onward += 1
            scored.append((onward, nb))

        scored.sort()

        # Try candidates in order of fewest onward options first
        for _, nb in scored:
            if timeout[0]:
                return
            path.append(nb)
            new_mask = used_mask | (1 << nb)
            _search(depth + 1)
            path.pop()
            if timeout[0]:
                return
            # If we found a better solution, we could prune, but for now continue
            # Early termination: if best_len >= NUM_VERTICES, we're done
            if best_len >= NUM_VERTICES:
                timeout[0] = True
                return

    _search(0)
    return best_path

def _dfs_with_random_restart(deadline, num_restarts=None):
    """
    Multiple DFS restarts from different starting vertices.
    Returns the best path found.
    """
    _init_data()
    start_time = time.time()
    best_path = None
    best_len = 0

    # Try starting from vertex 0 first (deterministic)
    p = _dfs_backtrack(deadline, start_vertex=0)
    if p and len(p) > best_len:
        best_path = p
        best_len = len(p)

    # If time remains, try a few random restarts
    if time.time() < deadline - 2.0:
        random.seed(42)
        restarts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
        for sv in restarts:
            if time.time() > deadline - 2.0:
                break
            p = _dfs_backtrack(deadline, start_vertex=sv)
            if p and len(p) > best_len:
                best_path = p
                best_len = len(p)

    return best_path if best_path else [0]

def _extend_path(path, deadline):
    """Try to extend a given path using greedy Warnsdorff."""
    if not path:
        return path
    used_mask = 0
    for v in path:
        used_mask |= (1 << v)

    current = list(path)
    while True:
        if time.time() > deadline:
            break
        cur = current[-1]
        candidates = []
        for nb in _NEIGHBORS[cur]:
            if (used_mask >> nb) & 1:
                continue
            # Check nb not adjacent to any non-consecutive vertex
            valid = True
            for i in range(len(current) - 2, -1, -1):
                if ((1 << current[i]) & _NEIGHBOR_MASKS[nb]):
                    valid = False
                    break
            if valid:
                # Count onward options
                onward = 0
                for nb2 in _NEIGHBORS[nb]:
                    if nb2 == cur:
                        continue
                    if (used_mask >> nb2) & 1:
                        continue
                    ok = True
                    for i in range(len(current) - 2, -1, -1):
                        if ((1 << current[i]) & _NEIGHBOR_MASKS[nb2]):
                            ok = False
                            break
                    if ok:
                        onward += 1
                candidates.append((onward, nb))

        if not candidates:
            break
        candidates.sort()
        best_nb = candidates[0][1]
        current.append(best_nb)
        used_mask |= (1 << best_nb)

    return current

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    _init_data()

    # Primary: DFS backtracking with Warnsdorff
    best_path = _dfs_with_random_restart(deadline)
    best_len = len(best_path)

    # Try to extend with greedy if time remains
    if time.time() < deadline - 5.0:
        extended = _extend_path(best_path, deadline)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)

    # Validate
    if not _is_valid_snake(best_path):
        # Fallback: simple greedy
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best_path = path

    return best_path
