"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
FULL_MASK = (1 << N) - 1
DEADLINE = 120.0  # wall-clock seconds; we guard with a safety margin

# EVOLVE-BLOCK-START
def _is_valid_snake(path):
    """Check that path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:  # not a power of two
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def _greedy_extend(path, used, max_steps=None):
    """Greedily extend the path. Returns the extended path."""
    if max_steps is None:
        max_steps = 10000
    for _ in range(max_steps):
        cur = path[-1]
        found = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check induced property: v must differ from all path[:-1] in >= 2 bits
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    return path


def _try_splice(path, used, start, end, new_vertex):
    """Try to insert new_vertex between path[start] and path[end].
    Returns True if successful, modifying path and used in place."""
    # new_vertex must be adjacent to path[start] and path[end]
    if (new_vertex ^ path[start]).bit_count() != 1:
        return False
    if (new_vertex ^ path[end]).bit_count() != 1:
        return False
    # Check induced property with all other vertices
    for i, p in enumerate(path):
        if i == start or i == end:
            continue
        if (new_vertex ^ p).bit_count() == 1:
            return False
    # Also need to check that removing the edge start->end doesn't violate
    # anything else (it shouldn't since we're adding a vertex, not removing)
    path.insert(end, new_vertex)
    used.add(new_vertex)
    return True


def _random_greedy(path_seed, used_seed, time_limit):
    """Run a randomized greedy extension with random bit ordering."""
    path = list(path_seed)
    used = set(used_seed)
    deadline = time.time() + time_limit
    while time.time() < deadline:
        cur = path[-1]
        bits = list(range(N))
        random.shuffle(bits)
        found = False
        for b in bits:
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    return path, used


def _local_search(path, used, time_limit):
    """Try to improve the path by splicing in new vertices at gaps."""
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to find vertices that can be spliced in
        for start in range(len(path) - 1):
            if time.time() >= deadline:
                break
            a, b = path[start], path[start + 1]
            # Candidate vertices adjacent to both a and b (in Q_N, only one such vertex exists: a^b)
            # Actually in a hypercube, two adjacent vertices have exactly one common neighbor: a^b
            candidate = a ^ b
            if candidate in used:
                continue
            # Check if candidate can be spliced between a and b
            if _try_splice(path, used, start, start + 1, candidate):
                improved = True
                # Reset start to 0 since path changed
                start = -1
        # Also try to extend ends
        for _ in range(100):
            if time.time() >= deadline:
                break
            # Try extending the front
            cur = path[0]
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for p in path[1:]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    path.insert(0, v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
            # Try extending the back
            cur = path[-1]
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                    break
                if ok:
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
    return path, used


def construct():
    """Construct the longest induced path (snake) in Q_9."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5 second safety margin

    # Phase 1: Greedy baseline
    path = [0]
    used = {0}
    path = _greedy_extend(path, used)
    best_path = list(path)
    best_used = set(used)
    best_len = len(path)

    # Phase 2: Multiple randomized restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random start vertex
        start_v = random.randint(0, (1 << N) - 1)
        path = [start_v]
        used = {start_v}
        
        # Randomized greedy extension
        remaining = max(0.1, deadline - time.time())
        path, used = _random_greedy(path, used, remaining)
        
        if len(path) > best_len:
            best_path = list(path)
            best_used = set(used)
            best_len = len(path)

    # Phase 3: Local search / splicing on best path
    if time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        best_path, best_used = _local_search(best_path, best_used, remaining)
        best_len = len(best_path)

    # Final validation
    if not _is_valid_snake(best_path):
        # Fallback to simple greedy
        path = [0]
        used = {0}
        path = _greedy_extend(path, used)
        return path

    return best_path
# EVOLVE-BLOCK-END
