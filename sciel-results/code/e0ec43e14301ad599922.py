"""Snake-in-the-box in Q_9: systematic gap-insertion + exhaustive polish."""
import time
import random

N = 9
DEADLINE = 150.0  # wall-clock limit; keep safety margin

# EVOLVE-BLOCK-START

def _can_insert(path, idx, v):
    """Check if vertex v can be inserted at position idx in path (between idx-1 and idx).
    path is a list of ints. v must not be in path.
    For insertion at position idx (0 <= idx <= len(path)):
    - if idx > 0: v must differ from path[idx-1] in exactly 1 bit (edge)
    - if idx < len(path): v must differ from path[idx] in exactly 1 bit (edge)
    - v must differ from all other vertices in path in >= 2 bits
    """
    n = len(path)
    # Check v not already in path
    # We'll check bit-distance to all other vertices
    # Edge constraints
    if idx > 0:
        if (v ^ path[idx - 1]).bit_count() != 1:
            return False
    if idx < n:
        if (v ^ path[idx]).bit_count() != 1:
            return False
    # Non-adjacent: v must differ from all path[i] for i != idx-1, idx in >= 2 bits
    for i, p in enumerate(path):
        if i == idx or i == idx - 1:
            continue
        if (v ^ p).bit_count() < 2:
            return False
    return True


def _try_insert_vertex(path, v):
    """Try to insert v into path at any valid gap. Returns (new_path, idx) or (None, None)."""
    n = len(path)
    for idx in range(n + 1):
        if _can_insert(path, idx, v):
            new_path = path[:idx] + [v] + path[idx:]
            return new_path, idx
    return None, None


def _greedy_construct():
    """Build a greedy snake by always extending with the lowest valid bit flip."""
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check non-adjacent constraint
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _polish_path(path, deadline):
    """Exhaustive polish: repeatedly try to insert any unused vertex into any gap.
    Repeat until no more insertions are possible or deadline is reached."""
    used = set(path)
    all_vertices = set(range(1 << N))
    unused = all_vertices - used

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try each unused vertex in each gap
        for v in list(unused):
            if time.time() > deadline - 2.0:
                break
            new_path, idx = _try_insert_vertex(path, v)
            if new_path is not None:
                path = new_path
                used.add(v)
                unused.remove(v)
                improved = True
                break  # restart scan with updated path
    return path


def _random_gap_insertion(path, num_iters, deadline):
    """Do random gap insertions: pick random unused vertex, try all gaps."""
    used = set(path)
    all_vertices = list(range(1 << N))
    unused = [v for v in all_vertices if v not in used]
    random.shuffle(unused)

    for _ in range(num_iters):
        if time.time() > deadline - 2.0:
            break
        if not unused:
            break
        v = unused.pop(random.randint(0, len(unused) - 1))
        new_path, idx = _try_insert_vertex(path, v)
        if new_path is not None:
            path = new_path
            used.add(v)
    return path


def _systematic_sweep(path, deadline):
    """Sweep through all gaps for all unused vertices systematically."""
    used = set(path)
    all_vertices = list(range(1 << N))
    unused = [v for v in all_vertices if v not in used]

    for v in unused:
        if time.time() > deadline - 2.0:
            break
        new_path, idx = _try_insert_vertex(path, v)
        if new_path is not None:
            path = new_path
            used.add(v)
    return path


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin

    # Phase 1: Build initial greedy snake
    path = _greedy_construct()

    # Phase 2: Systematic sweep to insert unused vertices
    if time.time() < deadline - 3.0:
        path = _systematic_sweep(path, deadline)

    # Phase 3: Random gap insertions for diversity
    if time.time() < deadline - 3.0:
        path = _random_gap_insertion(path, 100, deadline)

    # Phase 4: Another systematic sweep
    if time.time() < deadline - 3.0:
        path = _systematic_sweep(path, deadline)

    # Phase 5: Exhaustive polish (repeat until no improvement)
    path = _polish_path(path, deadline)

    # Phase 6: Final check - try one more round of systematic + polish
    if time.time() < deadline - 2.0:
        path = _systematic_sweep(path, deadline)
        path = _polish_path(path, deadline)

    return path

# EVOLVE-BLOCK-END
