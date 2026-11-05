"""Snake-in-the-box Q_9: structural change to a forbidden-set representation
plus a different search family (simulated-annealing-style local search over
partial paths with restarts and a final beam polish)."""
import time
import random
import sys

N = 9
V = 1 << N
TIME_LIMIT = 145  # safety margin below 150s

# Precompute adjacency: for each vertex, list of neighbors (bit flips)
_ADJ = [[] for _ in range(V)]
for _v in range(V):
    for _b in range(N):
        _ADJ[_v].append(_v ^ (1 << _b))

# Precompute popcount for XOR of two vertices? Too big (V^2). Use bit_count.

def _popcount(x):
    return x.bit_count()

def _verify_snake(path):
    """Verify that the path is a valid induced path (snake)."""
    if not path:
        return False
    n = len(path)
    if len(set(path)) != n:
        return False
    for v in path:
        if v < 0 or v >= V:
            return False
    for i in range(n - 1):
        if _popcount(path[i] ^ path[i + 1]) != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if _popcount(path[i] ^ path[j]) == 1:
                return False
    return True


def _greedy_build(forbidden, start_vertex, bit_order):
    """
    Build a snake greedily from start_vertex, maintaining a forbidden set.
    forbidden[v] is True if v cannot be added to the path (it is adjacent to
    some non-end vertex in the path, or already used, or is the start).
    Returns the path.
    """
    path = [start_vertex]
    used = set(path)
    # Initialize forbidden: all neighbors of start are forbidden
    for nb in _ADJ[start_vertex]:
        forbidden[nb] = True
    forbidden[start_vertex] = True  # mark used

    cur = start_vertex
    while True:
        # Find a valid neighbor to extend to
        nxt = None
        for b in bit_order:
            v = cur ^ (1 << b)
            if not forbidden[v]:
                nxt = v
                break
        if nxt is None:
            break
        # Extend
        path.append(nxt)
        used.add(nxt)
        # Update forbidden: all neighbors of cur (except nxt) become forbidden
        # because they would be adjacent to a non-end vertex (cur)
        for nb in _ADJ[cur]:
            if nb != nxt:
                forbidden[nb] = True
        # Also mark nxt's neighbors that are not the next candidate? 
        # Actually, the invariant is: forbidden[v] = True if v is used or 
        # adjacent to any vertex in path[:-1]. 
        # When we add nxt, the new "non-end" vertices include cur (which was end).
        # So neighbors of cur (except nxt) become forbidden.
        # nxt itself is now end, so its neighbors are not yet forbidden.
        cur = nxt

    return path


def _greedy_build_with_lookahead(forbidden, start_vertex, bit_order, lookahead=2):
    """
    Greedy build with lookahead: at each step, choose the neighbor that
    maximizes the number of future options (degree in the remaining graph).
    """
    path = [start_vertex]
    used = set(path)
    for nb in _ADJ[start_vertex]:
        forbidden[nb] = True
    forbidden[start_vertex] = True

    cur = start_vertex
    while True:
        candidates = []
        for b in bit_order:
            v = cur ^ (1 << b)
            if not forbidden[v]:
                candidates.append(v)
        if not candidates:
            break

        if lookahead <= 1 or len(candidates) <= 2:
            nxt = candidates[0]
        else:
            best_score = -1
            nxt = candidates[0]
            for v in candidates:
                # Score: number of neighbors of v that are not forbidden
                # and not cur (since cur is already used)
                score = 0
                for nb in _ADJ[v]:
                    if nb != cur and not forbidden[nb]:
                        score += 1
                if score > best_score:
                    best_score = score
                    nxt = v

        path.append(nxt)
        used.add(nxt)
        for nb in _ADJ[cur]:
            if nb != nxt:
                forbidden[nb] = True
        cur = nxt

    return path


def _local_search(path, bit_order, max_iters=500, start_temp=10.0, end_temp=0.1):
    """
    Simulated-annealing-style local search on a partial path.
    Moves:
    1. Extend the path at the end if possible.
    2. Remove the last vertex and try a different extension.
    3. Insert a vertex at a random position (if valid).
    4. Swap two consecutive vertices (if valid).
    
    We focus on extension and backtracking moves for simplicity and speed.
    """
    if len(path) < 2:
        return path

    best_path = list(path)
    best_len = len(best_path)
    current_path = list(path)
    current_len = len(current_path)

    # Build forbidden set for current_path
    def build_forbidden(p):
        forb = [False] * V
        used = set(p)
        for v in used:
            forb[v] = True
        # For each vertex in p[:-1], mark its neighbors as forbidden
        for i in range(len(p) - 1):
            for nb in _ADJ[p[i]]:
                forb[nb] = True
        return forb

    forbidden = build_forbidden(current_path)

    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters))

    for it in range(max_iters):
        if time.time() > TIME_LIMIT + 5:
            break

        move_type = random.random()
        new_path = None

        if move_type < 0.4:
            # Try to extend
            cur = current_path[-1]
            candidates = [v for v in _ADJ[cur] if not forbidden[v]]
            if candidates:
                v = random.choice(candidates)
                new_path = current_path + [v]
                # Update forbidden
                for nb in _ADJ[cur]:
                    if nb != v:
                        forbidden[nb] = True
                forbidden[v] = True  # v is now used
                current_path = new_path
                current_len += 1
                if current_len > best_len:
                    best_len = current_len
                    best_path = list(current_path)
        elif move_type < 0.7:
            # Backtrack: remove last vertex
            if len(current_path) > 1:
                removed = current_path[-1]
                current_path.pop()
                current_len -= 1
                # Rebuild forbidden (expensive but safe)
                forbidden = build_forbidden(current_path)
                # Try to extend with a different choice
                cur = current_path[-1]
                candidates = [v for v in _ADJ[cur] if not forbidden[v] and v != removed]
                if candidates:
                    v = random.choice(candidates)
                    current_path.append(v)
                    current_len += 1
                    for nb in _ADJ[cur]:
                        if nb != v:
                            forbidden[nb] = True
                    forbidden[v] = True
                    if current_len > best_len:
                        best_len = current_len
                        best_path = list(current_path)
                else:
                    # Keep it short
                    pass
        else:
            # Random restart from a prefix
            if len(current_path) > 2:
                cut = random.randint(2, len(current_path) - 1)
                prefix = current_path[:cut]
                forbidden = build_forbidden(prefix)
                cur = prefix[-1]
                candidates = [v for v in _ADJ[cur] if not forbidden[v]]
                if candidates:
                    v = random.choice(candidates)
                    new_prefix = prefix + [v]
                    forbidden = build_forbidden(new_prefix)
                    # Greedy extend from here
                    for nb in _ADJ[cur]:
                        if nb != v:
                            forbidden[nb] = True
                    forbidden[v] = True
                    cur = v
                    while True:
                        cands = [x for x in _ADJ[cur] if not forbidden[x]]
                        if not cands:
                            break
                        v2 = random.choice(cands)
                        new_prefix.append(v2)
                        for nb in _ADJ[cur]:
                            if nb != v2:
                                forbidden[nb] = True
                        forbidden[v2] = True
                        cur = v2
                    if len(new_prefix) > best_len:
                        best_len = len(new_prefix)
                        best_path = list(new_prefix)
                    current_path = new_prefix
                    current_len = len(new_prefix)

        temp *= cooling

    return best_path


def _beam_polish(path, beam_width=20, max_steps=50):
    """
    Beam search polish: take the current path and try to extend it using
    beam search with a small beam width.
    """
    if not path:
        return path

    # State: (path_tuple, forbidden_tuple)
    # To save memory, we store forbidden as a frozenset of indices
    # But V=512, so a set of ints is fine.

    start_forbidden = [False] * V
    used = set(path)
    for v in used:
        start_forbidden[v] = True
    for i in range(len(path) - 1):
        for nb in _ADJ[path[i]]:
            start_forbidden[nb] = True

    beam = [(list(path), start_forbidden)]

    for step in range(max_steps):
        if time.time() > TIME_LIMIT + 5:
            break
        new_beam = []
        for p, forb in beam:
            cur = p[-1]
            for v in _ADJ[cur]:
                if not forb[v]:
                    new_forb = forb[:]
                    for nb in _ADJ[cur]:
                        if nb != v:
                            new_forb[nb] = True
                    new_forb[v] = True
                    new_p = p + [v]
                    new_beam.append((new_p, new_forb))
        if not new_beam:
            break
        # Sort by length, then by a heuristic (e.g., number of available moves)
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = new_beam[:beam_width]

    if not beam:
        return path
    best = max(beam, key=lambda x: len(x[0]))
    return best[0]


def construct():
    start_time = time.time()

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    best_path = []
    best_len = 0

    # Strategy 1: Greedy with default bit order
    forbidden = [False] * V
    path = _greedy_build(forbidden, 0, list(range(N)))
    if len(path) > best_len:
        best_len = len(path)
        best_path = path

    # Strategy 2: Greedy with lookahead
    forbidden = [False] * V
    path = _greedy_build_with_lookahead(forbidden, 0, list(range(N)), lookahead=2)
    if len(path) > best_len:
        best_len = len(path)
        best_path = path

    # Strategy 3: Randomized greedy restarts
    for i in range(20):
        if time_left() < 10:
            break
        bit_order = list(range(N))
        random.shuffle(bit_order)
        start_vertex = random.randint(0, V - 1)
        forbidden = [False] * V
        path = _greedy_build(forbidden, start_vertex, bit_order)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Strategy 4: Local search on the best path
    if time_left() > 20:
        improved = _local_search(best_path, list(range(N)), max_iters=300, start_temp=10.0, end_temp=0.1)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Strategy 5: Beam polish
    if time_left() > 10:
        polished = _beam_polish(best_path, beam_width=10, max_steps=30)
        if len(polished) > best_len:
            best_len = len(polished)
            best_path = polished

    # Final verification
    if not _verify_snake(best_path):
        # Fallback to a simple valid snake
        best_path = [0]

    return best_path
