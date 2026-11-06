import time
import random
import sys

N = 9
SIZE = 1 << N          # 512
MASK = SIZE - 1        # 0x1FF

# Precompute neighbors and bitmask-of-neighbors for every vertex
NEIGHBORS = [[] for _ in range(SIZE)]
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        nb = v ^ (1 << b)
        NEIGHBORS[v].append(nb)
        m |= (1 << nb)
    NEIGHBOR_MASK[v] = m


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # ------------------------------------------------------------------
    # Helper: check whether a full path is a valid snake (consecutive
    # Hamming distance 1, non-consecutive Hamming distance >= 2).
    # ------------------------------------------------------------------
    def is_valid_snake(path):
        L = len(path)
        if L < 2:
            return True
        # Consecutive check
        for i in range(L - 1):
            if (path[i] ^ path[i + 1]).bit_count() != 1:
                return False
        # Non-consecutive check
        for i in range(L):
            for j in range(i + 2, L):
                if (path[i] ^ path[j]).bit_count() == 1:
                    return False
        return True

    # ------------------------------------------------------------------
    # Phase 0: fast greedy initial snake (deterministic start = 0)
    # ------------------------------------------------------------------
    def greedy_snake(start, rng):
        path = [start]
        visited = 1 << start
        forbidden = NEIGHBOR_MASK[start]  # vertices that would create a chord
        cur = start
        while True:
            # Available neighbours: unvisited, not in forbidden, and not cur
            avail = NEIGHBOR_MASK[cur] & ~visited & ~forbidden
            if avail == 0:
                break
            # Pick a neighbour at random among available ones
            bits = []
            a = avail
            while a:
                lsb = a & (-a)
                bits.append(lsb.bit_length() - 1)
                a ^= lsb
            rng.shuffle(bits)
            nxt = bits[0]
            path.append(nxt)
            visited |= (1 << nxt)
            forbidden |= NEIGHBOR_MASK[nxt]
            cur = nxt
        return path

    # ------------------------------------------------------------------
    # Phase 1: beam search with forbidden_mask
    #
    # State: (path_tuple, visited_int, forbidden_int, current)
    # At each step we keep the top `beam_width` states by a simple
    # heuristic (prefer states with more available moves).
    # ------------------------------------------------------------------
    def beam_search(start, beam_width, max_depth, time_limit, rng):
        # Initial state
        states = [
            (
                [start],                # path (list for speed of append)
                1 << start,             # visited bitmask
                NEIGHBOR_MASK[start],   # forbidden bitmask
                start,                  # current vertex
            )
        ]
        best_path = list(states[0][0])

        for depth in range(max_depth):
            if time.time() > time_limit:
                break
            if not states:
                break

            next_states = []
            for path, visited, forbidden, cur in states:
                avail = NEIGHBOR_MASK[cur] & ~visited & ~forbidden
                if avail == 0:
                    continue
                # Enumerate available neighbours
                a = avail
                cands = []
                while a:
                    lsb = a & (-a)
                    v = lsb.bit_length() - 1
                    cands.append(v)
                    a ^= lsb
                # Score each candidate: how many moves it leaves (lookahead 1)
                scored = []
                for v in cands:
                    new_visited = visited | (1 << v)
                    new_forbidden = forbidden | NEIGHBOR_MASK[v]
                    # Count available moves from v after adding it
                    moves = (NEIGHBOR_MASK[v] & ~new_visited & ~new_forbidden).bit_count()
                    scored.append((moves, v))
                # Sort descending by available moves (keep the most promising)
                scored.sort(key=lambda x: -x[0])
                # Take up to beam_width // len(states) + 1 per parent to keep total bounded
                per_parent = max(1, beam_width // max(1, len(states)))
                for moves, v in scored[:per_parent]:
                    new_path = path + [v]
                    new_visited = visited | (1 << v)
                    new_forbidden = forbidden | NEIGHBOR_MASK[v]
                    next_states.append((new_path, new_visited, new_forbidden, v))
                    if len(new_path) > len(best_path):
                        best_path = list(new_path)

            if not next_states:
                break

            # Prune to beam_width: keep states with most available moves
            def state_score(s):
                path, visited, forbidden, cur = s
                return (NEIGHBOR_MASK[cur] & ~visited & ~forbidden).bit_count()

            next_states.sort(key=state_score, reverse=True)
            states = next_states[:beam_width]

        return best_path

    # ------------------------------------------------------------------
    # Phase 2: local improvement – try to extend the best path found
    # by restarting the beam from each endpoint / interior vertex.
    # ------------------------------------------------------------------
    def try_extend(path, time_limit, rng):
        """Try to extend path from both ends using short beam searches."""
        best = list(path)

        # Try extending from the tail (append)
        tail = path[-1]
        visited = 0
        for v in path:
            visited |= (1 << v)
        forbidden = 0
        for v in path[:-1]:  # chord constraint: new vertex can't be adj to path[0..-2]
            forbidden |= NEIGHBOR_MASK[v]

        # Beam from tail
        ext = beam_search(tail, 4, 20, time_limit, rng)
        # ext starts at tail; prepend path[:-1]
        candidate = path[:-1] + ext
        if len(candidate) > len(best) and is_valid_snake(candidate):
            best = candidate

        # Try extending from the head (prepend)
        head = path[0]
        visited = 0
        for v in path:
            visited |= (1 << v)
        forbidden = 0
        for v in path[1:]:  # new vertex can't be adj to path[1..]
            forbidden |= NEIGHBOR_MASK[v]

        ext2 = beam_search(head, 4, 20, time_limit, rng)
        candidate2 = ext2 + path[1:]
        if len(candidate2) > len(best) and is_valid_snake(candidate2):
            best = candidate2

        return best

    # ------------------------------------------------------------------
    # Main loop: phased multi-restart
    # ------------------------------------------------------------------
    best_path = greedy_snake(0, random.Random(0))
    best_len = len(best_path)

    # Phase A: many quick greedy restarts (cheap baseline improvement)
    t_phase_a = time.time() + 5.0
    seed = 42
    while time.time() < t_phase_a:
        seed += 1
        s = random.Random(seed)
        start_v = s.randint(0, SIZE - 1)
        p = greedy_snake(start_v, s)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)

    # Phase B: beam searches with increasing width
    # Allocate remaining time to beam searches of various widths.
    remaining = deadline - time.time() - 3.0  # safety margin
    if remaining > 1:
        # Try several beam widths, each getting a time slice
        widths = [8, 16, 32, 64]
        slices = [0.15, 0.20, 0.25, 0.40]  # fractions of remaining time
        for w, frac in zip(widths, slices):
            t_slice = time.time() + remaining * frac
            if time.time() >= t_slice:
                continue
            # Run a few restarts within this slice
            max_restarts = 5
            per_restart_time = (t_slice - time.time()) / max_restarts
            for r in range(max_restarts):
                if time.time() >= t_slice - 0.5:
                    break
                s = random.Random(seed + 1000 + w * 100 + r)
                start_v = s.randint(0, SIZE - 1)
                p = beam_search(start_v, w, 250, time.time() + per_restart_time, s)
                if len(p) > best_len:
                    best_path = p
                    best_len = len(p)

    # Phase C: local improvement on the best path found
    remaining = deadline - time.time() - 2.0
    if remaining > 2:
        t_local = time.time() + remaining * 0.7
        # Try extending from several interior points
        L = len(best_path)
        # Pick a few anchor points
        anchors = set()
        if L >= 4:
            anchors.add(0)
            anchors.add(L - 1)
            anchors.add(L // 2)
            anchors.add(L // 4)
            anchors.add(3 * L // 4)
        for a in anchors:
            if time.time() > t_local:
                break
            s = random.Random(seed + 5000 + a)
            improved = try_extend(best_path, time.time() + (t_local - time.time()) / max(1, len(anchors) - len([x for x in anchors if x <= a])), s)
            if len(improved) > best_len:
                best_path = improved
                best_len = len(improved)

    # Final safety check
    if not is_valid_snake(best_path):
        # Fallback: regenerate a simple valid snake
        best_path = greedy_snake(0, random.Random(999))

    return best_path


if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p) - 1}")
    # Validate
    valid = True
    for i in range(len(p) - 1):
        if (p[i] ^ p[i + 1]).bit_count() != 1:
            valid = False
            break
    if valid:
        for i in range(len(p)):
            for j in range(i + 2, len(p)):
                if (p[i] ^ p[j]).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                break
    print(f"Valid: {valid}")
