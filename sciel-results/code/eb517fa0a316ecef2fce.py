"""
Snake-in-the-box Q_9: Bidirectional Warnsdorff growth with bitmask validity,
local insertion, and iterative deepening backtracking.
Optimized for speed: numpy adjacency, precomputed masks, tight time budget.
"""
import time
import random
import numpy as np

N = 9
MAX_V = 1 << N

# Precompute neighbors as a numpy array for vectorized ops
_NEIGH = np.zeros((MAX_V, N), dtype=np.int32)
for v in range(MAX_V):
    for b in range(N):
        _NEIGH[v, b] = v ^ (1 << b)

# ADJ_MASK: bit i set iff v and i are adjacent (i.e., Hamming distance 1)
# For a 9-bit hypercube, vertex v is adjacent to exactly 9 vertices.
# We represent the set of neighbors as a 512-bit integer (Python int).
ADJ_MASK = [0] * MAX_V
for v in range(MAX_V):
    m = 0
    for b in range(N):
        m |= (1 << (v ^ (1 << b)))
    ADJ_MASK[v] = m

# Also precompute the neighbor list as a flat list of lists for iteration
NEIGHBORS = [[] for _ in range(MAX_V)]
for v in range(MAX_V):
    NEIGHBORS[v] = [v ^ (1 << b) for b in range(N)]


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Returns a list of distinct integers (vertices) forming an induced path.
    """
    start_time = time.time()
    DEADLINE = start_time + 112.0  # safety margin before 120s kill

    best_path = None
    best_len = 0

    def time_ok():
        return time.time() < DEADLINE

    def valid_tail(v, path, used):
        """Can v be appended to tail? v~path[-1], v not in used, v not adj to path[:-1]"""
        if used >> v & 1:
            return False
        # v must not be adjacent to any vertex in path except path[-1]
        other = used ^ (1 << path[-1])
        return (ADJ_MASK[v] & other) == 0

    def valid_head(v, path, used):
        """Can v be prepended to head? v~path[0], v not in used, v not adj to path[1:]"""
        if used >> v & 1:
            return False
        other = used ^ (1 << path[0])
        return (ADJ_MASK[v] & other) == 0

    def onward_count(v, path, used, end):
        """Count valid onward moves from v after adding it.
        end=-1: v is new tail; end=0: v is new head."""
        new_used = used | (1 << v)
        cnt = 0
        if end == -1:
            # Next w from v: w~v, w not in new_used, w not adj to path[:-1]
            # w adj to v, v adj to path[-1]. w could be adj to path[-1] only if w==path[-1] (in used).
            # So check w against path[:-1] = used ^ (1<<path[-1])
            exclude = used ^ (1 << path[-1])
            for w in NEIGHBORS[v]:
                if new_used >> w & 1:
                    continue
                if ADJ_MASK[w] & exclude:
                    continue
                cnt += 1
        else:
            exclude = used ^ (1 << path[0])
            for w in NEIGHBORS[v]:
                if new_used >> w & 1:
                    continue
                if ADJ_MASK[w] & exclude:
                    continue
                cnt += 1
        return cnt

    def grow_bidir(path, used, rng, max_steps=400):
        """Bidirectional Warnsdorff growth."""
        steps = 0
        while steps < max_steps and time_ok():
            # Collect valid tail moves
            tail_cands = []
            for v in NEIGHBORS[path[-1]]:
                if used >> v & 1:
                    continue
                if ADJ_MASK[v] & (used ^ (1 << path[-1])):
                    continue
                sc = onward_count(v, path, used, -1)
                tail_cands.append((sc, v))

            # Collect valid head moves
            head_cands = []
            for v in NEIGHBORS[path[0]]:
                if used >> v & 1:
                    continue
                if ADJ_MASK[v] & (used ^ (1 << path[0])):
                    continue
                sc = onward_count(v, path, used, 0)
                head_cands.append((sc, v))

            if not tail_cands and not head_cands:
                break

            # Choose end: prefer the end with fewer options (Warnsdorff)
            if not tail_cands:
                go_head = True
            elif not head_cands:
                go_head = False
            elif len(head_cands) < len(tail_cands):
                go_head = True
            elif len(tail_cands) < len(head_cands):
                go_head = False
            else:
                go_head = rng.random() < 0.5

            if go_head:
                min_sc = min(c[0] for c in head_cands)
                pool = [c[1] for c in head_cands if c[0] == min_sc]
                v = rng.choice(pool)
                path.insert(0, v)
                used |= (1 << v)
            else:
                min_sc = min(c[0] for c in tail_cands)
                pool = [c[1] for c in tail_cands if c[0] == min_sc]
                v = rng.choice(pool)
                path.append(v)
                used |= (1 << v)
            steps += 1

        return path, used

    def local_insert(path, used, rng, max_iters=50):
        """Try to insert a vertex between consecutive path vertices."""
        cur = path[:]
        cur_used = used
        improved = False
        for _ in range(max_iters):
            if not time_ok():
                break
            if len(cur) < 2:
                break
            pos = rng.randint(1, len(cur) - 1)
            a = cur[pos - 1]
            c = cur[pos]
            # v must satisfy: v~a, v~c, v not in cur_used, v not adj to cur except a,c
            cands = []
            for v in NEIGHBORS[a]:
                if v == c or (cur_used >> v & 1):
                    continue
                if (v ^ c).bit_count() != 1:
                    continue
                other = cur_used ^ (1 << a) ^ (1 << c)
                if ADJ_MASK[v] & other:
                    continue
                cands.append(v)
            if cands:
                v = rng.choice(cands)
                cur.insert(pos, v)
                cur_used |= (1 << v)
                improved = True
        if improved:
            return cur, cur_used
        return path, used

    def backtrack_and_regrow(path, used, rng, depth=6):
        """Backtrack up to `depth` vertices from the tail, try a different branch, regrow."""
        if len(path) < 4:
            return path, used, False
        for d in range(1, min(depth, len(path) - 1) + 1):
            if not time_ok():
                break
            body = path[:-d]
            body_used = 0
            for v in body:
                body_used |= (1 << v)
            end = body[-1]
            valid = []
            for v in NEIGHBORS[end]:
                if body_used >> v & 1:
                    continue
                if ADJ_MASK[v] & (body_used ^ (1 << end)):
                    continue
                valid.append(v)
            if valid:
                rng.shuffle(valid)
                v = valid[0]
                new_path = body + [v]
                new_used = body_used | (1 << v)
                new_path, new_used = grow_bidir(new_path, new_used, rng, max_steps=60)
                return new_path, new_used, True
        return path, used, False

    def run_attempt(rng):
        """One full construction attempt."""
        v0 = rng.randint(0, MAX_V - 1)
        path = [v0]
        used = 1 << v0
        v1 = rng.choice(NEIGHBORS[v0])
        path.append(v1)
        used |= (1 << v1)

        # Main growth
        path, used = grow_bidir(path, used, rng, max_steps=500)

        # Local insertion
        path, used = local_insert(path, used, rng, max_iters=40)

        # If stuck short, backtrack and regrow
        if len(path) < 200:
            path, used, ok = backtrack_and_regrow(path, used, rng, depth=8)
            if ok:
                path, used = local_insert(path, used, rng, max_iters=20)

        # Another round of insertion
        path, used = local_insert(path, used, rng, max_iters=20)

        return path

    # Multi-start loop
    attempt = 0
    while time_ok():
        attempt += 1
        seed = attempt * 2654435761 + 12345
        rng = random.Random(seed)
        path = run_attempt(rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

        # If we've done a lot of attempts and are going well, keep going
        # but tighten the per-attempt budget implicitly via time checks

    if best_path is None:
        # Fallback greedy
        path = [0]
        used = 1
        while True:
            found = False
            for v in NEIGHBORS[path[-1]]:
                if used >> v & 1:
                    continue
                if ADJ_MASK[v] & (used ^ (1 << path[-1])):
                    continue
                path.append(v)
                used |= (1 << v)
                found = True
                break
            if not found:
                break
        return path

    return best_path
