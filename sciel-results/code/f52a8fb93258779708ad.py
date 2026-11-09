import time
import random
import sys
from collections import deque

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s wall-clock limit


def construct():
    """
    Construct the longest induced path (snake) in Q_9.

    Structural core: represent the set of used vertices as a single integer
    bitmask (pathmask).  Validity of a candidate end-extension c is then an
    O(1) bit-op: c must be unused and its only path-neighbor must be the end
    it attaches to, i.e. popcount(pathmask & adjmask[c]) == 1.  Scoring uses
    Warnsdorff (min unused-neighbor count) = popcount(adjmask[c] & ~pathmask).
    This removes the O(N) deg_in_path bookkeeping of the previous design and
    the O(N*L) improvement passes, freeing the budget for many more
    multi-start restarts (the measured 1.221 -> 1.317 improvement mechanism).
    """
    start_time = time.time()
    best_path = None
    best_len = 0

    # Precompute adjacency bitmask and neighbor list for every vertex.
    adjmask = [0] * MAX_VERTS
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        m = 0
        nb = []
        for b in range(N):
            u = v ^ (1 << b)
            m |= (1 << u)
            nb.append(u)
        adjmask[v] = m
        neighbors[v] = nb

    def bidirectional_grow(start_vertex=0, time_limit=None, randomness=0.1):
        """Grow a snake bidirectionally with O(1) bitmask validity checks."""
        v0 = start_vertex
        dq = deque([v0])
        pathmask = (1 << v0)
        complement = (1 << MAX_VERTS) - 1  # all bits set

        # First step: any unused neighbor of v0 is valid (v0 is its only path-neighbor).
        first = neighbors[v0]
        if not first:
            return list(dq)
        v1 = random.choice(first)
        dq.append(v1)
        pathmask |= (1 << v1)

        while True:
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break

            left_end = dq[0]
            right_end = dq[-1]

            # Candidate extensions for the left end (must be adjacent to left_end).
            valid_left = [v for v in neighbors[left_end]
                          if not (pathmask >> v) & 1
                          and bin(pathmask & adjmask[v]).count("1") == 1]
            # Candidate extensions for the right end.
            valid_right = [v for v in neighbors[right_end]
                           if not (pathmask >> v) & 1
                           and bin(pathmask & adjmask[v]).count("1") == 1]

            if not valid_left and not valid_right:
                break

            def best_of(cands):
                """Warnsdorff (min onward moves) + small randomness; returns (vertex, score)."""
                if not cands:
                    return None, float('inf')
                best_v = None
                best_s = float('inf')
                for v in cands:
                    onward = bin(adjmask[v] & ~pathmask).count("1")
                    s = onward + random.random() * randomness
                    if s < best_s:
                        best_s = s
                        best_v = v
                return best_v, best_s

            lv, ls = best_of(valid_left)
            rv, rs = best_of(valid_right)

            if valid_left and (not valid_right or ls <= rs):
                dq.appendleft(lv)
                pathmask |= (1 << lv)
            else:
                dq.append(rv)
                pathmask |= (1 << rv)

        return list(dq)

    # Multi-start: random start vertices break symmetry; vary greediness.
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        rand_param = random.choice([0.0, 0.1, 0.5])
        p = bidirectional_grow(start_vertex=v0,
                               time_limit=DEADLINE - (time.time() - start_time),
                               randomness=rand_param)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        return [0]

    return best_path


# EVOLVE-BLOCK-END
