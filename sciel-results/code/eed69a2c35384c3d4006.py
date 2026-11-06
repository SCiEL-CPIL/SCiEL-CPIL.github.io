"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter


def _elapsed_guard(start, margin=5.0):
    return time.time() - start < 150.0 - margin


def construct():
    start = time.time()

    def ok():
        return _elapsed_guard(start)

    # Precompute all 4-cycles of the K_{10,10} core.
    # left = 0..9 (index i), right = 10..19 (index j = v-10).
    # A 4-cycle is determined by a left pair (a,b) and right pair (c,d).
    all_cycles = []
    for a, b in itertools.combinations(range(10), 2):
        for c, d in itertools.combinations(range(10), 2):
            all_cycles.append(frozenset([(a, c), (a, d), (b, c), (b, d)]))

    # edge -> list of cycles containing it (for fast polish checks)
    edge_to_cycles = {}
    for cyc in all_cycles:
        for e in cyc:
            edge_to_cycles.setdefault(e, []).append(cyc)

    best = None
    best_len = -1

    for k in range(1, 13):
        if not ok():
            break

        # Full K_{10,10} with linear colouring (i*k + j) mod 20.
        emap = {}
        for i in range(10):
            for j in range(10):
                emap[(i, j)] = (i * k + j) % 20

        # Identify non-rainbow 4-cycles.
        bad = []
        for cyc in all_cycles:
            cols = [emap[e] for e in cyc]
            if len(set(cols)) < 4:
                bad.append(cyc)

        # Greedy hitting set: repeatedly remove the edge in the most bad cycles.
        cnt = Counter()
        for cyc in bad:
            for e in cyc:
                cnt[e] += 1
        removed = set()
        active = list(bad)
        while active:
            best_e = None
            best_c = -1
            for e in cnt:
                if e in emap and cnt[e] > best_c:
                    best_c = cnt[e]
                    best_e = e
            if best_e is None:
                break
            del emap[best_e]
            removed.add(best_e)
            new_active = []
            for cyc in active:
                if best_e in cyc:
                    for e in cyc:
                        cnt[e] -= 1
                else:
                    new_active.append(cyc)
            active = new_active

        # Backup the guaranteed-valid (post-removal) state.
        backup = dict(emap)

        # Polish: re-add removed edges that no longer create a bad 4-cycle.
        for e in list(removed):
            if not ok():
                break
            e_colour = (e[0] * k + e[1]) % 20
            safe = True
            for cyc in edge_to_cycles[e]:
                others = [x for x in cyc if x != e]
                if all(o in emap for o in others):
                    cols = [emap[o] for o in others] + [e_colour]
                    if len(set(cols)) < 4:
                        safe = False
                        break
            if safe:
                emap[e] = e_colour
                removed.discard(e)

        edges = [[i, 10 + j, c] for (i, j), c in emap.items()]

        # Verify validity; fall back to the guaranteed-valid backup if broken.
        emap2 = {}
        for e in edges:
            emap2[(e[0], e[1] - 10)] = e[2]
        valid = True
        for cyc in all_cycles:
            if all(e in emap2 for e in cyc):
                cols = [emap2[e] for e in cyc]
                if len(set(cols)) < 4:
                    valid = False
                    break
        if not valid:
            emap = backup
            edges = [[i, 10 + j, c] for (i, j), c in emap.items()]

        if len(edges) > best_len:
            best_len = len(edges)
            best = edges

    if best is None:
        # A star has no 4-cycle, so it is always valid.
        return [[0, v, 0] for v in range(1, 20)]
    return best
# EVOLVE-BLOCK-END
