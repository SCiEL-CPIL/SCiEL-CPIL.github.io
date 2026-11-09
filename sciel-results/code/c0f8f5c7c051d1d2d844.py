import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before the 120s/150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_incompat(edges, adj):
        m = len(edges)
        # bitset of incompatible edges for each edge
        incompat = [0] * m
        for i in range(m):
            u1, v1 = edges[i]
            bit_i = 1 << i
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                bad = False
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    bad = True
                else:
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        bad = True
                if bad:
                    incompat[i] |= (1 << j)
                    incompat[j] |= (1 << i)
        return incompat, m

    def greedy_mis_color(incompat, m, rng):
        """Pack edges into colour classes via repeated MIS extraction."""
        remaining = (1 << m) - 1
        classes = []  # list of sets of edge indices
        while remaining:
            # candidate edges in remaining
            cands = []
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                cands.append(idx)
                r ^= lsb
            # order by descending incompatibility-degree within remaining, random tie-break
            cands.sort(key=lambda x: (-(incompat[x] & remaining).bit_count(), rng.random()))
            chosen = []
            avail = remaining
            for idx in cands:
                if not (avail & (1 << idx)):
                    continue
                chosen.append(idx)
                # remove idx and all its incompatible partners from avail
                avail &= ~(incompat[idx] | (1 << idx))
            classes.append(chosen)
            # remove chosen from remaining
            for idx in chosen:
                remaining &= ~(1 << idx)
        return classes

    def merge_classes(classes, incompat, m):
        """Try to merge colour classes to reduce the total count."""
        classes = [set(c) for c in classes]
        changed = True
        # iterate until no more merges possible
        while changed:
            changed = False
            # work on a copy list, remove empty
            classes = [c for c in classes if c]
            for a in range(len(classes)):
                for b in range(a + 1, len(classes)):
                    if not classes[a] or not classes[b]:
                        continue
                    # try to move edges from larger into smaller where compatible
                    big, small = (classes[a], classes[b]) if len(classes[a]) >= len(classes[b]) else (classes[b], classes[a])
                    moved = []
                    for idx in list(big):
                        # idx can join small if it is compatible with every member of small
                        ok = True
                        for other in small:
                            if incompat[idx] & (1 << other):
                                ok = False
                                break
                        if ok:
                            moved.append(idx)
                    if moved:
                        for idx in moved:
                            big.discard(idx)
                            small.add(idx)
                        changed = True
        # final cleanup of empty classes
        classes = [c for c in classes if c]
        return classes

    def solve_instance(edges, adj, time_left):
        incompat, m = build_incompat(edges, adj)
        if m == 0:
            return []

        def classes_to_colors(classes):
            colors = [0] * m
            for cidx, cls in enumerate(classes):
                for idx in cls:
                    colors[idx] = cidx
            return colors

        best_classes = None
        rng = random.Random(12345)

        # A couple of MIS restarts with different tie-break seeds
        for restart in range(4):
            if time.time() > time_left:
                break
            classes = greedy_mis_color(incompat, m, rng)
            classes = merge_classes(classes, incompat, m)
            if best_classes is None or len(classes) < len(best_classes):
                best_classes = classes

        # If we still have time, do a few more restarts to try to beat it
        extra = 0
        while time.time() < time_left - 2.0 and extra < 30:
            extra += 1
            classes = greedy_mis_color(incompat, m, rng)
            classes = merge_classes(classes, incompat, m)
            if len(classes) < len(best_classes):
                best_classes = classes

        return classes_to_colors(best_classes)

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # Fallback: cannot produce a valid colouring for this instance in time.
            # Return an empty list as a last-resort (will be invalid, but avoids a crash).
            out.append([])
            continue
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colors = solve_instance(edges, adj, deadline)
        out.append(colors)

    return out
