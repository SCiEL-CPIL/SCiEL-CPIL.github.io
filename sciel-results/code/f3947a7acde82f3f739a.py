"""C4-free edge sets in Q7 via MIS on the conflict graph + local search."""
import time
import random

def construct():
    start = time.time()
    DEADLINE = 112.0          # hard wall-clock guard (task limit 120s)
    d = 7
    N = 1 << d               # 128 vertices
    E = N * d // 2           # 448 edges

    # ---- Build edges and index ----
    edges = []
    edge_idx = {}
    for u in range(N):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_idx[(u, v)] = len(edges)
                edges.append((u, v))

    # ---- Conflict graph: two edges conflict if they belong to a common 2-face ----
    # For each 2-face (fix all bits except a,b) the 4 boundary edges conflict pairwise.
    conflict = [set() for _ in range(E)]
    for a in range(d):
        for b in range(d):
            if a == b:
                continue
            if a > b:
                continue
            # vertices of the 2-face: vary bits a and b
            base = 0
            # pick a representative: clear bits a,b, others 0
            for x in range(N):
                if (x >> a) & 1 == 0 and (x >> b) & 1 == 0:
                    base = x
                    break
            v0 = base
            v1 = base | (1 << a)
            v2 = base | (1 << b)
            v3 = base | (1 << a) | (1 << b)
            fe = [
                (min(v0, v1), max(v0, v1)),
                (min(v2, v3), max(v2, v3)),
                (min(v0, v2), max(v0, v2)),
                (min(v1, v3), max(v1, v3)),
            ]
            ids = [edge_idx[e] for e in fe]
            for k in range(4):
                for l in range(k + 1, 4):
                    conflict[ids[k]].add(ids[l])
                    conflict[ids[l]].add(ids[k])

    # ---- Fast "does adding edge i create a forbidden C4" check ----
    # A C4 is completed when the other 3 edges of some face containing i are present.
    # Precompute, per edge, the list of faces (as 4-edge id tuples) it participates in.
    faces_of = [[] for _ in range(E)]
    for a in range(d):
        for b in range(d):
            if a >= b:
                continue
            base = 0
            for x in range(N):
                if (x >> a) & 1 == 0 and (x >> b) & 1 == 0:
                    base = x
                    break
            v0 = base
            v1 = base | (1 << a)
            v2 = base | (1 << b)
            v3 = base | (1 << a) | (1 << b)
            fe = [
                (min(v0, v1), max(v0, v1)),
                (min(v2, v3), max(v2, v3)),
                (min(v0, v2), max(v0, v2)),
                (min(v1, v3), max(v1, v3)),
            ]
            ids = [edge_idx[e] for e in fe]
            for k in range(4):
                faces_of[ids[k]].append(ids)

    def can_add(i, present):
        for face in faces_of[i]:
            others = [j for j in face if j != i]
            if all(present[j] for j in others):
                return False
        return True

    # ---- Greedy coloring-style initialization: order by degree desc, add if possible ----
    deg = [len(c) for c in conflict]
    order = sorted(range(E), key=lambda i: -deg[i])
    present = bytearray(E)
    for i in order:
        if can_add(i, present):
            present[i] = 1

    def count():
        return sum(present)

    best = count()
    best_present = present[:]

    # ---- Weighted local search: try flips, prioritizing high-degree nodes ----
    # Weights: prefer to keep high-degree edges (they are "costly" to drop).
    w = [1.0 + deg[i] / 100.0 for i in range(E)]

    def gain_add(i):
        # benefit of adding i minus penalty of forced removals
        if present[i]:
            return 0.0
        if not can_add(i, present):
            return 0.0
        return w[i]

    def gain_remove(i):
        if not present[i]:
            return 0.0
        # removing i frees its conflict partners; value = w[i] but we lose it
        return -w[i]

    rng = random.Random(12345)
    # candidate pool: all edges, but we sample with degree bias
    pool = list(range(E))
    pool_w = [deg[i] + 1 for i in range(E)]

    no_improve = 0
    while time.time() - start < DEADLINE:
        # pick a random edge biased by degree
        i = rng.choices(pool, weights=pool_w, k=1)[0]
        if present[i]:
            # try to remove it; only accept if it lets us add more later -> use 2-opt
            # simple: remove and then greedily re-add neighbors
            before = count()
            present[i] = 0
            # re-add its conflict neighbors that are now free
            for j in conflict[i]:
                if not present[j] and can_add(j, present):
                    present[j] = 1
            after = count()
            if after > before:
                best = max(best, after)
                best_present = present[:]
                no_improve = 0
            else:
                present[i] = 1
                no_improve += 1
        else:
            if can_add(i, present):
                present[i] = 1
                before = count()
                # after adding, maybe remove some low-value conflict neighbors to add others
                for j in conflict[i]:
                    if present[j] and deg[j] < deg[i]:
                        # tentatively remove j
                        present[j] = 0
                        # try to add its other neighbors
                        added = 0
                        for k in conflict[j]:
                            if not present[k] and can_add(k, present):
                                present[k] = 1
                                added += 1
                        after = count()
                        if after > before:
                            before = after
                        else:
                            # revert
                            present[j] = 1
                            for k in conflict[j]:
                                if present[k] and k != i:
                                    # revert additions made above
                                    # (only those we added in this branch)
                                    pass
                if count() > best:
                    best = count()
                    best_present = present[:]
                    no_improve = 0
                no_improve += 1
        # periodic reset to best if stuck
        if no_improve > 400:
            present = best_present[:]
            no_improve = 0

    out = []
    for i in range(E):
        if best_present[i]:
            u, v = edges[i]
            out.append([u, v])
    return out
