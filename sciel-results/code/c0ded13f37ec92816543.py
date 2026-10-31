import numpy as np
import time
import itertools
from collections import defaultdict


def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin under 150s kill

    D = 11

    # ---- Build the universe of weight-4 vectors in {-1,0,1}^11 ----
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330

    all_vectors = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)

    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280

    # ---- Conflict graph: two vectors conflict iff dot > 2 (i.e. dot >= 3) ----
    # Only supports sharing >=3 coords can have dot >= 3.
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)

    support_pairs = set()
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                support_pairs.add((min(i, j), max(i, j)))

    adj = [[] for _ in range(n_total)]
    for s1, s2 in support_pairs:
        base1 = s1 * 16
        base2 = s2 * 16
        v1 = all_vectors_np[base1:base1 + 16]
        v2 = all_vectors_np[base2:base2 + 16]
        d = v1 @ v2.T
        idx = np.argwhere(d > 2)
        for i, j in idx:
            a = base1 + i
            b = base2 + j
            adj[a].append(b)
            adj[b].append(a)

    # Convert to sorted neighbor arrays
    nbr = [np.array(sorted(adj[i]), dtype=np.int64) for i in range(n_total)]
    deg0 = np.array([len(nbr[i]) for i in range(n_total)], dtype=np.int64)

    rng = np.random.default_rng(12345)

    # ---- Greedy warm-start (minimum-degree) to seed the incumbent ----
    def greedy_min_deg(order):
        avail = np.ones(n_total, dtype=bool)
        deg = deg0.copy()
        selected = np.zeros(n_total, dtype=bool)
        for u in order:
            if not avail[u]:
                continue
            selected[u] = True
            avail[u] = False
            for v in nbr[u]:
                if avail[v]:
                    avail[v] = False
                    for w in nbr[v]:
                        if avail[w]:
                            deg[w] -= 1
        return int(np.sum(selected))

    # ---- Tomita branch-and-bound for maximum independent set ----
    # We use a recursive branch-and-bound with a candidate set P.
    # Order vertices by degree within P (greedy coloring bound).

    def tomita(order, incumbent):
        # order: permutation of vertices; we process in this order
        # We maintain P as a set of candidate indices.
        # Use a list-based P with the standard Tomita scheme.
        # For efficiency we work with the neighbor arrays.

        best = [incumbent]

        # color_sort: greedy coloring of P to get upper bound
        def color_sort(P):
            # P is a list of vertices. Returns (P_sorted, colors) where colors[i]
            # is an upper bound on the size of an IS in P[i:].
            # Greedy coloring: assign color classes.
            uncolored = list(P)
            color_classes = []
            while uncolored:
                color_class = []
                remaining = []
                # pick vertices greedily that are not adjacent to each other in this class
                # Actually for coloring bound we color so that adjacent vertices get different colors.
                # Greedy: iterate, if vertex not adjacent to any already in color_class, add it.
                used_in_class = set()
                for v in uncolored:
                    conflict = False
                    for u in color_class:
                        # check adjacency
                        if _adjacent(u, v):
                            conflict = True
                            break
                    if not conflict:
                        color_class.append(v)
                        used_in_class.add(v)
                    else:
                        remaining.append(v)
                color_classes.append(color_class)
                uncolored = remaining
            # Now build sorted P and color bounds
            P_sorted = []
            colors = []
            for cc in color_classes:
                for v in cc:
                    P_sorted.append(v)
                    colors.append(len(color_classes))
            return P_sorted, colors

        # Fast adjacency test using a global set of neighbor pairs
        # Precompute a set of frozenset pairs for O(1) adjacency.
        # Too large? 5280^2 is fine to build a set of sorted tuples.
        # Actually build a boolean matrix is 5280^2 = 27.9M bools ~ 28MB, ok.
        # But we already have neighbor arrays; use a set of pairs for speed.
        # We'll build adjacency as a set of (min,max) tuples.
        adj_set = set()
        for i in range(n_total):
            for j in nbr[i]:
                if i < j:
                    adj_set.add((i, j))
                else:
                    adj_set.add((j, i))

        def _adjacent(u, v):
            if u < v:
                return (u, v) in adj_set
            else:
                return (v, u) in adj_set

        # Precompute neighbor sets for faster intersection
        nbr_set = [set(nbr[i].tolist()) for i in range(n_total)]

        def branch(P, colors, depth):
            if time.time() - start_time > deadline - 3:
                return
            if not P:
                if depth > best[0]:
                    best[0] = depth
                return
            # Upper bound: depth + colors[-1]
            if depth + colors[-1] <= best[0]:
                return
            # Process vertices in reverse color order (higher color first)
            i = len(P) - 1
            while i >= 0:
                if time.time() - start_time > deadline - 3:
                    return
                v = P[i]
                # If including v can't beat best, prune
                if depth + 1 + colors[i] <= best[0]:
                    i -= 1
                    continue
                # Branch: include v
                new_P = [u for u in P[:i] if u not in nbr_set[v]]
                if new_P:
                    new_P, new_colors = color_sort(new_P)
                    branch(new_P, new_colors, depth + 1)
                else:
                    if depth + 1 > best[0]:
                        best[0] = depth + 1
                i -= 1

        # Initial P = all vertices in `order`
        P = list(order)
        P, colors = color_sort(P)
        branch(P, colors, 0)
        return best[0]

    # Precompute adj_set and nbr_set once (move out of tomita to avoid rebuild)
    adj_set = set()
    for i in range(n_total):
        for j in nbr[i]:
            if i < j:
                adj_set.add((i, j))
            else:
                adj_set.add((j, i))
    nbr_set = [set(nbr[i].tolist()) for i in range(n_total)]

    def _adjacent(u, v):
        if u < v:
            return (u, v) in adj_set
        else:
            return (v, u) in adj_set

    def color_sort(P):
        uncolored = list(P)
        color_classes = []
        while uncolored:
            color_class = []
            remaining = []
            for v in uncolored:
                conflict = False
                for u in color_class:
                    if _adjacent(u, v):
                        conflict = True
                        break
                if not conflict:
                    color_class.append(v)
                else:
                    remaining.append(v)
            color_classes.append(color_class)
            uncolored = remaining
        P_sorted = []
        colors = []
        for cc in color_classes:
            for v in cc:
                P_sorted.append(v)
                colors.append(len(color_classes))
        return P_sorted, colors

    def tomita(order, incumbent):
        best = [incumbent]

        def branch(P, colors, depth):
            if time.time() - start_time > deadline - 3:
                return
            if not P:
                if depth > best[0]:
                    best[0] = depth
                return
            if depth + colors[-1] <= best[0]:
                return
            i = len(P) - 1
            while i >= 0:
                if time.time() - start_time > deadline - 3:
                    return
                v = P[i]
                if depth + 1 + colors[i] <= best[0]:
                    i -= 1
                    continue
                new_P = [u for u in P[:i] if u not in nbr_set[v]]
                if new_P:
                    new_P, new_colors = color_sort(new_P)
                    branch(new_P, new_colors, depth + 1)
                else:
                    if depth + 1 > best[0]:
                        best[0] = depth + 1
                i -= 1

        P = list(order)
        P, colors = color_sort(P)
        branch(P, colors, 0)
        return best[0]

    # ---- Run greedy warm-starts to get a good incumbent ----
    best_k = 0
    best_selected = None

    # A few greedy restarts with random orderings
    for r in range(30):
        if time.time() - start_time > deadline - 40:
            break
        order = list(range(n_total))
        rng.shuffle(order)
        k = greedy_min_deg(order)
        if k > best_k:
            best_k = k
            # Reconstruct selected for this order
            avail = np.ones(n_total, dtype=bool)
            deg = deg0.copy()
            selected = np.zeros(n_total, dtype=bool)
            for u in order:
                if not avail[u]:
                    continue
                selected[u] = True
                avail[u] = False
                for v in nbr[u]:
                    if avail[v]:
                        avail[v] = False
                        for w in nbr[v]:
                            if avail[w]:
                                deg[w] -= 1
            best_selected = selected.copy()

    # ---- Run Tomita with several orderings to try to beat the incumbent ----
    # Use degree-based ordering (ascending degree) which often helps Tomita.
    deg_order = list(np.argsort(deg0))
    if time.time() - start_time < deadline - 5:
        k = tomita(deg_order, best_k)
        if k > best_k:
            best_k = k

    # A couple of randomized orderings for Tomita
    for r in range(5):
        if time.time() - start_time > deadline - 5:
            break
        order = list(range(n_total))
        rng.shuffle(order)
        k = tomita(order, best_k)
        if k > best_k:
            best_k = k

    # ---- Reconstruct the final selected set if we only have a count ----
    # If best_selected is None (e.g., Tomita improved count but we didn't track),
    # re-run a greedy to get a valid set of size >= best_k is not guaranteed.
    # Safer: if best_selected is None, do a final greedy and use that.
    if best_selected is None:
        order = list(range(n_total))
        rng.shuffle(order)
        avail = np.ones(n_total, dtype=bool)
        deg = deg0.copy()
        selected = np.zeros(n_total, dtype=bool)
        for u in order:
            if not avail[u]:
                continue
            selected[u] = True
            avail[u] = False
            for v in nbr[u]:
                if avail[v]:
                    avail[v] = False
                    for w in nbr[v]:
                        if avail[w]:
                            deg[w] -= 1
        best_selected = selected.copy()

    # ---- Local improvement: try to add any compatible vertex to best_selected ----
    selected = best_selected
    improved = True
    while improved and (time.time() - start_time < deadline - 2):
        improved = False
        sel_idx = np.where(selected)[0]
        for i in range(n_total):
            if selected[i]:
                continue
            # check if i conflicts with any selected
            ok = True
            for j in sel_idx:
                if _adjacent(i, j):
                    ok = False
                    break
            if ok:
                selected[i] = True
                improved = True

    result = all_vectors_np[selected].astype(np.int64)
    return result
