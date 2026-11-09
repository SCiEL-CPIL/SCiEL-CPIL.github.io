import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    n = 16
    E = n * (n - 1) // 2  # 120

    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    edge_idx_map = {}
    inv_edge_map = [None] * E
    for u in range(n):
        for v in range(u + 1, n):
            idx = edge_index(u, v)
            edge_idx_map[(u, v)] = idx
            inv_edge_map[idx] = (u, v)

    # ---- Build all Q3 edge-sets as global bitmasks (exact subgraph enumeration) ----
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    local_edge_to_idx = {}
    idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = idx
            idx += 1

    local_q3_sets = set()
    for p in permutations(range(8)):
        es = set()
        for a, b in q3_abstract_edges:
            x, y = p[a], p[b]
            es.add(local_edge_to_idx[(min(x, y), max(x, y))])
        if len(es) == 12:
            local_q3_sets.add(frozenset(es))

    all_subsets = list(combinations(range(n), 8))
    global_masks = set()
    for subset in all_subsets:
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                ug, vg = subset[u_loc], subset[v_loc]
                l2g[le] = edge_idx_map[(min(ug, vg), max(ug, vg))]
        for es in local_q3_sets:
            m = 0
            for le in es:
                m |= (1 << l2g[le])
            global_masks.add(m)

    q3_masks = list(global_masks)
    NQ = len(q3_masks)
    q3_edges = []
    for m in q3_masks:
        edges = []
        mm = m
        while mm:
            lsb = mm & -mm
            edges.append(lsb.bit_length() - 1)
            mm ^= lsb
        q3_edges.append(edges)

    def time_left():
        return deadline - time.time()

    # ---- Greedy hitting set: remove edges until every Q3 is hit (valid graph) ----
    removed = [False] * E
    q3s_of_edge = [[] for _ in range(E)]
    for i, edges in enumerate(q3_edges):
        for e in edges:
            q3s_of_edge[e].append(i)
    active = [True] * NQ
    counts = [0] * E
    for e in range(E):
        counts[e] = len(q3s_of_edge[e])

    while time_left() > 8.0:
        best_e, best_c = -1, 0
        for e in range(E):
            if not removed[e] and counts[e] > best_c:
                best_c, best_e = counts[e], e
        if best_e < 0:
            break
        removed[best_e] = True
        for q in q3s_of_edge[best_e]:
            if active[q]:
                active[q] = False
                for e2 in q3_edges[q]:
                    if e2 != best_e:
                        counts[e2] -= 1

    present = [not removed[e] for e in range(E)]
    present_mask = 0
    for e in range(E):
        if present[e]:
            present_mask |= (1 << e)

    # ---- Fast validity: is edge e the only missing edge in any Q3? ----
    # For each Q3, precompute which edges it contains. A Q3 is "complete" in the
    # current graph iff all its edges are present. Adding edge e creates a Q3 iff
    # some Q3 contains e, lacks e, and has all its other edges present.
    q3_missing = [0] * NQ  # number of edges of q3 not present
    for q in range(NQ):
        c = 0
        for e in q3_edges[q]:
            if not present[e]:
                c += 1
        q3_missing[q] = c

    def creates_q3_if_add(e):
        # Q3s that contain e and are missing exactly {e}
        for q in q3s_of_edge[e]:
            if q3_missing[q] == 1 and not present[e]:
                return True
        return False

    # ---- Local search: add edges, then 2-opt swaps (remove one, add another) ----
    def apply_add(e):
        present[e] = True
        present_mask_bit = (1 << e)
        for q in q3s_of_edge[e]:
            q3_missing[q] -= 1

    def apply_remove(e):
        present[e] = False
        for q in q3s_of_edge[e]:
            q3_missing[q] += 1

    def try_add_pass():
        changed = True
        while changed and time_left() > 4.0:
            changed = False
            for e in range(E):
                if not present[e] and not creates_q3_if_add(e):
                    apply_add(e)
                    changed = True
                    if time.time() > deadline - 4.0:
                        return

    def swap_search():
        # 2-opt: remove e, add f. Valid iff no Q3 contains e, lacks f, and has all
        # other edges present (after removing e). We check candidate removals.
        improved = True
        while improved and time_left() > 4.0:
            improved = False
            for e in range(E):
                if not present[e]:
                    continue
                # temporarily consider removing e
                present[e] = False
                for q in q3s_of_edge[e]:
                    q3_missing[q] += 1
                # now try to add some f not present
                for f in range(E):
                    if present[f]:
                        continue
                    ok = True
                    for q in q3s_of_edge[f]:
                        # q contains f, lacks f (present[f] False). After removing e,
                        # q is complete iff q3_missing[q] == 1 (only f missing) -> would form Q3
                        if q3_missing[q] == 1:
                            ok = False
                            break
                    if ok:
                        apply_add(f)
                        improved = True
                        break
                # restore e
                present[e] = True
                for q in q3s_of_edge[e]:
                    q3_missing[q] -= 1
                if time.time() > deadline - 4.0:
                    return

    # Initial polish
    try_add_pass()
    swap_search()

    def current_count():
        return sum(1 for e in range(E) if present[e])

    best_present = [p for p in present]
    best_count = current_count()

    # ---- Simulated annealing over swaps to escape local optima ----
    rng = random.Random(12345)
    T = 6.0
    cooling = 0.995
    while time_left() > 5.0 and T > 0.05:
        # pick a random present edge e to remove
        present_edges = [e for e in range(E) if present[e]]
        if not present_edges:
            break
        e = rng.choice(present_edges)
        present[e] = False
        for q in q3s_of_edge[e]:
            q3_missing[q] += 1
        # pick a random absent edge f to add
        absent_edges = [e for e in range(E) if not present[e]]
        if absent_edges:
            f = rng.choice(absent_edges)
            # check validity
            ok = True
            for q in q3s_of_edge[f]:
                if q3_missing[q] == 1:
                    ok = False
                    break
            if ok:
                apply_add(f)
                # accept (it's valid); keep
            else:
                # invalid: revert (restore e, do not add f)
                present[e] = True
                for q in q3s_of_edge[e]:
                    q3_missing[q] -= 1
        else:
            present[e] = True
            for q in q3s_of_edge[e]:
                q3_missing[q] -= 1
        # occasional re-add pass to fill gaps
        if rng.random() < 0.15:
            try_add_pass()
        # occasional swap search
        if rng.random() < 0.08:
            swap_search()
        if current_count() > best_count:
            best_count = current_count()
            best_present = [p for p in present]
        T *= cooling

    # Final exact polish
    for p in best_present:
        pass
    present = [p for p in best_present]
    for q in range(NQ):
        c = 0
        for e in q3_edges[q]:
            if not present[e]:
                c += 1
        q3_missing[q] = c
    try_add_pass()
    swap_search()

    edges = []
    for e in range(E):
        if present[e]:
            u, v = inv_edge_map[e]
            edges.append([u, v])
    return edges
