import time
import random

# ---------- Fast 3-colorability check using bitmasks ----------
def _is_3_colorable(adj, n=16):
    """
    adj: list of n sets (or frozensets) of neighbors.
    Returns True if graph is 3-colorable, False otherwise.
    Uses backtracking with bitmasks for speed.
    """
    # Precompute neighbor masks
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask

    # Order vertices by descending degree
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i

    # For each vertex in order, compute mask of neighbors that appear earlier
    earlier_mask = [0] * n
    for v in range(n):
        mask = 0
        for w in adj[v]:
            if pos[w] < pos[v]:
                mask |= (1 << w)
        earlier_mask[v] = mask

    color = [-1] * n

    def backtrack(i):
        if i == n:
            return True
        v = order[i]
        # Determine forbidden colors based on earlier neighbors
        forbidden = 0
        em = earlier_mask[v]
        # Iterate over set bits in em
        m = em
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            c = color[w]
            if c != -1:
                forbidden |= (1 << c)
            m ^= lsb

        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(i + 1):
                    return True
        color[v] = -1
        return False

    return backtrack(0)


def _build_adj(edges, n=16):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_triangle_free(adj, n=16):
    for u in range(n):
        for v in adj[u]:
            if v > u:
                # check common neighbor
                au = adj[u]
                av = adj[v]
                for w in au:
                    if w != v and w in av:
                        return False
    return True


def _can_add_edge(u, v, adj):
    """Check if adding edge (u,v) keeps graph triangle-free."""
    if u == v or v in adj[u]:
        return False
    au = adj[u]
    av = adj[v]
    # triangle iff common neighbor exists
    # iterate over smaller set
    if len(au) < len(av):
        for w in au:
            if w in av:
                return False
    else:
        for w in av:
            if w in au:
                return False
    return True


# ---------- Grötzsch graph (Mycielski of C5) on vertices 0..10 ----------
def _groetzsch_edges():
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin (limit 150s)
    n = 16

    # Start from Grötzsch graph (4-chromatic, triangle-free) on 11 vertices, 5 isolated
    base_edges = _groetzsch_edges()
    best_edges = [list(e) for e in base_edges]
    best_adj = _build_adj(best_edges, n)
    assert _is_triangle_free(best_adj, n)
    assert not _is_3_colorable(best_adj, n)

    best_count = len(best_edges)

    def elapsed_ok():
        return time.time() < deadline

    def try_add(u, v, adj, edges):
        """Return (new_adj, new_edges) if adding (u,v) keeps triangle-free and 4-chromatic."""
        if not _can_add_edge(u, v, adj):
            return None
        new_adj = [s.copy() for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        if _is_3_colorable(new_adj, n):
            return None
        new_edges = edges + [[u, v]]
        return new_adj, new_edges

    # Greedy: repeatedly add any feasible edge (in a fixed order) until none
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    changed = True
    while changed and elapsed_ok():
        changed = False
        for u, v in all_pairs:
            if not elapsed_ok():
                break
            if v in best_adj[u]:
                continue
            res = try_add(u, v, best_adj, best_edges)
            if res is not None:
                best_adj, best_edges = res
                best_count = len(best_edges)
                changed = True

    # Local search with multiple restarts and randomization
    # We'll do several rounds of: random perturbation + greedy + local search

    def to_set(edges):
        return set(frozenset((u, v)) for u, v in edges)

    es = to_set(best_edges)

    # Local search loop: 1-add moves and 2-swap moves
    improved = True
    while improved and elapsed_ok():
        improved = False
        # Move 1: try adding any single edge
        for u, v in all_pairs:
            if not elapsed_ok():
                break
            if frozenset((u, v)) in es:
                continue
            if v in best_adj[u]:
                continue
            res = try_add(u, v, best_adj, best_edges)
            if res is not None:
                best_adj, best_edges = res
                es = to_set(best_edges)
                best_count = len(best_edges)
                improved = True
                break  # restart scan with new state
        if improved or not elapsed_ok():
            continue

        # Move 2: remove one edge, add two edges (net +1)
        for i, e0 in enumerate(list(best_edges)):
            if not elapsed_ok():
                break
            u0, v0 = e0
            # candidate removal
            cand_adj = [s.copy() for s in best_adj]
            cand_adj[u0].discard(v0)
            cand_adj[v0].discard(u0)
            cand_edges = best_edges[:i] + best_edges[i + 1:]
            cand_es = to_set(cand_edges)
            # try to add edges to reach net +1 (add 2)
            added = 0
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in cand_adj[u]:
                    continue
                if frozenset((u, v)) in cand_es:
                    continue
                # check triangle-free with cand_adj
                if not _can_add_edge(u, v, cand_adj):
                    continue
                # tentative add and check 4-chromatic
                tmp_adj = [s.copy() for s in cand_adj]
                tmp_adj[u].add(v)
                tmp_adj[v].add(u)
                if _is_3_colorable(tmp_adj, n):
                    continue
                # accept this addition
                cand_adj = tmp_adj
                cand_edges = cand_edges + [[u, v]]
                cand_es = to_set(cand_edges)
                added += 1
                if added == 2:
                    break
            if added == 2:
                # net +1 edge, and still 4-chromatic (checked), triangle-free (checked)
                best_adj = cand_adj
                best_edges = cand_edges
                es = cand_es
                best_count = len(best_edges)
                improved = True
                break

    # Multiple restarts with random perturbation to escape local optima
    # Keep the best result found across all restarts
    global_best_edges = [list(e) for e in best_edges]
    global_best_count = best_count

    restarts = 0
    while elapsed_ok():
        restarts += 1
        # Perturb: randomly remove some edges from the current best
        perturb_edges = [list(e) for e in global_best_edges]
        # Remove 3-5 random edges
        num_remove = random.randint(3, 5)
        remove_indices = random.sample(range(len(perturb_edges)), min(num_remove, len(perturb_edges)))
        for idx in sorted(remove_indices, reverse=True):
            perturb_edges.pop(idx)

        # Check if still valid (should be, since removing edges preserves 4-chromatic? No, removing edges can make it 3-colorable)
        # Actually, removing edges from a 4-chromatic graph can make it 3-colorable. So we need to check.
        perturb_adj = _build_adj(perturb_edges, n)
        if not _is_triangle_free(perturb_adj, n):
            continue
        if _is_3_colorable(perturb_adj, n):
            continue  # invalid, skip this restart

        # Now do greedy + local search from this perturbed state
        cur_adj = perturb_adj
        cur_edges = perturb_edges
        cur_es = to_set(cur_edges)

        # Greedy phase
        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in cur_adj[u]:
                    continue
                if frozenset((u, v)) in cur_es:
                    continue
                res = try_add(u, v, cur_adj, cur_edges)
                if res is not None:
                    cur_adj, cur_edges = res
                    cur_es = to_set(cur_edges)
                    changed = True

        # Local search phase
        improved = True
        while improved and elapsed_ok():
            improved = False
            # Move 1: try adding any single edge
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if frozenset((u, v)) in cur_es:
                    continue
                if v in cur_adj[u]:
                    continue
                res = try_add(u, v, cur_adj, cur_edges)
                if res is not None:
                    cur_adj, cur_edges = res
                    cur_es = to_set(cur_edges)
                    improved = True
                    break
            if improved or not elapsed_ok():
                continue

            # Move 2: remove one edge, add two edges (net +1)
            for i, e0 in enumerate(list(cur_edges)):
                if not elapsed_ok():
                    break
                u0, v0 = e0
                cand_adj = [s.copy() for s in cur_adj]
                cand_adj[u0].discard(v0)
                cand_adj[v0].discard(u0)
                cand_edges = cur_edges[:i] + cur_edges[i + 1:]
                cand_es = to_set(cand_edges)
                added = 0
                for u, v in all_pairs:
                    if not elapsed_ok():
                        break
                    if v in cand_adj[u]:
                        continue
                    if frozenset((u, v)) in cand_es:
                        continue
                    if not _can_add_edge(u, v, cand_adj):
                        continue
                    tmp_adj = [s.copy() for s in cand_adj]
                    tmp_adj[u].add(v)
                    tmp_adj[v].add(u)
                    if _is_3_colorable(tmp_adj, n):
                        continue
                    cand_adj = tmp_adj
                    cand_edges = cand_edges + [[u, v]]
                    cand_es = to_set(cand_edges)
                    added += 1
                    if added == 2:
                        break
                if added == 2:
                    cur_adj = cand_adj
                    cur_edges = cand_edges
                    cur_es = cand_es
                    improved = True
                    break

        # Check if this restart improved the global best
        if len(cur_edges) > global_best_count:
            global_best_count = len(cur_edges)
            global_best_edges = [list(e) for e in cur_edges]

        # If we've found a very good solution, we can stop early
        # Theoretical max for triangle-free 16-vertex graph: floor(16^2/4) = 64
        # But we need 4-chromatic, so max is less.
        if global_best_count >= 30:
            break

    # Final validation
    final_adj = _build_adj(global_best_edges, n)
    assert _is_triangle_free(final_adj, n)
    assert not _is_3_colorable(final_adj, n)

    return global_best_edges
