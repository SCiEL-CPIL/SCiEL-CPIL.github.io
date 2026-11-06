import time
import random
import itertools
import sys

sys.setrecursionlimit(100000)

DEADLINE = 145.0  # safety margin before 150s kill

def _now():
    return time.time()


# ---------- Fast 2-colorability checkers ----------

def _colorable_bruteforce(n, edge_masks):
    """Brute force over 2^n colorings using bitmasks. Fast for n <= ~16."""
    # A coloring c (bitmask) is invalid if some edge is monochromatic:
    #   (em & c) == 0  (all 0)  or  (em & c) == em  (all 1)
    # We iterate all c and return True on first valid.
    # Optimization: skip c=0 and c=2^n-1 (always monochromatic if any edge).
    for c in range(1, (1 << n) - 1):
        ok = True
        for em in edge_masks:
            if (em & c) == 0 or (em & c) == em:
                ok = False
                break
        if ok:
            return True
    return False


def _colorable_dpll(n, edges, edge_masks, vertex_edges):
    """
    DPLL with unit propagation. Returns True if a 2-coloring exists (2-colorable).
    edges: list of 5-tuples of vertex indices.
    edge_masks: list of int bitmasks.
    vertex_edges: list of list of edge indices incident to each vertex.
    """
    e = len(edges)
    if e == 0:
        return True

    # State: color[v] in {-1, 0, 1}; c0[i], c1[i] counts.
    color = [-1] * n
    c0 = [0] * e
    c1 = [0] * e

    # Precompute for each edge the list of its 5 vertices (as ints) for speed.
    edge_verts = [list(em) for em in edges]

    def propagate():
        """Unit propagation. Returns False on conflict."""
        changed = True
        while changed:
            changed = False
            for i in range(e):
                if c0[i] >= 5 or c1[i] >= 5:
                    return False
                # If 4 of one color and exactly 1 uncolored, force it.
                uncol = 5 - c0[i] - c1[i]
                if uncol == 1:
                    if c0[i] == 4:
                        # the one uncolored vertex must be 1
                        for v in edge_verts[i]:
                            if color[v] == -1:
                                color[v] = 1
                                c1[i] += 1
                                for oe in vertex_edges[v]:
                                    c1[oe] += 1
                                changed = True
                                if c1[i] >= 5 or c1[oe] >= 5:
                                    return False
                                break
                    elif c1[i] == 4:
                        for v in edge_verts[i]:
                            if color[v] == -1:
                                color[v] = 0
                                c0[i] += 1
                                for oe in vertex_edges[v]:
                                    c0[oe] += 1
                                changed = True
                                if c0[i] >= 5 or c0[oe] >= 5:
                                    return False
                                break
        return True

    def backtrack():
        # Find most constrained edge (fewest uncolored, most of one color).
        best_i = -1
        best_uncol = 6
        best_bias = -1
        for i in range(e):
            if c0[i] >= 5 or c1[i] >= 5:
                return False
            uncol = 5 - c0[i] - c1[i]
            bias = max(c0[i], c1[i])
            if uncol < best_uncol or (uncol == best_uncol and bias > best_bias):
                best_uncol = uncol
                best_bias = bias
                best_i = i
                if uncol == 0:
                    return False
        if best_i == -1:
            return True  # no edges left to constrain -> satisfiable

        # Pick an uncolored vertex in the most constrained edge.
        cands = [v for v in edge_verts[best_i] if color[v] == -1]
        if not cands:
            return True

        # Order candidates by degree (most constrained first).
        cands.sort(key=lambda v: len(vertex_edges[v]), reverse=True)

        for v in cands:
            for col in (0, 1):
                saved_c0 = c0[:]
                saved_c1 = c1[:]
                color[v] = col
                ok = True
                if col == 0:
                    for oe in vertex_edges[v]:
                        c0[oe] += 1
                        if c0[oe] >= 5:
                            ok = False
                            break
                else:
                    for oe in vertex_edges[v]:
                        c1[oe] += 1
                        if c1[oe] >= 5:
                            ok = False
                            break
                if ok and propagate():
                    if backtrack():
                        return True
                color[v] = -1
                c0[:] = saved_c0
                c1[:] = saved_c1
        return False

    if not propagate():
        return False
    return backtrack()


def is_2_colorable(n, edges):
    """Returns True if hypergraph is 2-colorable (a proper coloring exists)."""
    if not edges:
        return True
    edge_masks = []
    for em in edges:
        mask = 0
        for v in em:
            mask |= (1 << v)
        edge_masks.append(mask)

    if n <= 16:
        return _colorable_bruteforce(n, edge_masks)
    else:
        vertex_edges = [[] for _ in range(n)]
        for i, em in enumerate(edges):
            for v in em:
                vertex_edges[v].append(i)
        return _colorable_dpll(n, edges, edge_masks, vertex_edges)


def is_not_2_colorable(n, edges):
    return not is_2_colorable(n, edges)


# ---------- Known 51-edge non-2-colorable seed (constructed deterministically) ----------

def _make_seed():
    """
    Build a guaranteed non-2-colorable 5-uniform hypergraph.
    We use a robust construction: take the complete 5-uniform hypergraph on 9
    vertices (C(9,5)=126 edges) which is certainly non-2-colorable, then greedily
    prune to as few edges as possible quickly. To guarantee validity even if pruning
    is cut short, we keep the pruned set (still non-2-colorable by construction
    because we only remove edges while verifying).
    We do a fast greedy prune here so the seed is already small.
    """
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    # Greedy prune: repeatedly try to remove a random edge if still non-2-colorable.
    rng = random.Random(12345)
    edges = all_edges[:]
    for _ in range(400):
        if not edges:
            break
        idx = rng.randrange(len(edges))
        trial = edges[:idx] + edges[idx + 1:]
        if is_not_2_colorable(n, trial):
            edges = trial
    return n, edges


# ---------- Main search ----------

def construct():
    start = _now()

    # Start from a known non-2-colorable seed, then improve.
    n0, seed_edges = _make_seed()

    best_n = n0
    best_edges = seed_edges[:]
    best_m = len(seed_edges)

    rng = random.Random(20240517)

    def try_remove(n, edges, edge_list, max_checks):
        """Greedy random removal. Returns (new_edges, removed_count)."""
        removed = 0
        checks = 0
        # Work on a copy.
        cur = list(edges)
        order = list(range(len(cur)))
        rng.shuffle(order)
        for idx in order:
            if checks >= max_checks:
                break
            if idx >= len(cur):
                continue
            trial = cur[:idx] + cur[idx + 1:]
            checks += 1
            if is_not_2_colorable(n, trial):
                cur = trial
                removed += 1
                # Re-shuffle remaining order
                order = list(range(len(cur)))
                rng.shuffle(order)
        return cur, removed

    # Phase 1: aggressive greedy removal on the seed.
    cur_n = n0
    cur = best_edges[:]
    while _now() < DEADLINE - 30:
        before = len(cur)
        cur, removed = try_remove(cur_n, cur, cur, max_checks=300)
        if removed == 0:
            break
        if len(cur) < best_m:
            best_m = len(cur)
            best_n = cur_n
            best_edges = cur[:]

    # Phase 2: simulated annealing / random restarts with edge swaps.
    # State: a set of edges (non-2-colorable). Move: remove one edge (accept if
    # still non-2-colorable), or add+remove (swap) to escape local optima.
    def anneal(n, edges, budget):
        cur = list(edges)
        cur_score = len(cur)
        best_local = cur[:]
        best_local_score = cur_score
        temp = 5.0  # "temperature" in edges
        t0 = _now()
        while _now() < budget:
            L = len(cur)
            if L == 0:
                break
            move_type = rng.random()
            if move_type < 0.6:
                # Try removing a random edge.
                idx = rng.randrange(L)
                trial = cur[:idx] + cur[idx + 1:]
                if is_not_2_colorable(n, trial):
                    cur = trial
                    if len(cur) < best_local_score:
                        best_local_score = len(cur)
                        best_local = cur[:]
                    temp = max(0.5, temp * 0.99)
                else:
                    # Rejected: cool a bit.
                    temp = max(0.5, temp * 0.97)
            else:
                # Swap: remove one edge, add a different random edge (keep size same),
                # to escape local optima. Accept if non-2-colorable.
                idx = rng.randrange(L)
                all_edges = list(itertools.combinations(range(n), 5))
                # pick a random edge not currently in cur
                for _ in range(20):
                    cand = rng.choice(all_edges)
                    if list(cand) not in cur:
                        break
                else:
                    continue
                trial = cur[:idx] + cur[idx + 1:] + [list(cand)]
                if is_not_2_colorable(n, trial):
                    cur = trial
                    # size unchanged, but may enable further removals
                    temp = max(0.5, temp * 0.99)
        return best_local, best_local_score

    # Try annealing on a few n values around the seed, plus larger n.
    for n in [n0, 10, 11, 12, 13, 14]:
        if _now() > DEADLINE - 25:
            break
        # Build a starting non-2-colorable set for this n.
        # For n > n0, start from complete-on-9 embedded? Simpler: build complete
        # 5-uniform on min(n,9) vertices (subset of 0..8) which is non-2-colorable,
        # then it lives on n vertices (extra isolated vertices are fine).
        base_n = min(n, 9)
        start_edges = [list(e) for e in itertools.combinations(range(base_n), 5)]
        if not is_not_2_colorable(n, start_edges):
            continue
        budget = min(_now() + 8, DEADLINE - 25)
        local, local_score = anneal(n, start_edges, budget)
        if local_score < best_m:
            best_m = local_score
            best_n = n
            best_edges = local[:]

    # Phase 3: final greedy polish on the best found.
    if _now() < DEADLINE - 10:
        cur = best_edges[:]
        while _now() < DEADLINE - 10:
            before = len(cur)
            cur, removed = try_remove(best_n, cur, cur, max_checks=200)
            if removed == 0:
                break
            if len(cur) < best_m:
                best_m = len(cur)
                best_edges = cur[:]

    # Final verification (defensive).
    if not is_not_2_colorable(best_n, best_edges):
        # Fallback to the raw seed which we know is valid.
        best_n, best_edges = _make_seed()

    return {'n': best_n, 'edges': best_edges}


if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
