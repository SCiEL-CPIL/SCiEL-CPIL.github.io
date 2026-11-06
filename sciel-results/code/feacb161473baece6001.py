"""Fresh design: minimize edges of a non-2-colorable 5-uniform hypergraph."""
import time
import random
from itertools import combinations

DEADLINE = 110.0  # safety margin under 120s wall clock
_start = time.time()


def _time_left():
    return DEADLINE - (time.time() - _start)


def _edge_mask(edge):
    m = 0
    for v in edge:
        m |= (1 << v)
    return m


def _is_2_colorable(n, masks):
    """Return True if hypergraph with vertex set 0..n-1 and edge masks is 2-colorable."""
    N = 1 << n
    # state: (c0, c1, i) -> True if satisfiable from here
    memo = {}

    def prop(c0, c1, i):
        # unit propagation: an edge with all but one vertex in c0 forces last into c1, etc.
        changed = True
        while changed:
            changed = False
            for j in range(i, len(masks)):
                m = masks[j]
                # vertices forced to 0 (in c0)
                z = m & c0
                o = m & c1
                if z == m:
                    return (c0, c1, i, False)  # edge all in c0 -> monochromatic 0, bad? 
                    # Wait: edge monochromatic means NOT a proper coloring. We want to AVOID monochromatic edges.
                    # So an edge fully in c0 is a violation.
                if o == m:
                    return (c0, c1, i, False)
                # count how many vertices of m are unassigned
                un = m & ~(c0 | c1)
                # if all assigned except one, force it
                if un and (un & (un - 1)) == 0:
                    vbit = un
                    if z == m & ~vbit:
                        # all other in c0 -> this must be 1
                        c1 |= vbit
                        changed = True
                    elif o == m & ~vbit:
                        c0 |= vbit
                        changed = True
        return (c0, c1, i, True)

    def rec(c0, c1, i):
        if i >= len(masks):
            return True
        key = (c0, c1, i)
        if key in memo:
            return memo[key]
        r = prop(c0, c1, i)
        if not r[3]:
            memo[key] = False
            return False
        c0, c1, i, _ = r
        m = masks[i]
        un = m & ~(c0 | c1)
        # if edge already has both colors -> satisfied, skip
        if (m & c0) and (m & c1):
            memo[key] = rec(c0, c1, i + 1)
            return memo[key]
        # if edge has no unassigned vertices -> it's monochromatic -> fail
        if un == 0:
            memo[key] = False
            return False
        # pick a vertex in un to branch
        vbit = un & (-un)
        v = (vbit.bit_length() - 1)
        res = rec(c0 | vbit, c1, i) or rec(c0, c1 | vbit, i)
        memo[key] = res
        return res

    return rec(0, 0, 0)


def _core_extract(n, edges):
    """Iteratively remove edges that are not needed for non-2-colorability."""
    masks = [_edge_mask(e) for e in edges]
    # keep track of which are present
    present = [True] * len(masks)
    improved = True
    while improved:
        if _time_left() < 5.0:
            break
        improved = False
        # try removing each present edge
        for idx in range(len(masks)):
            if not present[idx]:
                continue
            if _time_left() < 5.0:
                break
            present[idx] = False
            sub = [masks[j] for j in range(len(masks)) if present[j]]
            # if still non-2-colorable, keep removed
            if not _is_2_colorable(n, sub):
                improved = True
            else:
                present[idx] = True
    final = [masks[j] for j in range(len(masks)) if present[j]]
    # convert back to edges
    out = []
    for m in final:
        e = []
        for v in range(n):
            if m & (1 << v):
                e.append(v)
        out.append(e)
    return out


def _try_seed(n, edges):
    """Run core extraction on a seed; return minimized edge list."""
    if _time_left() < 10.0:
        return None
    # quick check seed is non-2-colorable
    masks = [_edge_mask(e) for e in edges]
    if _is_2_colorable(n, masks):
        return None
    return _core_extract(n, edges)


def construct():
    best = None
    best_count = 10 ** 9

    def consider(n, edges):
        nonlocal best, best_count
        if edges is None:
            return
        # verify
        masks = [_edge_mask(e) for e in edges]
        if _is_2_colorable(n, masks):
            return
        # dedupe
        seen = set()
        uniq = []
        for e in edges:
            t = tuple(sorted(e))
            if t not in seen:
                seen.add(t)
                uniq.append(list(t))
        if len(uniq) < best_count and 5 <= n <= 26 and len(uniq) <= 200:
            best = {'n': n, 'edges': uniq}
            best_count = len(uniq)

    # Seed 1: complete 5-graph on 9 vertices
    edges9 = [list(e) for e in combinations(range(9), 5)]
    r = _try_seed(9, edges9)
    consider(9, r)

    # Try more seeds if time allows: complete 5-graph on 10,11 vertices
    for n in (10, 11):
        if _time_left() < 15.0:
            break
        edges = [list(e) for e in combinations(range(n), 5)]
        r = _try_seed(n, edges)
        consider(n, r)

    # Random sparse seeds: pick random edges until non-2-colorable, then extract
    if _time_left() > 20.0:
        for n in (9, 10, 11, 12):
            if _time_left() < 15.0:
                break
            for _ in range(200):
                if _time_left() < 15.0:
                    break
                k = random.randint(40, 150)
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:k]]
                masks = [_edge_mask(e) for e in edges]
                if not _is_2_colorable(n, masks):
                    r = _core_extract(n, edges)
                    consider(n, r)

    if best is None:
        # fallback: the original complete 5-graph on 9
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    return best
