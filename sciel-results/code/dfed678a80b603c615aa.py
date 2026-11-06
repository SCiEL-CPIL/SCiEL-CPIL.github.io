"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Design: start from the complete 5-uniform hypergraph K_9^5 (known non-2-colourable),
then prune edges with a fast exact backtracking+unit-propagation checker, and
finally try to reduce the number of vertices. Minimizes edge count.
"""
import time
from itertools import combinations

def _check_non2col(edges, n):
    """Return True if the hypergraph has NO proper 2-colouring (checked exactly
    by backtracking with unit propagation)."""
    if not edges:
        return False
    m = len(edges)
    es = [tuple(e) for e in edges]
    edge_mask = []
    for e in es:
        mm = 0
        for v in e:
            mm |= (1 << v)
        edge_mask.append(mm)

    def rec(col0, col1, uncol, idx):
        # unit propagation
        changed = True
        while changed:
            changed = False
            for i in range(m):
                mm = edge_mask[i]
                if mm & col1 == 0:
                    if mm & col0 == mm:
                        return False
                    if mm & uncol == 0:
                        col1 |= mm
                        uncol &= ~mm
                        changed = True
                elif mm & col0 == 0:
                    if mm & col1 == mm:
                        return False
                    if mm & uncol == 0:
                        col0 |= mm
                        uncol &= ~mm
                        changed = True
        if idx == m:
            return True
        mm = edge_mask[idx]
        if mm & col1 == 0 and mm & col0 == 0 and (mm & uncol) == mm:
            # undecided edge: branch on its vertices
            verts = [v for v in range(n) if (mm >> v) & 1]
            for v in verts:
                if rec(col0 | (1 << v), col1, uncol & ~(1 << v), idx + 1):
                    return True
            return False
        return rec(col0, col1, uncol, idx + 1)

    return rec(0, 0, (1 << n) - 1, 0)


def construct():
    deadline = time.time() + 110.0
    allc = list(combinations(range(9), 5))
    edges = [list(e) for e in allc]
    n = 9
    # verify base
    assert _check_non2col(edges, n)

    def prune(e_list, nn):
        # randomised greedy removal, keep the smallest valid set
        best = list(e_list)
        for _ in range(40):
            if time.time() > deadline - 15:
                break
            cur = list(e_list)
            import random
            random.shuffle(cur)
            for e in cur:
                if time.time() > deadline - 15:
                    break
                rest = [x for x in cur if x != e]
                if _check_non2col(rest, nn):
                    cur = rest
            if len(cur) < len(best):
                best = cur
        return best

    edges = prune(edges, n)

    # try to reduce vertex count: keep only edges within 0..n-1
    for nn in range(n - 1, 4, -1):
        if time.time() > deadline - 10:
            break
        cand = [e for e in edges if max(e) < nn]
        if not cand:
            continue
        if _check_non2col(cand, nn):
            n = nn
            edges = prune(cand, n)
            break

    return {'n': n, 'edges': [list(e) for e in edges]}
