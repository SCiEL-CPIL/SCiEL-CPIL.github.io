"""Find minimal non-2-colorable 5-uniform hypergraph via greedy set-cover over colorings."""
import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 15.0


def _is_2_colorable(n, edges, deadline):
    """True iff there exists a 2-colouring with NO monochromatic edge.
    Uses backtracking with unit propagation."""
    edges = [tuple(e) for e in edges]
    m = len(edges)
    if m == 0:
        return True

    color = [-1] * n
    e0 = [0] * m
    e1 = [0] * m
    uncol = [list(e) for e in edges]

    def propagate(queue):
        while queue:
            ei = queue.pop()
            u = uncol[ei]
            if e0[ei] == 5 or e1[ei] == 5:
                return False
            if len(u) == 1:
                v = u[0]
                if e0[ei] == 4:
                    forced = 1
                elif e1[ei] == 4:
                    forced = 0
                else:
                    continue
                if color[v] == -1:
                    color[v] = forced
                    uncol[ei].remove(v)
                    if forced == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    for ej in range(m):
                        if ej != ei and v in uncol[ej]:
                            uncol[ej].remove(v)
                            if forced == 0:
                                e0[ej] += 1
                            else:
                                e1[ej] += 1
                            if e0[ej] == 5 or e1[ej] == 5:
                                return False
                            if len(uncol[ej]) == 1:
                                queue.append(ej)
                elif color[v] != forced:
                    return False
        return True

    import sys
    sys.setrecursionlimit(100000)

    def rec(k):
        if time.time() > deadline:
            return None
        if k >= n:
            return True
        for c in (0, 1):
            color[k] = c
            touched = []
            for ei, e in enumerate(edges):
                if k in e:
                    uncol[ei].remove(k)
                    if c == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    touched.append(ei)
                    if e0[ei] == 5 or e1[ei] == 5:
                        color[k] = -1
                        for ej in touched:
                            uncol[ej].append(k)
                            if c == 0:
                                e0[ej] -= 1
                            else:
                                e1[ej] -= 1
                        break
            else:
                queue = [ei for ei in touched if len(uncol[ei]) == 1]
                if propagate(queue):
                    result = rec(k + 1)
                    if result is not None:
                        return result
                color[k] = -1
                for ej in touched:
                    if k in uncol[ej]:
                        uncol[ej].append(k)
                    if c == 0:
                        e0[ej] -= 1
                    else:
                        e1[ej] -= 1
                continue
        return False

    result = rec(0)
    if result is None:
        return False
    return result


def _build_edge_masks(n):
    """Precompute for each 5-edge the bitmask of colorings it covers.
    Coloring c in 0..2^n-1: bit i of c = color of vertex i.
    Edge e is monochromatic in coloring c iff all 5 vertices have same color.
    Returns (all_edges, list_of_masks) where each mask is a Python int with bit c set
    if coloring c makes edge e monochromatic.
    """
    all_edges = list(combinations(range(n), 5))
    masks = []
    for e in all_edges:
        v0, v1, v2, v3, v4 = e
        mask = 0
        for c in range(1 << n):
            b0 = (c >> v0) & 1
            b1 = (c >> v1) & 1
            b2 = (c >> v2) & 1
            b3 = (c >> v3) & 1
            b4 = (c >> v4) & 1
            if b0 == b1 == b2 == b3 == b4:
                mask |= (1 << c)
        masks.append(mask)
    return all_edges, masks


def _greedy_set_cover(n, all_edges, masks, deadline):
    """Greedy set cover: pick edges to cover all 2^n colorings."""
    total_colorings = 1 << n
    uncovered = (1 << total_colorings) - 1
    chosen = []
    used = [False] * len(all_edges)

    while uncovered and time.time() < deadline:
        best_gain = 0
        best_idx = -1
        for i in range(len(all_edges)):
            if used[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        used[best_idx] = True
        chosen.append(all_edges[best_idx])
        uncovered &= ~masks[best_idx]

    if uncovered == 0:
        return chosen
    return None


def _local_prune(n, edges, deadline):
    """Try to remove edges while maintaining non-2-colorability."""
    edges = [list(e) for e in edges]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        order = list(range(len(edges)))
        random.shuffle(order)
        for i in order:
            if i >= len(edges):
                continue
            candidate = edges[:i] + edges[i+1:]
            if not _is_2_colorable(n, candidate, deadline):
                del edges[i]
                improved = True
                break
    return edges


def _try_n(n, deadline, rng):
    """Try greedy set-cover on n vertices, then prune."""
    if time.time() > deadline:
        return None
    all_edges, masks = _build_edge_masks(n)
    result = _greedy_set_cover(n, all_edges, masks, deadline)
    if result is None:
        return None
    pruned = _local_prune(n, result, deadline)
    return pruned


def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(42)

    fallback = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    best = fallback

    for n in (11, 10, 12, 9, 13):
        if time.time() > deadline:
            break
        try:
            res = _try_n(n, deadline, rng)
        except Exception:
            res = None
        if res is not None and len(res) < len(best['edges']):
            best = {'n': n, 'edges': res}

    return best


if __name__ == '__main__':
    _t0 = time.time()
    _res = construct()
    _ok = _is_2_colorable(_res['n'], _res['edges'], _t0 + 120.0)
    assert not _ok, "constructed hypergraph must NOT be 2-colourable"
    print("edges:", len(_res['edges']), "n:", _res['n'], "2-colourable:", _ok)
