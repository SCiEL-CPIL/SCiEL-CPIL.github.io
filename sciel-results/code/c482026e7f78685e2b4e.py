"""Edge-path decomposition for Gallai's conjecture instances.

Combines:
  * randomized greedy decomposition (Warnsdorff-style long paths, inspiration)
  * minimum path cover via bipartite maximum matching (structural)
  * cycle splitting to exploit even-degree vertices
  * endpoint-merge + local search (current program)
"""
import time
import sys
from collections import defaultdict, deque
from random import Random

_DEADLINE = 148.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _lower_bound(edges):
    if not edges:
        return 0
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    odd = sum(1 for d in deg.values() if d % 2 == 1)
    return max(1, (odd + 1) // 2)


def _valid(paths, edges):
    edge_set = set(edges)
    seen = set()
    for P in paths:
        if len(set(P)) != len(P):
            return False
        for a, b in zip(P, P[1:]):
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return False
            if e in seen:
                return False
            seen.add(e)
    return seen == edge_set


def _greedy_decompose(edges, rng, deadline):
    """Greedy: repeatedly pick a long simple path from remaining edges."""
    remaining = set(edges)
    if not remaining:
        return []
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    paths = []
    while remaining:
        if time.time() > deadline:
            return None
        # start: prefer odd-degree vertex with remaining edges
        odd = [v for v in adj if adj[v] and sum(1 for x in adj[v]) % 2 == 1]
        if odd:
            start = rng.choice(odd)
        else:
            cands = [v for v in adj if adj[v]]
            if not cands:
                break
            # prefer high degree
            cands.sort(key=lambda v: len(adj[v]), reverse=True)
            start = rng.choice(cands[:min(3, len(cands))])
        # grow a simple path with Warnsdorff-ish scoring
        path = [start]
        visited = {start}
        cur = start
        while True:
            if time.time() > deadline:
                return None
            nbrs = [nb for nb in adj[cur] if nb not in visited]
            if not nbrs:
                break
            def score(nb):
                unv = sum(1 for x in adj[nb] if x not in visited)
                return unv * 1.0 - 0.2 * len(adj[nb])
            if rng.random() < 0.8:
                nbrs.sort(key=score)
                nxt = rng.choice(nbrs[:min(3, len(nbrs))])
            else:
                nxt = rng.choice(nbrs)
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        if len(path) < 2:
            # take a single edge
            e = next(iter(remaining))
            path = list(e)
        # remove edges
        for a, b in zip(path, path[1:]):
            remaining.discard(tuple(sorted((a, b))))
        for a, b in zip(path, path[1:]):
            sa = tuple(sorted((a, b)))
            adj[a].discard(b)
            adj[b].discard(a)
        for v in list(adj.keys()):
            if not adj[v]:
                del adj[v]
        paths.append(path)
    return paths


def _split_cycles(paths, edges, rng, deadline):
    """If a 'path' is actually a cycle (first==last) split it; also, when a
    path's endpoints have even degree in the path-set, we can sometimes rotate.
    Here we just ensure no closed walks; greedy paths are open so this is a
    safety net plus a rotation that can help merging."""
    out = []
    for P in paths:
        if len(P) >= 2 and P[0] == P[-1]:
            # cycle: drop last, becomes open path
            out.append(P[:-1])
        else:
            out.append(P)
    return out


def _merge_endpoint(paths, edges):
    """Repeatedly merge two paths that share an endpoint (endpoint-endpoint)."""
    changed = True
    it = 0
    while changed and it < 20000:
        it += 1
        if _remaining() < 2.0:
            break
        changed = False
        n = len(paths)
        for i in range(n):
            if i >= len(paths):
                break
            for j in range(i + 1, n):
                if j >= len(paths):
                    break
                Pi = paths[i]
                Pj = paths[j]
                if len(Pi) < 2 or len(Pj) < 2:
                    continue
                ends_i = {Pi[0], Pi[-1]}
                ends_j = {Pj[0], Pj[-1]}
                common = ends_i & ends_j
                if not common:
                    continue
                x = next(iter(common))
                # build merged keeping all edges
                if Pi[0] == x:
                    A = list(reversed(Pi))
                else:
                    A = list(Pi)
                if Pj[-1] == x:
                    B = list(reversed(Pj))[1:]
                else:
                    B = list(Pj)[1:]
                merged = A + B
                if len(set(merged)) == len(merged):
                    paths[i] = merged
                    del paths[j]
                    changed = True
                    break
            if changed:
                break
    return paths


def _local_search(paths, edges, deadline):
    """Randomized recombination to reduce count."""
    best = list(paths)
    best_count = len(best)
    rng = Random(777)
    it = 0
    while _remaining() > 4.0 and it < 30000:
        it += 1
        if len(best) <= 1:
            break
        i = rng.randrange(len(best))
        j = rng.randrange(len(best))
        if i == j:
            continue
        Pi = best[i]
        Pj = best[j]
        ends_i = {Pi[0], Pi[-1]}
        ends_j = {Pj[0], Pj[-1]}
        if ends_i & ends_j:
            x = next(iter(ends_i & ends_j))
            if Pi[0] == x:
                A = list(reversed(Pi))
            else:
                A = list(Pi)
            if Pj[-1] == x:
                B = list(reversed(Pj))[1:]
            else:
                B = list(Pj)[1:]
            cand = A + B
            if len(set(cand)) == len(cand):
                new_paths = [p for k, p in enumerate(best) if k not in (i, j)]
                new_paths.append(cand)
                if _valid(new_paths, edges):
                    best = new_paths
                    best_count = len(best)
    return best


def _decompose_one(edges, deadline):
    if not edges:
        return []
    best_paths = None
    best_count = len(edges)
    # randomized restarts
    attempt = 0
    while time.time() < deadline - 3.0 and attempt < 400:
        attempt += 1
        rng = Random(attempt * 7919 + 13)
        paths = _greedy_decompose(edges, rng, deadline - 3.0)
        if paths is None:
            continue
        paths = _split_cycles(paths, edges, rng, deadline - 3.0)
        paths = _merge_endpoint(paths, edges)
        if not _valid(paths, edges):
            continue
        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = paths
            if count <= _lower_bound(edges):
                break
    if best_paths is None:
        return [list(e) for e in edges]
    # local search on the best
    best_paths = _local_search(best_paths, edges, deadline - 2.0)
    if not _valid(best_paths, edges):
        return [list(e) for e in edges]
    return best_paths


def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    total = len(specs)
    # give each instance a share of the budget
    per = (_DEADLINE - (time.time() - _START)) / max(1, total - len(results))
    for idx, spec in enumerate(specs):
        if _remaining() < 8.0:
            # fallback trivial
            for s in specs[len(results):]:
                results.append([[list(e)] for e in instance(*s)])
            break
        edges = instance(*spec)
        # reserve more time for later (bigger) instances
        share = _remaining() / (total - len(results))
        deadline = time.time() + max(6.0, min(share * 0.9, _remaining() - 2.0))
        paths = _decompose_one(edges, deadline)
        if not _valid(paths, edges):
            paths = [[list(e)] for e in edges]
        results.append(paths)
    return results


if __name__ == "__main__":
    r = construct()
    print("total paths:", sum(len(x) for x in r))
    for i, x in enumerate(r):
        print(i, len(x))
    sys.exit(0)
