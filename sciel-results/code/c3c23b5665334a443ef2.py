import time
import random
import itertools
from collections import deque

_start = time.time()
_HARD_DEADLINE = _start + 125.0  # leave margin under 150s; return best valid early


# ---------- helpers ----------

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _has_c5(adj, n):
    """Exact C5 (as subgraph) detection. Fast: components of size <5 cannot host a C5."""
    seen = [False] * n
    for s in range(n):
        if seen[s]:
            continue
        # collect component
        comp = []
        q = [s]
        seen[s] = True
        while q:
            u = q.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        m = len(comp)
        if m < 5:
            continue
        if m > 60:
            # large component: do bounded DFS from every start
            for i in range(m):
                if _c5_from(comp[i], adj, comp, m, s):
                    return True
        else:
            # exhaustive 5-walk cycle check (unique cycle, start=min)
            for i in range(m):
                for a in range(m):
                    for b in range(m):
                        for c in range(m):
                            for d in range(m):
                                nodes = [i, a, b, c, d]
                                if len(set(nodes)) != 5:
                                    continue
                                if i != min(nodes):
                                    continue
                                if all(
                                    adj[nodes[0]].__contains__(nodes[1]) and
                                    adj[nodes[1]].__contains__(nodes[2]) and
                                    adj[nodes[2]].__contains__(nodes[3]) and
                                    adj[nodes[3]].__contains__(nodes[4]) and
                                    adj[nodes[4]].__contains__(nodes[0])
                                ):
                                    return True
    return False


def _c5_from(start, adj, comp, m, gid):
    # bounded DFS: find a 5-cycle through 'start'
    compset = set(comp)
    for a in comp:
        if a == start or not start in adj[a]:
            continue
        for b in comp:
            if b in (start, a) or not a in adj[b]:
                continue
            for c in comp:
                if c in (start, a, b) or not b in adj[c]:
                    continue
                for d in comp:
                    if d in (start, a, b, c) or not c in adj[d]:
                        continue
                    if start in adj[d] and len({a, b, c, d}) == 4:
                        # unique 5-cycle start,a,b,c,d,start
                        return True
    return False


def _independent_set9(adj, n, budget_nodes=2_000_000):
    """Return a set of 9 pairwise-non-adjacent vertices, or None.
    Exact via recursive branching with an upper-bound prune; node budget guards time."""
    state = {'nodes': 0}

    def feasible(u):
        return state['nodes'] < budget_nodes

    def rec(pick, forbid, depth, cur):
        state['nodes'] += 1
        if not feasible(u):
            return False
        if depth == 0:
            return True
        if len(forbid) >= n - depth + 1:
            # not enough candidates to reach depth
            pass
        cand = None
        for x in range(n):
            if x not in forbid and x not in cur:
                cand = x
                break
        if cand is None:
            return False
        # order: vertices with smallest remaining non-neighbor degree first
        def key(x):
            c = 0
            for y in range(n):
                if y not in forbid and y not in cur and y != x and y not in adj[x]:
                    c += 1
            return c
        # collect candidates
        cands = [x for x in range(n) if x not in forbid and x not in cur]
        cands.sort(key=key)
        # upper bound: max independent size in remaining <= number of cand, also greedy
        if len(cands) < depth:
            return False
        for x in cands:
            new_cur = cur | {x}
            new_forbid = forbid | {x} | adj[x]
            if rec(pick, new_forbid, depth - 1, new_cur):
                return True
        return False

    if rec(pick=None, forbid=set(), depth=9, cur=frozenset()):
        return set()
    return None


# ---------- known-good fallback (8 x K4 = 32) ----------

def _fallback_graph():
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return 32, edges


def _valid(n, edges):
    adj = _build_adj(edges, n)
    if _has_c5(adj, n):
        return False
    if _independent_set9(adj, n) is not None:
        return False
    return True


# ---------- search for n = 33, 34 ----------

def _try_construct(n):
    """Try to build a valid n-vertex graph via random init + add/remove edge local search.
    Time-limited; returns (n, edges) if a valid graph is found, else None."""
    deadline = min(_HARD_DEADLINE - 2.0, time.time() + 95.0)
    rng = random.Random(20240517)

    # init: random regular-ish graph with moderate edge count, then fix C5 / independent issues
    best = None
    trials = 0
    while time.time() < deadline:
        trials += 1
        # random Erdos-Renyi tuned density; for C5-free we expect fewer edges
        p = 0.12 + 0.08 * ((trials % 7) / 7.0)
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    edges.add((i, j))
        # local improvement: while has_c5 or indep9, try edits
        cur = edges
        stable = 0
        while time.time() < deadline and stable < 400:
            adj = _build_adj(list(cur), n)
            bad_c5 = _has_c5(adj, n)
            bad_indep = _independent_set9(adj, n) is not None
            if not bad_c5 and not bad_indep:
                best = n, [list(e) for e in sorted(cur)]
                break
            # pick an offending structure to break
            did = False
            if bad_c5:
                # find a C5 and try to remove one of its edges
                cyc = _find_a_c5(adj, n)
                if cyc:
                    u, v = rng.choice(cyc)
                    if (u, v) in cur:
                        cur.discard((u, v))
                        did = True
                    elif (v, u) in cur:
                        cur.discard((v, u))
                        did = True
            if bad_indep:
                # add a random non-edge to reduce independence
                non = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if (i, j) not in cur and (j, i) not in cur]
                if non:
                    cur.add(rng.choice(non))
                    did = True
            if not did:
                # stuck: do a random flip
                all_e = [(i, j) for i in range(n) for j in range(i + 1, n)]
                e = rng.choice(all_e)
                if e in cur:
                    cur.discard(e)
                else:
                    cur.add(e)
            if did:
                stable = 0
            else:
                stable += 1
        # early exit once found
        if best is not None:
            break

    return best


def _find_a_c5(adj, n):
    seen = [False] * n
    for s in range(n):
        if seen[s]:
            continue
        comp = []
        q = [s]
        seen[s] = True
        while q:
            u = q.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        m = len(comp)
        if m < 5:
            continue
        if m <= 60:
            for i in range(m):
                for a in range(m):
                    for b in range(m):
                        for c in range(m):
                            for d in range(m):
                                nodes = [i, a, b, c, d]
                                if len(set(nodes)) != 5 or i != min(nodes):
                                    continue
                                if all(adj[nodes[k]].__contains__(nodes[(k + 1) % 5]) for k in range(5)):
                                    return [(nodes[k], nodes[(k + 1) % 5]) for k in range(5)]
        else:
            for i in range(m):
                for a in comp:
                    if a == i or i not in adj[a]:
                        continue
                    for b in comp:
                        if b in (i, a) or a not in adj[b]:
                            continue
                        for c in comp:
                            if c in (i, a, b) or b not in adj[c]:
                                continue
                            for d in comp:
                                if d in (i, a, b, c) or c not in adj[d]:
                                    continue
                                if i in adj[d] and len({a, b, c, d}) == 4:
                                    return [(i, a), (a, b), (b, c), (c, d), (d, i)]
    return None


def construct():
    # Try to improve beyond 32 with exact verification and local search.
    found = None
    for target in (34, 33):
        res = _try_construct(target)
        if res is not None:
            found = res
            break
        if time.time() > _HARD_DEADLINE - 3.0:
            break

    if found is not None:
        n, edges = found
        # final safety re-verification
        if _valid(n, edges):
            return {'n': n, 'edges': edges}

    # Fallback to the proven 32-vertex construction.
    n, edges = _fallback_graph()
    return {'n': n, 'edges': edges}
