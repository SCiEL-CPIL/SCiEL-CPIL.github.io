"""Vertex-cover formulation for meeting every maximal clique of size >= 2."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, n):
        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            pivot_candidates = P | X
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def max_matching(adj, n):
        """Edmonds' blossom algorithm. Returns list of matched vertex pairs."""
        match = [-1] * n
        parent = [0] * n
        base = list(range(n))
        used = [False] * n
        blossom = [False] * n
        q = []

        def lca(a, b):
            used_path = [False] * n
            while True:
                a = base[a]
                used_path[a] = True
                if match[a] == -1:
                    break
                a = parent[match[a]]
            while True:
                b = base[b]
                if used_path[b]:
                    return b
                b = parent[match[b]]

        def mark_path(v, b, children):
            while base[v] != b:
                blossom[base[v]] = True
                blossom[base[match[v]]] = True
                parent[v] = children
                children = match[v]
                v = parent[match[v]]

        def find_path(root):
            used = [False] * n
            parent = [0] * n
            q = [root]
            used[root] = True
            qi = 0
            while qi < len(q):
                v = q[qi]
                qi += 1
                for u in adj[v]:
                    if base[v] == base[u] or match[v] == u:
                        continue
                    if u == root or (match[u] != -1 and parent[match[u]] != -1):
                        curbase = lca(v, u)
                        blossom = [False] * n
                        mark_path(v, curbase, u)
                        mark_path(u, curbase, v)
                        for i in range(n):
                            if blossom[base[i]]:
                                base[i] = curbase
                                if not used[i]:
                                    used[i] = True
                                    q.append(i)
                    elif parent[u] == -1:
                        parent[u] = v
                        if match[u] == -1:
                            return u
                        used[match[u]] = True
                        q.append(match[u])
            return -1

        for s in range(n):
            if match[s] != -1:
                continue
            u = find_path(s)
            if u == -1:
                continue
            while u != -1:
                pv = parent[u]
                nv = match[pv]
                match[u] = pv
                match[pv] = u
                u = nv
        pairs = []
        for i in range(n):
            if match[i] != -1 and i < match[i]:
                pairs.append((i, match[i]))
        return pairs

    def matching_vc(adj, n, edges):
        """2-approx vertex cover from a maximum matching."""
        pairs = max_matching(adj, n)
        S = set()
        for u, v in pairs:
            S.add(u)
            S.add(v)
        return S

    def local_search(S, edges, n, time_budget, max_passes=200000):
        """Remove redundant vertices and do 2-for-1 swaps."""
        S = set(S)
        if not edges:
            return S
        eidx = [(u, v, i) for i, (u, v) in enumerate(edges)]
        last_change = time.time()

        def valid(T):
            for u, v, _ in eidx:
                if u not in T and v not in T:
                    return False
            return True

        passes = 0
        while time.time() - last_change < time_budget and passes < max_passes:
            passes += 1
            changed = False
            # removal pass
            for v in list(S):
                if valid(S - {v}):
                    S.discard(v)
                    changed = True
                    last_change = time.time()
            # 2-for-1 swap pass
            if time.time() - last_change >= time_budget:
                break
            for v in list(S):
                if time.time() - last_change >= time_budget:
                    break
                cand = None
                for w in range(n):
                    if w in S:
                        continue
                    T = (S - {v}) | {w}
                    if valid(T):
                        cand = w
                        break
                if cand is not None:
                    S.discard(v)
                    S.add(cand)
                    changed = True
                    last_change = time.time()
            if not changed:
                break
        return S

    def exact_vc_lp(n, edges, time_budget):
        """Exact minimum vertex cover via LP relaxation + branch and bound."""
        if not edges:
            return set()
        try:
            from scipy.optimize import linprog
        except Exception:
            return None

        m = len(edges)
        A = []
        for (u, v) in edges:
            row = [0.0] * n
            row[u] = 1.0
            row[v] = 1.0
            A.append(row)
        A = [A]
        c = [1.0] * n
        bounds = [(0, 1)] * n
        r = linprog(c, A_ub=A[0], b_ub=[1.0] * m, bounds=bounds, method="highs")
        if r is None or not r.success:
            return None
        x = r.x
        best = [None]
        best_val = [float("inf")]
        nodes = 0
        node_limit = 300000
        start = time.time()

        def branch(fixed):
            if time.time() - start > time_budget or nodes[0] > node_limit:
                return
            nodes[0] += 1
            A2 = []
            b2 = []
            for (u, v) in edges:
                if fixed.get(u) is True:
                    continue
                if fixed.get(v) is True:
                    continue
                row = [0.0] * n
                if fixed.get(u) is False:
                    row[v] = 1.0
                elif fixed.get(v) is False:
                    row[u] = 1.0
                else:
                    row[u] = 1.0
                    row[v] = 1.0
                A2.append(row)
                b2.append(1.0)
            if not A2:
                val = sum(1 for k, val in fixed.items() if val is True)
                if val < best_val[0]:
                    best_val[0] = val
                    best[0] = {k for k, val in fixed.items() if val is True}
                return
            c2 = [0.0] * n
            for k, val in fixed.items():
                if val is True:
                    c2[k] = 1.0
            bounds2 = []
            for k in range(n):
                if fixed.get(k) is True:
                    bounds2.append((1, 1))
                elif fixed.get(k) is False:
                    bounds2.append((0, 0))
                else:
                    bounds2.append((0, 1))
            r2 = linprog(c2, A_ub=A2, b_ub=b2, bounds=bounds2, method="highs")
            if r2 is None or not r2.success:
                return
            lb = r2.fun
            if lb >= best_val[0] - 1e-9:
                return
            x2 = r2.x
            frac = None
            best_frac = 0.49
            for k in range(n):
                if fixed.get(k) is None:
                    f = x2[k] - int(x2[k])
                    if 1e-6 < f < 1 - 1e-6:
                        if f > best_frac:
                            best_frac = f
                            frac = k
            if frac is None:
                val = int(round(lb))
                if val < best_val[0]:
                    best_val[0] = val
                    best[0] = {k for k in range(n) if x2[k] > 0.5}
                return
            fixed[frac] = True
            branch(fixed)
            fixed[frac] = False
            branch(fixed)
            fixed.pop(frac, None)

        branch({})
        return best[0]

    def solve_instance(n, p, seed, time_budget):
        edges = generate_graph(n, p, seed)
        adj = build_adjacency(n, edges)
        if not edges:
            return []
        # Fallback: matching 2-approx
        S = matching_vc(adj, n, edges)
        S = local_search(S, edges, n, 1.5)
        # Try exact via LP + branch and bound
        t0 = time.time()
        exact = exact_vc_lp(n, edges, max(0.0, time_budget - (time.time() - t0) - 2.0))
        if exact is not None and len(exact) <= len(S):
            S = set(exact)
            S = local_search(S, edges, n, max(0.5, time_budget - (time.time() - t0)))
        return sorted(S)

    results = []
    per_budget = 42.0
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 6:
            results.append([])
            continue
        budget = min(per_budget, max(4.0, remaining - 12.0))
        results.append(solve_instance(n, p, seed, budget))
    return results

# EVOLVE-BLOCK-END
