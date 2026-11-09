import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s limit

    n = 40
    target_edges = 401
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # --- Asymmetric structural seed: K9,31 core + 122 triangle-free inner edges ---
    A = list(range(9))
    B = list(range(9, 40))
    needed_inner = 401 - 9 * 31  # = 122

    def build_inner(seed):
        random.seed(seed)
        inner = set()
        adj = {v: set() for v in B}
        # Phase 1: add edges with no common neighbor inside B (keeps inner triangle-free)
        cands = [(u, v) for u in B for v in B if u < v]
        random.shuffle(cands)
        for u, v in cands:
            if len(inner) >= needed_inner:
                break
            if not (adj[u] & adj[v]):
                inner.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        # Phase 2: fill remainder, preferring edges with fewest common neighbors
        if len(inner) < needed_inner:
            remaining = [(u, v) for (u, v) in cands if (u, v) not in inner]
            remaining.sort(key=lambda e: len(adj[e[0]] & adj[e[1]]))
            for u, v in remaining:
                if len(inner) >= needed_inner:
                    break
                inner.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        return inner

    def make_state(inner):
        edges = set()
        for a in A:
            for b in B:
                edges.add((a, b))
        for e in inner:
            edges.add(e)
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        books = {}
        for u, v in edges:
            books[(u, v)] = (adj[u] & adj[v]).bit_count()
        mx = max(books.values())
        cnt = sum(1 for s in books.values() if s == mx)
        return edges, adj, books, mx, cnt

    # --- Search machinery ---
    def add_edge(edges, adj, u, v):
        if u > v:
            u, v = v, u
        edges.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(edges, adj, u, v):
        if u > v:
            u, v = v, u
        edges.discard((u, v))
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def recompute(books, adj, edges):
        for (u, v), s in list(books.items()):
            if (u, v) not in edges:
                del books[(u, v)]
        for (u, v) in edges:
            books[(u, v)] = (adj[u] & adj[v]).bit_count()

    def get_max(books):
        if not books:
            return 0, 0
        mx = max(books.values())
        cnt = sum(1 for s in books.values() if s == mx)
        return mx, cnt

    def refine(edges, adj, books, mx, cnt, t0, t_end):
        """Lexicographic (max-book, count-of-max) local search with restarts."""
        best = (mx, cnt)
        best_edges = set(edges)
        no_improve = 0

        def swap_pick():
            # choose a max-book edge to drop
            max_edges = [e for e, s in books.items() if s == mx]
            if not max_edges:
                return None
            u, v = random.choice(max_edges)
            # try non-edges, score by (new book size, whether it creates triangles)
            best_key = None
            best_add = None
            # deterministic full scan is cheap (n=40); sample many non-edges
            for _ in range(60):
                x, y = random.choice(all_pairs)
                if x > y:
                    x, y = y, x
                if (x, y) in edges:
                    continue
                b = (adj[x] & adj[y]).bit_count()
                key = (b, b > 0)
                if best_key is None or key < best_key:
                    best_key = key
                    best_add = (x, y)
                    if b == 0:
                        break
            return (u, v), best_add

        while time.time() < t_end:
            if best[0] <= 6:
                break
            mv, mc = get_max(books)
            if (mv, mc) < best:
                best = (mv, mc)
                best_edges = set(edges)
                no_improve = 0
            else:
                no_improve += 1
            pick = swap_pick()
            if pick is None:
                break
            (u, v), add = pick
            if add is None:
                break
            x, y = add
            remove_edge(edges, adj, u, v)
            add_edge(edges, adj, x, y)
            # update books for the four touched vertices
            for vv in (u, v, x, y):
                for w in range(n):
                    if (adj[vv] >> w) & 1:
                        a, b = (vv, w) if vv < w else (w, vv)
                        books[(a, b)] = (adj[a] & adj[b]).bit_count()
            nmv, nmc = get_max(books)
            if (nmv, nmc) <= (mv, mc):
                mx, cnt = nmv, nmc
                no_improve = 0
            else:
                # revert
                remove_edge(edges, adj, x, y)
                add_edge(edges, adj, u, v)
                for vv in (u, v, x, y):
                    for w in range(n):
                        if (adj[vv] >> w) & 1:
                            a, b = (vv, w) if vv < w else (w, vv)
                            books[(a, b)] = (adj[a] & adj[b]).bit_count()
                no_improve += 1
            if no_improve > 400:
                # restart from best with fresh inner graph
                if time.time() > t_end:
                    break
                inner = build_inner(random.randint(0, 10**6))
                edges, adj, books, mx, cnt = make_state(inner)
                no_improve = 0
                if (mx, cnt) < best:
                    best = (mx, cnt)
                    best_edges = set(edges)
        return best, best_edges

    # --- Multi-start over inner-graph seeds, refine each with a time slice ---
    best = (10**9, 10**9)
    best_edges = None

    seeds = [42, 123, 999, 555, 777, 111, 222, 333, 444, 666, 101, 202, 303, 404, 505]
    # quick eval of each seed's raw score first to order them
    scored = []
    for s in seeds:
        edges, adj, books, mx, cnt = make_state(build_inner(s))
        scored.append((mx, cnt, s))
    scored.sort()

    # give more time to better seeds; cap total within deadline
    for mx, cnt, s in scored:
        if time.time() > deadline - 8:
            break
        edges, adj, books, m0, c0 = make_state(build_inner(s))
        t_end = min(deadline - 3, start_time + (30 if (mx, cnt) <= (7, 5) else 14))
        res, res_edges = refine(edges, adj, books, m0, c0, time.time(), t_end)
        if res < best:
            best = res
            best_edges = res_edges
        if best[0] <= 6:
            break

    if best_edges is None:
        # fallback: K20,20 + 1
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    return [[u, v] for u, v in best_edges]
