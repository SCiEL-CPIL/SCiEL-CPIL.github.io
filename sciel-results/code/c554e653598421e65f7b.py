import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s limit

    n = 40
    target_edges = 401
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_set = set(all_pairs)

    # --- Asymmetric structural seed: K9,31 core + 122 triangle-free inner edges ---
    A = list(range(9))
    B = list(range(9, 40))
    needed_inner = 401 - 9 * 31  # = 122

    def build_inner(seed):
        random.seed(seed)
        inner = set()
        adj = {v: set() for v in B}
        cands = [(u, v) for u in B for v in B if u < v]
        random.shuffle(cands)
        for u, v in cands:
            if len(inner) >= needed_inner:
                break
            if not (adj[u] & adj[v]):
                inner.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
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

    def get_max(books):
        if not books:
            return 0, 0
        mx = max(books.values())
        cnt = sum(1 for s in books.values() if s == mx)
        return mx, cnt

    def polish(edges, adj, books, t_end):
        """Final high-precision polish: exhaustive scan for best swap reducing max-book."""
        while time.time() < t_end:
            mv, mc = get_max(books)
            if mv <= 6:
                break
            max_edges = [e for e, s in books.items() if s == mv]
            best_swap = None
            best_key = None
            for (u, v) in max_edges:
                # scan all non-edges to find best addition
                for (x, y) in all_pairs:
                    if (x, y) in edges:
                        continue
                    b = (adj[x] & adj[y]).bit_count()
                    # after swap: drop (u,v), add (x,y)
                    # new max is roughly max(mv - (1 if (u,v) was unique max else 0), b, other affected)
                    # conservative: key by (b, b > 0)
                    key = (b, b > 0)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_swap = ((u, v), (x, y))
                    if b == 0:
                        break
            if best_swap is None:
                break
            (u, v), (x, y) = best_swap
            # apply swap
            remove_edge(edges, adj, u, v)
            add_edge(edges, adj, x, y)
            # update books for touched vertices
            touched = set()
            for vv in (u, v, x, y):
                nb = adj[vv]
                w = 0
                while nb:
                    if nb & 1:
                        a, b = (vv, w) if vv < w else (w, vv)
                        touched.add((a, b))
                    w += 1
                    nb >>= 1
            for e in touched:
                if e in edges:
                    a, b = e
                    books[e] = (adj[a] & adj[b]).bit_count()
                else:
                    books.pop(e, None)
            books[(x, y)] = (adj[x] & adj[y]).bit_count()
            nmv, nmc = get_max(books)
            if (nmv, nmc) > (mv, mc):
                # revert
                remove_edge(edges, adj, x, y)
                add_edge(edges, adj, u, v)
                for e in touched:
                    if e in edges:
                        a, b = e
                        books[e] = (adj[a] & adj[b]).bit_count()
                books[(u, v)] = (adj[u] & adj[v]).bit_count()
                break
        return edges

    def refine(edges, adj, books, mx, cnt, t0, t_end):
        """Local search with restarts, accepting equal-max moves to escape plateaus."""
        best = (mx, cnt)
        best_edges = set(edges)
        no_improve = 0

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

            # pick a max-book edge to drop
            max_edges = [e for e, s in books.items() if s == mv]
            if not max_edges:
                break
            u, v = random.choice(max_edges)

            # find best non-edge to add
            best_add = None
            best_key = None
            for _ in range(80):
                x, y = random.choice(all_pairs)
                if (x, y) in edges:
                    continue
                b = (adj[x] & adj[y]).bit_count()
                key = (b, b > 0)
                if best_key is None or key < best_key:
                    best_key = key
                    best_add = (x, y)
                    if b == 0:
                        break
            if best_add is None:
                break
            x, y = best_add

            # apply swap
            remove_edge(edges, adj, u, v)
            add_edge(edges, adj, x, y)
            # update books for touched vertices
            touched = set()
            for vv in (u, v, x, y):
                nb = adj[vv]
                w = 0
                while nb:
                    if nb & 1:
                        a, b = (vv, w) if vv < w else (w, vv)
                        touched.add((a, b))
                    w += 1
                    nb >>= 1
            for e in touched:
                if e in edges:
                    a, b = e
                    books[e] = (adj[a] & adj[b]).bit_count()
                else:
                    books.pop(e, None)
            books[(x, y)] = (adj[x] & adj[y]).bit_count()

            nmv, nmc = get_max(books)
            # Accept if improvement OR equal max with fewer count, OR equal (to escape plateaus)
            if (nmv, nmc) <= (mv, mc):
                no_improve = 0
            else:
                # revert
                remove_edge(edges, adj, x, y)
                add_edge(edges, adj, u, v)
                for e in touched:
                    if e in edges:
                        a, b = e
                        books[e] = (adj[a] & adj[b]).bit_count()
                books[(u, v)] = (adj[u] & adj[v]).bit_count()
                no_improve += 1

            if no_improve > 200:
                if time.time() > t_end:
                    break
                inner = build_inner(random.randint(0, 10**6))
                edges, adj, books, mx, cnt = make_state(inner)
                no_improve = 0
                if (mx, cnt) < best:
                    best = (mx, cnt)
                    best_edges = set(edges)

        # final polish on best
        if time.time() < t_end - 1:
            edges = polish(best_edges, None, None, min(t_end, time.time() + 5))
        return best, best_edges

    # --- Multi-start over inner-graph seeds ---
    best = (10**9, 10**9)
    best_edges = None

    seeds = [42, 123, 999, 555, 777, 111, 222, 333, 444, 666, 101, 202, 303, 404, 505]
    scored = []
    for s in seeds:
        edges, adj, books, mx, cnt = make_state(build_inner(s))
        scored.append((mx, cnt, s))
    scored.sort()

    for mx, cnt, s in scored:
        if time.time() > deadline - 10:
            break
        edges, adj, books, m0, c0 = make_state(build_inner(s))
        t_end = min(deadline - 5, start_time + (35 if (mx, cnt) <= (8, 5) else 18))
        res, res_edges = refine(edges, adj, books, m0, c0, time.time(), t_end)
        if res < best:
            best = res
            best_edges = res_edges
        if best[0] <= 6:
            break

    # Final polish pass on the best solution
    if best_edges is not None and time.time() < deadline - 2:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        books = {}
        for u, v in best_edges:
            books[(u, v)] = (adj[u] & adj[v]).bit_count()
        polished = polish(set(best_edges), adj, books, deadline - 1)
        final_books = {}
        for u, v in polished:
            final_books[(u, v)] = (adj[u] & adj[v]).bit_count()
        fmx, fcnt = get_max(final_books)
        if (fmx, fcnt) < best:
            best = (fmx, fcnt)
            best_edges = polished

    if best_edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    return [[u, v] for u, v in best_edges]
