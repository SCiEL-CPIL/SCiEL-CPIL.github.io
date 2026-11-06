import time
import random
import heapq
import numpy as np

def construct():
    N = 40
    M = 401
    deadline = time.time() + 120
    safety = 10.0  # return at least this many seconds before the hard limit

    def time_left():
        return deadline - time.time() - safety

    def build_adj(edge_list):
        adj = np.zeros((N, N), dtype=np.int8)
        for u, v in edge_list:
            adj[u, v] = 1
            adj[v, u] = 1
        np.fill_diagonal(adj, 0)
        return adj

    def edge_set(adj):
        return {(i, j) for i in range(N) for j in range(i + 1, N) if adj[i, j]}

    def compute_books(adj, edge_set):
        # book[(u,v)] = number of common neighbors of u and v
        books = {}
        for (u, v) in edge_set:
            books[(u, v)] = int(np.dot(adj[u], adj[v]))
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def neighbors_of(adj, v):
        return np.where(adj[v] == 1)[0]

    def evaluate_add(adj, edge_set, books, u, v):
        # add edge (u,v); return new max book and the new books dict
        # common neighbors of u and v in base graph
        common = np.logical_and(adj[u], adj[v])
        common_idx = np.where(common)[0]
        b_new = len(common_idx)
        new_books = {}
        for e in edge_set:
            new_books[e] = books[e]
        new_books[(min(u, v), max(u, v))] = b_new
        for a in common_idx:
            e1 = (min(u, a), max(u, a))
            e2 = (min(v, a), max(v, a))
            new_books[e1] = books[e1] + 1
            new_books[e2] = books[e2] + 1
        return max_book(new_books), new_books

    def evaluate_swap_out_in(adj, edge_set, books, out_edge, in_edge):
        # remove out_edge, add in_edge; return new max book and new books dict
        u, v = out_edge
        a, b = in_edge
        # common neighbors of a,b in graph WITHOUT the out edge
        # base common:
        common = np.logical_and(adj[a], adj[b])
        # if out edge (u,v) is a common neighbor path? out edge doesn't affect adjacency of a,b
        # but common neighbors are vertices w with adj[a][w] and adj[b][w]; out edge removal
        # only changes adjacency of u and v.
        # So common is unchanged by removal of out edge, EXCEPT if w==u or w==v and
        # out edge was (u,?) making u a neighbor of v... out edge is (u,v).
        # Removing (u,v): adj[u][v]=0, adj[v][u]=0.
        # common neighbors of a,b: w such that adj[a][w] and adj[b][w].
        # If w==u: needs adj[a][u] and adj[b][u]. adj[b][u] is affected only if out edge is (b,u) i.e. b==v and u is the other endpoint.
        # This is getting complicated; handle by directly recomputing on a modified adjacency.
        adj2 = adj.copy()
        adj2[u, v] = 0
        adj2[v, u] = 0
        common = np.logical_and(adj2[a], adj2[b])
        common_idx = np.where(common)[0]
        b_new = len(common_idx)

        # Build new edge set
        new_edge_set = set(edge_set)
        new_edge_set.discard(out_edge)
        new_edge_set.add((min(a, b), max(a, b)))

        new_books = {}
        for e in new_edge_set:
            if e == (min(a, b), max(a, b)):
                new_books[e] = b_new
            else:
                # recompute book on adj2 for safety
                x, y = e
                new_books[e] = int(np.dot(adj2[x], adj2[y]))
        return max_book(new_books), new_books, adj2

    # ---------- Build base 400-edge 20-regular graph ----------
    # Use circulant with offset set S of size 10 chosen to minimize max |S ∩ (S-d)|
    all_offsets = list(range(1, 20))
    best_S = None
    best_circ_max = 10**9
    # enumerate all 10-subsets of 19 (92378)
    from itertools import combinations
    for S in combinations(all_offsets, 10):
        S_set = set(S)
        mx = 0
        for d in S:
            cnt = 0
            for s in S:
                if (s - d) % N in S_set:
                    cnt += 1
            if cnt > mx:
                mx = cnt
        if mx < best_circ_max:
            best_circ_max = mx
            best_S = S
            if mx <= 6:
                break
    S = best_S
    S_set = set(S)
    base_edges = []
    for i in range(N):
        for s in S:
            j1 = (i + s) % N
            if j1 > i:
                base_edges.append((i, j1))
    # dedupe
    base_edges = sorted(set(base_edges))
    assert len(base_edges) == 400, len(base_edges)

    adj = build_adj(base_edges)
    es = edge_set(adj)
    books = compute_books(adj, es)
    maxb = max_book(books)
    best_maxb = maxb
    best_edges_list = base_edges[:]

    # ---------- Add the 401st edge that minimizes max book ----------
    non_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i, j]]
    best_add = None
    best_add_max = maxb + 10**9
    for (u, v) in non_edges:
        nm, _ = evaluate_add(adj, es, books, u, v)
        if nm < best_add_max:
            best_add_max = nm
            best_add = (u, v)
    u, v = best_add
    base_edges.append((u, v))
    adj = build_adj(base_edges)
    es = edge_set(adj)
    books = compute_books(adj, es)
    maxb = max_book(books)
    best_maxb = maxb
    best_edges_list = base_edges[:]

    # ---------- Local search: swap high-book edges for low-book non-edges ----------
    # Priority queue of (new_max_estimate, out_edge) for edges with high book
    def recompute_all():
        nonlocal adj, es, books, maxb
        es = edge_set(adj)
        books = compute_books(adj, es)
        maxb = max_book(books)

    # candidate non-edges: those that, when added, don't immediately blow up the max
    # We'll maintain a heap of "promising" out-edges keyed by their current book size.
    # For each out-edge with large book, try swapping with a few non-edges.
    # Use a simple iterative improvement loop with a deadline guard.

    def candidate_non_edges(adj, es):
        return [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i, j]]

    improved = True
    while improved and time_left() > 1.0:
        improved = False
        # find out-edges sorted by descending book
        out_candidates = sorted(es, key=lambda e: -books[e])
        # only consider the top ones (high book)
        top_out = out_candidates[:40]
        non_edges = candidate_non_edges(adj, es)
        # For speed, limit non-edge trials per out-edge
        # Try to find a swap that reduces the global max book
        best_swap = None
        best_swap_max = maxb
        for out_e in top_out:
            if time_left() < 1.0:
                break
            # For this out edge, try a subset of non-edges
            # Prefer non-edges that are "far" (low common neighbor count) to keep new book small
            # Compute b_new quickly for each non-edge (common neighbors)
            # Limit to first K non-edges sorted by b_new ascending
            scored = []
            for (a, b) in non_edges:
                if a == out_e[0] or a == out_e[1] or b == out_e[0] or b == out_e[1]:
                    # avoid trivial re-add of same edge
                    if (a, b) == out_e:
                        continue
                bnew = int(np.dot(adj[a], adj[b]))
                scored.append((bnew, (a, b)))
            scored.sort(key=lambda t: t[0])
            # try the lowest-bnew non-edges
            for (bnew, in_e) in scored[:25]:
                if time_left() < 1.0:
                    break
                if in_e == out_e:
                    continue
                nm, nb, adj2 = evaluate_swap_out_in(adj, es, books, out_e, in_e)
                if nm < best_swap_max:
                    best_swap_max = nm
                    best_swap = (out_e, in_e, nb, adj2)
        if best_swap is not None:
            out_e, in_e, nb, adj2 = best_swap
            adj = adj2
            es = edge_set(adj)
            books = nb
            maxb = max_book(books)
            improved = True
            if maxb < best_maxb:
                best_maxb = maxb
                best_edges_list = sorted(es)
            # early stop at theoretical lower bound
            if best_maxb <= 7:
                break

    # Final: return the best edges list found
    # Reconstruct a valid edge list from best_edges_list
    result = [list(e) for e in best_edges_list]
    # Ensure exactly M distinct edges
    # dedupe
    seen = set()
    deduped = []
    for e in result:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            seen.add(key)
            deduped.append([key[0], key[1]])
    if len(deduped) != M:
        # fallback to a valid random-ish graph if something went wrong
        # Rebuild from base + best_add
        fallback = [list(e) for e in base_edges]
        if len(fallback) != M:
            # last resort random
            pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            random.shuffle(pairs)
            fallback = [list(p) for p in pairs[:M]]
        return fallback
    return deduped
