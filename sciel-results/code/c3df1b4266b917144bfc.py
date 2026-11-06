"""Clique partition via maximal-clique pool + greedy set cover + merge improvement."""
import time
import random
import heapq

_DEADLINE = 140.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _random_graph(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _validate(n, edges, cliques):
    edge_set = set(edges)
    seen = set()
    for c in cliques:
        if len(c) < 2:
            return False
        k = len(c)
        for i in range(k):
            a = c[i]
            for j in range(i + 1, k):
                b = c[j]
                e = (a, b) if a < b else (b, a)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edge_set


def _bron_kerbosch_maximal_cliques(adj, n):
    """Enumerate all maximal cliques using Bron–Kerbosch with pivot."""
    cliques = []
    # adj is list of int bitmasks

    def bk(R, P, X):
        if _elapsed() > _DEADLINE:
            return
        if not P and not X:
            cliques.append(R)
            return
        # Pick pivot u from P ∪ X maximizing |P ∩ N(u)|
        px = P | X
        if px:
            # choose pivot with max common neighbors in P
            best_u = -1
            best_cnt = -1
            # iterate bits of px
            temp = px
            while temp:
                lb = temp & (-temp)
                u = lb.bit_length() - 1
                cnt = (P & adj[u]).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_u = u
                temp ^= lb
            # candidates = P \ N(pivot)
            cand = P & ~adj[best_u]
        else:
            cand = P
        temp = cand
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            temp ^= lb
            new_R = R | (1 << v)
            new_P = P & adj[v]
            new_X = X & adj[v]
            bk(new_R, new_P, new_X)
            P &= ~(1 << v)
            X |= (1 << v)
            if _elapsed() > _DEADLINE:
                return

    bk(0, (1 << n) - 1, 0)
    return cliques


def _solve_instance(n, p, seed):
    edges = _random_graph(n, p, seed)
    edge_set = set(edges)
    if not edges:
        return []

    # Adjacency bitmasks
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Enumerate all maximal cliques (size >= 2)
    all_max_cliques = _bron_kerbosch_maximal_cliques(adj, n)

    # Build a pool of candidate cliques: all maximal cliques of size >= 2
    # Also include all triangles and all edges as fallback
    candidate_cliques = []
    seen_clique = set()
    for mc in all_max_cliques:
        # mc is a bitmask
        verts = []
        temp = mc
        while temp:
            lb = temp & (-temp)
            verts.append(lb.bit_length() - 1)
            temp ^= lb
        if len(verts) >= 2:
            key = tuple(sorted(verts))
            if key not in seen_clique:
                seen_clique.add(key)
                candidate_cliques.append(key)

    # If enumeration was cut short or found nothing, fall back to triangles + edges
    if not candidate_cliques:
        # Find all triangles
        for (u, v) in edges:
            if u > v:
                u, v = v, u
            cand = adj[u] & adj[v]
            cand &= ~((1 << (v + 1)) - 1)
            temp = cand
            while temp:
                lb = temp & (-temp)
                w = lb.bit_length() - 1
                temp ^= lb
                key = tuple(sorted([u, v, w]))
                if key not in seen_clique:
                    seen_clique.add(key)
                    candidate_cliques.append(key)
        # Add all edges
        for (u, v) in edges:
            key = tuple(sorted([u, v]))
            if key not in seen_clique:
                seen_clique.add(key)
                candidate_cliques.append(key)

    # Sort candidates by "efficiency": larger cliques first (cover more edges per clique)
    # Use a heuristic: prefer cliques that cover many currently-uncovered edges
    # For initial sort, just use size descending
    candidate_cliques.sort(key=lambda c: -len(c))

    # Greedy set cover: repeatedly pick the clique covering the most uncovered edges
    uncovered = set(edge_set)
    cliques = []
    # Map from clique tuple to its edge set
    clique_edges_cache = {}
    for c in candidate_cliques:
        es = set()
        k = len(c)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = c[i], c[j]
                es.add((a, b) if a < b else (b, a))
        clique_edges_cache[c] = es

    # For speed, precompute edge counts
    # Greedy: pick clique maximizing |clique_edges ∩ uncovered|
    # To speed up, use a heap or just iterate (n is small)
    while uncovered and _elapsed() < _DEADLINE:
        best_c = None
        best_count = 0
        for c in candidate_cliques:
            ce = clique_edges_cache[c]
            cnt = len(ce & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_c = c
            elif cnt == best_count and best_c is not None:
                # tie-break: prefer larger clique
                if len(c) > len(best_c):
                    best_c = c
            if _elapsed() > _DEADLINE:
                break
        if best_c is None or best_count == 0:
            break
        cliques.append(list(best_c))
        uncovered -= clique_edges_cache[best_c]

    # Remaining uncovered edges become 2-cliques
    for e in uncovered:
        cliques.append([e[0], e[1]])

    # Local improvement: try to merge pairs of 2-cliques into a triangle
    # or merge a 2-clique into an existing larger clique if possible
    # This is a post-processing pass
    if _elapsed() < _DEADLINE - 5:
        # Build edge -> clique index map
        edge_to_clique = {}
        for idx, c in enumerate(cliques):
            k = len(c)
            for i in range(k):
                for j in range(i + 1, k):
                    a, b = c[i], c[j]
                    e = (a, b) if a < b else (b, a)
                    edge_to_clique[e] = idx

        # Try to merge two 2-cliques sharing a vertex into a triangle
        # Find all 2-cliques
        two_cliques = [i for i, c in enumerate(cliques) if len(c) == 2]
        # Group 2-cliques by their vertices
        vertex_to_two = {}
        for i in two_cliques:
            u, v = cliques[i]
            vertex_to_two.setdefault(u, []).append(i)
            vertex_to_two.setdefault(v, []).append(i)

        merged = set()
        for u in list(vertex_to_two.keys()):
            for i in vertex_to_two[u]:
                if i in merged:
                    continue
                u1, v1 = cliques[i]
                # Find another 2-clique at u1 that could form a triangle
                for j in vertex_to_two.get(u1, []):
                    if j == i or j in merged:
                        continue
                    u2, v2 = cliques[j]
                    # The shared vertex is u1; the other endpoints are v1 and (u2 or v2)
                    other1 = v1
                    other2 = v2 if u2 == u1 else u2
                    # Check if other1-other2 is an edge
                    e_check = (other1, other2) if other1 < other2 else (other2, other1)
                    if e_check in edge_set:
                        # Form triangle {u1, other1, other2}
                        new_c = tuple(sorted([u1, other1, other2]))
                        # Replace cliques i and j with the triangle
                        cliques[i] = list(new_c)
                        cliques[j] = None  # mark for removal
                        merged.add(i)
                        merged.add(j)
                        break
                if i in merged:
                    continue
                for j in vertex_to_two.get(v1, []):
                    if j == i or j in merged:
                        continue
                    u2, v2 = cliques[j]
                    other1 = u1
                    other2 = u2 if v2 == v1 else v2
                    e_check = (other1, other2) if other1 < other2 else (other2, other1)
                    if e_check in edge_set:
                        new_c = tuple(sorted([v1, other1, other2]))
                        cliques[i] = list(new_c)
                        cliques[j] = None
                        merged.add(i)
                        merged.add(j)
                        break

        # Remove None entries
        cliques = [c for c in cliques if c is not None]

    # Final validation and fallback
    if not _validate(n, edges, cliques):
        cliques = [[u, v] for (u, v) in edges]
    return cliques


def construct():
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
