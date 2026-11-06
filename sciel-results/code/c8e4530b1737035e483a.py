"""Strong edge colouring via vertex-colouring the strong conflict graph."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 110.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        m = len(edges)
        # conflict[i] = set of j != i such that edge i and edge j conflict
        # conflict if share a vertex OR are joined by an edge (endpoints adjacent)
        # For each edge i=(a,b): conflicting edges are those incident to a or b,
        # plus edges (x,y) where x in N(a) or x in N(b) ... actually joined by an edge
        # means there exists an edge of G between an endpoint of i and an endpoint of j.
        # i.e. (a,c) in G or (b,c) in G or (a,d) in G or (b,d) in G where j=(c,d).
        # So j conflicts with i if any endpoint of j is in {a,b} U N(a) U N(b).
        conflict = [set() for _ in range(m)]
        # index edges by their endpoints
        for i, (a, b) in enumerate(edges):
            forbidden = {a, b}
            forbidden.update(adj[a])
            forbidden.update(adj[b])
            for j, (c, d) in enumerate(edges):
                if j == i:
                    continue
                if c in forbidden or d in forbidden:
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def dsatur(conflict, m, order=None):
        # returns list of colours (0..k-1)
        if order is None:
            order = list(range(m))
        colour = [-1] * m
        sat = [set() for _ in range(m)]
        deg = [len(conflict[i]) for i in range(m)]
        remaining = set(order)
        # priority: (saturation, degree)
        import heapq
        # We'll just do a simple greedy in given order but choose colour by least used / first fit
        for i in order:
            # pick smallest colour not in sat[i]
            c = 0
            while c in sat[i]:
                c += 1
            colour[i] = c
            for j in conflict[i]:
                if colour[j] == -1:
                    sat[j].add(c)
        return colour

    def improve_kempe(colour, conflict, m, max_iter=20000):
        # Try to reduce number of colours by swapping colour classes (Kempe)
        # and by recoloring. Simple approach: repeatedly try to reduce a vertex's
        # colour to a lower available one, and try Kempe chain merges.
        k = max(colour) + 1 if m else 0
        if k == 0:
            return colour, 0
        # colour classes
        classes = defaultdict(list)
        for i in range(m):
            classes[colour[i]].append(i)
        improved = 0
        for _ in range(max_iter):
            # Try: for each colour c from k-1 down, see if class c can be merged into lower colours
            # via Kempe. Simpler: try to recolor one vertex of highest colour class to a free lower colour.
            # Find a vertex with the highest colour that has a free lower colour.
            # Build sat sets
            sat = [set() for _ in range(m)]
            for i in range(m):
                for j in conflict[i]:
                    sat[i].add(colour[j])
            # pick vertex with max colour
            best_i = None
            best_c = -1
            for i in range(m):
                if colour[i] > best_c:
                    best_c = colour[i]
                    best_i = i
            if best_i is None:
                break
            # try to find a free colour < best_c
            free = None
            for c in range(best_c):
                if c not in sat[best_i]:
                    free = c
                    break
            if free is not None:
                # recolor
                colour[best_i] = free
                improved += 1
                continue
            # Kempe: try to free a colour by swapping along a chain.
            # Pick a target colour t < best_c. We want to recolor best_i to t.
            # If t is used by a neighbour, do a Kempe swap of t and best_c in the
            # connected component of best_i in the subgraph induced by colours {t, best_c}.
            t = None
            for c in range(best_c):
                if c in sat[best_i]:
                    t = c
                    break
            if t is None:
                break
            # BFS in subgraph of vertices coloured t or best_c starting from best_i
            # (best_i is best_c). We want to flip t<->best_c in that component
            # such that after flip, best_i becomes t and its neighbours are not t.
            seen = {best_i}
            stack = [best_i]
            while stack:
                u = stack.pop()
                for v in conflict[u]:
                    if v in seen:
                        continue
                    if colour[v] in (t, best_c):
                        seen.add(v)
                        stack.append(v)
            # Check: is there a vertex in seen (other than best_i) that is a neighbour of best_i and coloured t?
            # If not, flipping is safe.
            ok = True
            for v in conflict[best_i]:
                if v in seen and colour[v] == t:
                    ok = False
                    break
            if ok:
                for v in seen:
                    if colour[v] == t:
                        colour[v] = best_c
                    elif colour[v] == best_c:
                        colour[v] = t
                improved += 1
            else:
                # try another t
                done = False
                for c in range(best_c):
                    if c == t:
                        continue
                    if c not in sat[best_i]:
                        colour[best_i] = c
                        improved += 1
                        done = True
                        break
                    # try kempe with c
                    t2 = c
                    seen2 = {best_i}
                    stack2 = [best_i]
                    while stack2:
                        u = stack2.pop()
                        for v in conflict[u]:
                            if v in seen2:
                                continue
                            if colour[v] in (t2, best_c):
                                seen2.add(v)
                                stack2.append(v)
                    ok2 = True
                    for v in conflict[best_i]:
                        if v in seen2 and colour[v] == t2:
                            ok2 = False
                            break
                    if ok2:
                        for v in seen2:
                            if colour[v] == t2:
                                colour[v] = best_c
                            elif colour[v] == best_c:
                                colour[v] = t2
                        improved += 1
                        done = True
                        break
                if not done:
                    break
        k = max(colour) + 1 if m else 0
        return colour, k

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        m = len(edges)
        if m == 0:
            out.append([])
            continue
        conflict = build_conflict(edges, spec[0])
        best_colour = None
        best_k = m + 1

        # multiple orderings
        orderings = []
        # degree order (high first)
        orderings.append(sorted(range(m), key=lambda i: -len(conflict[i])))
        # a few random shuffles
        rng = random.Random(0)
        for _ in range(4):
            o = list(range(m))
            rng.shuffle(o)
            orderings.append(o)
        # least-saturation dynamic: we approximate by degree then random

        for o in orderings:
            if time.time() > deadline - 5:
                break
            c = dsatur(conflict, m, o)
            k = max(c) + 1
            # improve
            c, k = improve_kempe(c, conflict, m, max_iter=5000)
            if k < best_k:
                best_k = k
                best_colour = c

        out.append(best_colour)
    return out
# EVOLVE-BLOCK-END
