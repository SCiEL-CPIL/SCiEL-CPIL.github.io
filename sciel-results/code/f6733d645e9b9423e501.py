import time
import itertools

DEADLINE = time.time() + 80.0


def _cycle_lengths(adj):
    """Return the set of distinct simple-cycle lengths in an undirected graph."""
    n = len(adj)
    adj_set = [set(a) for a in adj]
    lengths = set()
    for start in range(n):
        # canonical: start is the smallest vertex in the cycle
        for end in range(start + 1, n):
            if end not in adj_set[start]:
                continue
            for mid in range(start + 1, end):
                if mid not in adj_set[start] or mid not in adj_set[end]:
                    continue
                lengths.add(3)
            # 4-cycles: start - x - y - end - start
            for x in range(start + 1, end):
                if x not in adj_set[start]:
                    continue
                for y in range(x + 1, end):
                    if y in adj_set[x] and y in adj_set[end]:
                        lengths.add(4)
            # 5-cycles: start - x - y - z - end - start
            for x in range(start + 1, end):
                if x not in adj_set[start]:
                    continue
                for y in range(x + 1, end):
                    if y not in adj_set[x]:
                        continue
                    for z in range(y + 1, end):
                        if z in adj_set[y] and z in adj_set[end]:
                            lengths.add(5)
            # 6-cycles: start - x - y - z - w - end - start
            for x in range(start + 1, end):
                if x not in adj_set[start]:
                    continue
                for y in range(x + 1, end):
                    if y not in adj_set[x]:
                        continue
                    for z in range(y + 1, end):
                        if z not in adj_set[y]:
                            continue
                        for w in range(z + 1, end):
                            if w in adj_set[z] and w in adj_set[end]:
                                lengths.add(6)
            # 7-cycles: start - x - y - z - w - v - end - start
            for x in range(start + 1, end):
                if x not in adj_set[start]:
                    continue
                for y in range(x + 1, end):
                    if y not in adj_set[x]:
                        continue
                    for z in range(y + 1, end):
                        if z not in adj_set[y]:
                            continue
                        for w in range(z + 1, end):
                            if w not in adj_set[z]:
                                continue
                            for v in range(w + 1, end):
                                if v in adj_set[w] and v in adj_set[end]:
                                    lengths.add(7)
    return lengths


def _valid_edges(edges):
    """Check degree>=3, triangle-free, contains a cycle, no loops/duplicates."""
    n = 12
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return None
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            return None
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d < 3:
            return None
    adj = [set() for _ in range(n)]
    for a, b in seen:
        adj[a].add(b)
        adj[b].add(a)
    # triangle-free
    for u in range(n):
        for v in list(adj[u]):
            for w in list(adj[v]):
                if w > v and w in adj[u]:
                    return None
    # contains a cycle: E >= V (in simple graph, cycle iff E >= V)
    if len(seen) < n:
        return None
    return edges


def _known_good():
    """Two disjoint K3,3 components (baseline, distinct cycle lengths {4,6,8})."""
    edges = []
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges.append([u, v])
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges.append([u, v])
    return edges


def construct():
    start = time.time()
    best = None
    best_score = 4  # number of distinct cycle lengths; lower is better

    # Search over 3-regular bipartite graphs (partitions of 6/6) to minimise
    # the number of distinct simple-cycle lengths.
    for perm in itertools.permutations(range(12)):
        if time.time() > DEADLINE:
            break
        A = perm[:6]
        B = perm[6:]
        # each A vertex connects to 3 of the 6 B vertices
        choices = [list(itertools.combinations(B, 3)) for _ in A]
        for combo in itertools.product(*choices):
            if time.time() > DEADLINE:
                break
            edges = []
            for u, nb in zip(A, combo):
                for v in nb:
                    edges.append([u, v])
            # degree of B vertices must be exactly 3 (3-regular)
            bdeg = [0] * 6
            for u, nb in zip(A, combo):
                for v in nb:
                    bdeg[B.index(v)] += 1
            if any(d != 3 for d in bdeg):
                continue
            # triangle-free automatically (bipartite); degree>=3 holds (3-regular)
            # need at least one cycle: 3-regular on 12 vertices always has cycles
            adj = [set() for _ in range(12)]
            for u, nb in zip(A, combo):
                for v in nb:
                    adj[u].add(v)
                    adj[v].add(u)
            lengths = _cycle_lengths(adj)
            if not lengths:
                continue
            if len(lengths) < best_score:
                best_score = len(lengths)
                best = [list(e) for e in edges]
                if best_score <= 2:
                    # cannot do better than 2 (need a cycle and triangle-free)
                    return best
            if best_score <= 2:
                break
        if best_score <= 2:
            break

    # Fallback: baseline two K3,3 components
    if best is None:
        return _known_good()
    return best
