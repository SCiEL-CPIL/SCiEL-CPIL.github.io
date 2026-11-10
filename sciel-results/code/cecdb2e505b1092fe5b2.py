import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # ---- Build PG(2,5) polarity graph: the known 90-edge C4-extremal base ----
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    n = len(points)  # 31

    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    # adjacency as Python int bitsets (fast & / | / bit_count)
    adj = [0] * n
    edges = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            if dot(pi, points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])

    if len(edges) != 90:
        # safety: should never trigger, but guarantee a valid 91-edge output
        return _fill(edges, n)

    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))

    def is_edge(i, j):
        return (i, j) in edge_set or (j, i) in edge_set

    def total_c4(adj):
        # exact number of 4-cycles (subgraphs) via common-neighbor pairs
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    total += (adj[i] & adj[j]).bit_count()
        return total // 2

    def new_c4(i, j, adj):
        # new 4-cycles created by adding edge (i,j): distinct 3-paths i-a-b-j
        ai = adj[i] & ~(1 << j)
        aj = adj[j] & ~(1 << i)
        cnt = 0
        x = ai
        while x:
            lb = x & (-x)
            a = lb.bit_length() - 1
            cnt += (adj[a] & aj).bit_count()
            x ^= lb
        return cnt

    # ---- Choose the non-edge that adds the fewest 4-cycles ----
    best_edge = None
    best_cost = None
    for i in range(n):
        for j in range(i + 1, n):
            if is_edge(i, j):
                continue
            c = new_c4(i, j, adj)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_edge = (i, j)
        if time.time() > deadline - 25:
            break
    if time.time() > deadline - 25:
        # fallback: any non-edge
        for i in range(n):
            for j in range(i + 1, n):
                if not is_edge(i, j):
                    best_edge = (i, j)
                    break
            if best_edge:
                break

    if best_edge is None:
        return _fill(edges, n)

    i, j = best_edge
    edges.append([i, j])
    adj[i] |= (1 << j)
    adj[j] |= (1 << i)
    edge_set.add((i, j))
    edge_set.add((j, i))

    # ---- Light local search: try swapping in a better non-edge ----
    if time.time() < deadline - 15:
        cur_total = total_c4(adj)
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            improved = False
            for a in range(n):
                for b in range(a + 1, n):
                    if is_edge(a, b):
                        continue
                    # remove a random existing edge e, add (a,b); keep if better
                    # cheap: only test against a few candidate edges to bound time
                    cand = None
                    for e in edges:
                        if (e[0], e[1]) == (a, b):
                            continue
                        cand = e
                        break
                    if cand is None:
                        continue
                    x, y = cand
                    # cost delta of removing (x,y) and adding (a,b)
                    # new total = cur_total - c4_removed + c4_added  (approx via recompute is safer)
                    # do a bounded exact check: temporarily modify, recompute, revert
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    t = total_c4(adj)
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    if t < cur_total:
                        # commit
                        edges.remove([x, y])
                        edge_set.discard((x, y))
                        edge_set.discard((y, x))
                        edges.append([a, b])
                        edge_set.add((a, b))
                        edge_set.add((b, a))
                        adj[x] &= ~(1 << y)
                        adj[y] &= ~(1 << x)
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        cur_total = t
                        improved = True
                        break
                if improved or time.time() > deadline - 15:
                    break
            if not improved:
                break

    if len(edges) != 91:
        return _fill(edges, n)
    return edges[:91]


def _fill(edges, n):
    # guarantee exactly 91 distinct edges on 0..n-1
    existing = set()
    for e in edges:
        existing.add(tuple(sorted(e)))
    out = [list(t) for t in existing]
    for i in range(n):
        for j in range(i + 1, n):
            if len(out) >= 91:
                break
            if (i, j) not in existing:
                out.append([i, j])
                existing.add((i, j))
        if len(out) >= 91:
            break
    return out[:91]
