import random
import time
from collections import defaultdict

N = 40
DEADLINE = 115.0


def _build_adj(edge_set):
    A = [[0] * N for _ in range(N)]
    for (u, v) in edge_set:
        A[u][v] = 1
        A[v][u] = 1
    return A


def _has_c6(A):
    """Correct C6 detection. C6 exists iff some pair of non-adjacent vertices
    has >=2 internally-disjoint length-3 paths between them (they are opposite
    vertices of the C6). We detect by: for each pair (u,v), count the number of
    length-2 paths from u to a neighbor of v... 
    Efficient equivalent: C6 exists iff there exist vertices u,v with two
    distinct length-3 paths. We count, for each ordered pair (u,v), the number
    of intermediate 2-tuples (x,y) with edges u-x, x-y, y-v. If any pair has
    >=2 with distinct internal edges, C6 exists."""
    # For each middle edge (x,y), it contributes to all (u in N(x)\{y}, v in N(y)\{x}).
    # Count length-3 paths per (u,v) pair, tracking distinct (x,y) middles.
    # Use: for pair (u,v), a length-3 path is u-x-y-v. Two paths are internally
    # disjoint iff their (x,y) middles differ. So if for some (u,v) there are two
    # different (x,y) middles, C6 exists.
    # Build map: for each (x,y) edge, the set of (u,v) pairs it serves.
    # Then for each (u,v), count distinct (x,y) middles.
    middles = defaultdict(set)  # (u,v) -> set of (x,y)
    for x in range(N):
        for y in range(N):
            if x == y or not A[x][y]:
                continue
            for u in range(N):
                if u == y or not A[u][x]:
                    continue
                for v in range(N):
                    if v == x or v == u or not A[y][v]:
                        continue
                    key = (min(u, v), max(u, v))
                    middles[key].add((x, y))
    for key, ms in middles.items():
        if len(ms) >= 2:
            return True
    return False


def construct():
    start = time.time()
    rng = random.Random(1337)
    best_edges = 0
    best_set = None
    best_A = None

    # ---------- Phase 1: dense random init + greedy C6 removal ----------
    for trial in range(5):
        if time.time() - start > DEADLINE - 35:
            break
        edge_set = set()
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.5:
                    edge_set.add((u, v))
        A = _build_adj(edge_set)

        # Greedy removal to break C6s
        iters = 0
        while _has_c6(A):
            iters += 1
            if iters > 4000:
                break
            # find a C6 and remove one of its edges
            c6 = _find_c6(A)
            if c6 is None:
                break
            e = c6[rng.randrange(6)]
            A[e[0]][e[1]] = 0
            A[e[1]][e[0]] = 0
            edge_set.discard((min(e[0], e[1]), max(e[0], e[1])))

        if best_set is None or len(edge_set) > best_edges:
            best_edges = len(edge_set)
            best_set = set(edge_set)
            best_A = [row[:] for row in A]

    if best_set is None:
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges

    edge_set = best_set
    A = best_A

    def non_edges():
        ne = []
        for u in range(N):
            for v in range(u + 1, N):
                if not A[u][v]:
                    ne.append((u, v))
        return ne

    def add_pass():
        ne = non_edges()
        rng.shuffle(ne)
        added = 0
        for (u, v) in ne:
            A[u][v] = 1
            A[v][u] = 1
            if _has_c6(A):
                A[u][v] = 0
                A[v][u] = 0
            else:
                edge_set.add((u, v))
                added += 1
        return added

    def remove_readd_pass():
        edges = list(edge_set)
        rng.shuffle(edges)
        improved = False
        for (u, v) in edges:
            if time.time() - start > DEADLINE - 12:
                return
            A[u][v] = 0
            A[v][u] = 0
            edge_set.discard((u, v))
            added = add_pass()
            if added >= 1:
                improved = True
            else:
                A[u][v] = 1
                A[v][u] = 1
                edge_set.add((u, v))
        return improved

    no_improve = 0
    while time.time() - start < DEADLINE - 18:
        added = add_pass()
        if added > 0:
            no_improve = 0
        else:
            if remove_readd_pass():
                no_improve = 0
                continue
            no_improve += 1
        if no_improve >= 5:
            break

    # Final polish: random single-edge swaps
    while time.time() - start < DEADLINE - 6:
        edges = list(edge_set)
        if not edges:
            break
        rng.shuffle(edges)
        improved = False
        for (u, v) in edges:
            if time.time() - start > DEADLINE - 6:
                break
            A[u][v] = 0
            A[v][u] = 0
            edge_set.discard((u, v))
            ne = non_edges()
            rng.shuffle(ne)
            ok = False
            for (a, b) in ne:
                A[a][b] = 1
                A[b][a] = 1
                if _has_c6(A):
                    A[a][b] = 0
                    A[b][a] = 0
                else:
                    edge_set.add((a, b))
                    ok = True
                    break
            if not ok:
                A[u][v] = 1
                A[v][u] = 1
                edge_set.add((u, v))
            else:
                improved = True
        if not improved:
            break

    return [[u, v] for (u, v) in edge_set]


def _find_c6(A):
    """Return 6 edges of a C6, or None."""
    middles = defaultdict(set)
    for x in range(N):
        for y in range(N):
            if x == y or not A[x][y]:
                continue
            for u in range(N):
                if u == y or not A[u][x]:
                    continue
                for v in range(N):
                    if v == x or v == u or not A[y][v]:
                        continue
                    key = (min(u, v), max(u, v))
                    middles[key].add((x, y))
    for key, ms in middles.items():
        if len(ms) >= 2:
            ms = list(ms)
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    x1, y1 = ms[i]
                    x2, y2 = ms[j]
                    if x1 != x2 and x1 != y2 and y1 != x2 and y1 != y2:
                        a, b = key
                        # path1: a-x1-y1-b, path2: b-x2-y2-a
                        # determine orientation: need u in N(a), v in N(b)
                        # For (x1,y1): u in N(x1) with u!=y1, v in N(y1) with v!=x1
                        # We know a-x1 edge and b-y1 edge (since a in N(x1), b in N(y1))
                        edges = [(a, x1), (x1, y1), (y1, b),
                                 (b, x2), (x2, y2), (y2, a)]
                        return edges
    return None
