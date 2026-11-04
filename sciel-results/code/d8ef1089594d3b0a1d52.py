"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _nonbacktracking_counts(adj, n, max_len):
    """Count non-backtracking closed walks of each length 2..max_len from every vertex.
    Returns dict length -> total count (0 if girth >= that length+... actually counts walks of that exact length)."""
    # state: (v, prev) -> count of non-backtracking walks of current length ending at v coming from prev
    # length 1: walks of length 1 are just edges (open). We track length L walks.
    # For closed non-backtracking walks of length L: start at s, end at s, no immediate backtracking.
    # We do DP over (current, prev).
    # Initialize length-1: for each edge (u,v), state (v, u) = 1, and (u, v) = 1.
    cur = {}
    for u in range(n):
        for v in adj[u]:
            cur[(v, u)] = cur.get((v, u), 0) + 1
    counts = {1: 0}
    for L in range(2, max_len + 1):
        nxt = {}
        for (v, u), cnt in cur.items():
            for w in adj[v]:
                if w == u:
                    continue
                nxt[(w, v)] = nxt.get((w, v), 0) + cnt
        cur = nxt
        # closed walks of length L: state (v, u) with v == start? We lost start.
        # Instead track start: state (start, prev, cur). That's O(n^3). For n=320 too big for long L.
        # Alternative: number of closed NB walks of length L = sum over s of (NB walk from s to s).
        # We can compute via: for each start s, do DP. But that's n times. For n=320, L up to 9, states n^2=102400, *9*320 ~ 294M too big.
        # Use the standard: closed NB walks of length L = trace of the non-backtracking matrix power.
        # For our purposes we only need to know if ANY closed NB walk of length < g exists.
        # Simpler: BFS-based girth check is expensive. Use the counts of NB walks from each start via a different method.
        counts[L] = None  # placeholder
    return counts


def _girth_by_bfs(adj, n, g):
    """Return girth (length of shortest cycle) or g if no cycle shorter than g found.
    Uses BFS from each vertex but with early exit; for bipartite graphs cycles are even."""
    # For speed, use the non-backtracking closed-walk test via a start-indexed DP but only up to g-1,
    # and stop as soon as a closed walk is found.
    # We'll do: for each start s, do DP over (prev, cur) for lengths 2..g-1 and check if cur==s.
    # Optimize: only need to detect existence, so use sets.
    for s in range(n):
        # state: (prev, cur) meaning a non-backtracking walk of current length ending cur, came from prev, started at s
        # length 1: (s, v) for v in adj[s]
        cur_states = set()
        for v in adj[s]:
            cur_states.add((s, v))
        for L in range(2, g):
            nxt = set()
            for (u, v) in cur_states:
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == s:
                        return L
                    nxt.add((v, w))
            cur_states = nxt
            if not cur_states:
                break
    return g


def _build_random_4reg_bipartite(n_half, rng):
    """Build a random 4-regular bipartite graph on 2*n_half vertices (0..n_half-1 left, n_half..2*n_half-1 right).
    Returns adjacency list of length 2*n_half (each list of 4 neighbors)."""
    n = 2 * n_half
    # left vertices get 4 random right neighbors; right vertices get 4 random left neighbors.
    # To ensure regularity on both sides, use a configuration-model / bipartite random regular graph.
    # Simpler: generate left degree-4 via random assignment, then fix right degrees by permutation.
    # Use the standard: left side stubs = 4 per left vertex = 4*n_half stubs. Right side 4 per right vertex.
    # Pair left stubs to right stubs uniformly at random (a random 4-regular bipartite multigraph),
    # then derandomize: if multi-edges, swap. We'll just sample and retry/fix.
    left_stubs = []
    for i in range(n_half):
        for _ in range(4):
            left_stubs.append(i)
    right_stubs = []
    for j in range(n_half):
        for _ in range(4):
            right_stubs.append(n_half + j)
    # random bijection
    perm = list(range(4 * n_half))
    rng.shuffle(perm)
    adj = [[] for _ in range(n)]
    for i in range(4 * n_half):
        l = left_stubs[i]
        r = right_stubs[perm[i]]
        adj[l].append(r)
        adj[r].append(l)
    # remove duplicate edges (multi-edges) by swapping to make simple
    # We'll just check and if duplicates, do a few random swaps to fix.
    for _ in range(2000):
        # find a duplicate
        dup = None
        for l in range(n_half):
            seen = set()
            for r in adj[l]:
                if r in seen:
                    dup = l
                    break
                seen.add(r)
            if dup is not None:
                break
        if dup is None:
            break
        # swap a right neighbor of dup with a different right neighbor of another left vertex
        l1 = dup
        r1 = adj[l1][0]
        # find l2 != l1 with r1 not in adj[l2] and some r2 in adj[l2] not in adj[l1]
        found = False
        for l2 in range(n_half):
            if l2 == l1:
                continue
            if r1 in adj[l2]:
                continue
            for r2 in adj[l2]:
                if r2 in adj[l1]:
                    continue
                # swap: l1's r1 <-> l2's r2
                adj[l1][adj[l1].index(r1)] = r2
                adj[l2][adj[l2].index(r2)] = r1
                adj[r1].remove(l1); adj[r1].append(l2)
                adj[r2].remove(l2); adj[r2].append(l1)
                found = True
                break
            if found:
                break
        if not found:
            break
    # final: ensure each left has 4 unique right and each right has 4 unique left
    return adj


def construct(k, g):
    """Construct a k-regular graph of girth at least g on as few vertices as possible.
    Returns (n, edges)."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill
    rng = random.Random(1234567)

    # For k=4, g=10: target small n. Best known 320. Try to find n < 320 with girth >= 10.
    # Lower bound (Moore bound) for 4-regular girth 10: n >= 2*(1 + (k-1) + (k-1)^2 + ... ) for bipartite.
    # Moore bound for even girth g=2r: n >= k*( (k-1)^(r-1) - 1 )/(k-2) for k>2.
    # r=5: n >= 4*((3^4 -1)/(1)) = 4*(81-1)=320. So Moore bound is 320!
    # Wait: Moore bound for girth 10, degree 4: n >= 2 * sum_{i=0}^{4} (k-1)^i = 2*(1+3+9+27+81)=2*121=242.
    # The best lower bound stated is 243. Moore bound = 242 (even girth formula: n >= 2*sum_{i=0}^{r-1}(k-1)^i, r=g/2=5).
    # 2*(1+3+9+27+81)=2*121=242.
    # So n must be >= 242. Best known 320. We aim to find something in [242, 320).
    # Try n from 242 upward (even, since bipartite), find a 4-regular bipartite graph of girth >= 10.

    # We'll search n = 242, 244, ... up to 318 (step 2, bipartite even n).
    # For each n, try to build a random 4-regular bipartite graph and improve girth via local search.
    # n_half = n//2.

    best = None  # (n, edges)

    for n in range(242, 320, 2):
        if time.time() > deadline:
            break
        n_half = n // 2
        # Try several random starts
        found = False
        for attempt in range(30):
            if time.time() > deadline:
                break
            adj = _build_random_4reg_bipartite(n_half, rng)
            # verify degree-4 and simple
            ok = True
            for v in range(n):
                if len(adj[v]) != 4:
                    ok = False
                    break
                if len(set(adj[v])) != 4:
                    ok = False
                    break
            if not ok:
                continue
            cur_girth = _girth_by_bfs(adj, n, g)
            if cur_girth >= g:
                # success!
                edges = []
                for v in range(n_half):
                    for w in adj[v]:
                        edges.append((v, w))
                best = (n, edges)
                found = True
                break
            # local search to improve girth: try edge swaps
            # We'll do a simple hill-climb: repeatedly try a random swap and keep if girth improves.
            improved = True
            iters = 0
            while improved and iters < 200 and time.time() < deadline:
                improved = False
                iters += 1
                # pick random left vertex l1, one of its right neighbors r1
                l1 = rng.randrange(n_half)
                r1 = rng.choice(adj[l1])
                # pick random left vertex l2 != l1, one of its right neighbors r2, with r2 != r1
                l2 = rng.randrange(n_half)
                if l2 == l1:
                    continue
                r2 = rng.choice(adj[l2])
                if r2 == r1:
                    continue
                # perform swap
                adj[l1][adj[l1].index(r1)] = r2
                adj[l2][adj[l2].index(r2)] = r1
                adj[r1].remove(l1); adj[r1].append(l2)
                adj[r2].remove(l2); adj[r2].append(l1)
                new_girth = _girth_by_bfs(adj, n, g)
                if new_girth > cur_girth:
                    cur_girth = new_girth
                    improved = True
                    if cur_girth >= g:
                        edges = []
                        for v in range(n_half):
                            for w in adj[v]:
                                edges.append((v, w))
                        best = (n, edges)
                        found = True
                        break
                else:
                    # revert
                    adj[l1][adj[l1].index(r2)] = r1
                    adj[l2][adj[l2].index(r1)] = r2
                    adj[r1].remove(l2); adj[r1].append(l1)
                    adj[r2].remove(l1); adj[r2].append(l2)
            if found:
                break
        if found:
            break

    if best is None:
        # Fallback: use the Lazebnik-Ustimenko-Woldar construction (original) to guarantee validity
        return _luw_fallback(k, g)

    return best


def _luw_fallback(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
