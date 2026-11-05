import time
import random

# ---- Fast Q3 detector (bitmask adjacency) ----
# Q3 edges (canonical, vertices 0..7)
_Q3 = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

def _bits(mask):
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb

def _edge_makes_q3(adj, n, u, v):
    """Fast: is edge (u,v) part of some Q3? Two role assignments (u->0,v->1) and (u->1,v->0)."""
    if _check_role(adj, n, u, v):
        return True
    if _check_role(adj, n, v, u):
        return True
    return False

def _check_role(adj, n, a0, a1):
    """a0 plays Q3-vertex 0, a1 plays Q3-vertex 1 (edge (0,1))."""
    # 0's other Q3-neighbors: 2 and 4 (both in N(a0) \ {a1})
    # 1's other Q3-neighbors: 3 and 5 (both in N(a1) \ {a0})
    n0 = adj[a0] & ~(1 << a1)
    n1 = adj[a1] & ~(1 << a0)
    for a2 in _bits(n0):
        for a4 in _bits(n0):
            if a4 == a2:
                continue
            # 6 = common neighbor of a2,a4 (not a0,a1)
            common6 = adj[a2] & adj[a4]
            common6 &= ~(1 << a0)
            common6 &= ~(1 << a1)
            for a6 in _bits(common6):
                if a6 == a2 or a6 == a4:
                    continue
                # 7 must be common neighbor of a3,a5 and adjacent to a6
                for a3 in _bits(n1):
                    for a5 in _bits(n1):
                        if a5 == a3:
                            continue
                        common7 = adj[a3] & adj[a5]
                        common7 &= ~(1 << a0)
                        common7 &= ~(1 << a1)
                        common7 &= ~(1 << a6)
                        for a7 in _bits(common7):
                            if a7 == a3 or a7 == a5 or a7 == a6:
                                continue
                            # verify all required edges
                            if not ((adj[a0] >> a2) & 1): continue
                            if not ((adj[a0] >> a4) & 1): continue
                            if not ((adj[a1] >> a3) & 1): continue
                            if not ((adj[a1] >> a5) & 1): continue
                            if not ((adj[a2] >> a3) & 1): continue
                            if not ((adj[a2] >> a6) & 1): continue
                            if not ((adj[a3] >> a7) & 1): continue
                            if not ((adj[a4] >> a5) & 1): continue
                            if not ((adj[a4] >> a6) & 1): continue
                            if not ((adj[a5] >> a7) & 1): continue
                            if not ((adj[a6] >> a7) & 1): continue
                            verts = [a0,a1,a2,a4,a3,a5,a6,a7]
                            if len(set(verts)) == 8:
                                return True
    return False

def _edge_list(adj, n):
    es = []
    for u in range(n):
        au = adj[u]
        for v in _bits(au):
            if v > u:
                es.append([u, v])
    return es

def _count_edges(adj, n):
    return sum(bin(adj[i]).count('1') for i in range(n)) // 2

def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill

    rng = random.Random(12345)

    # ---- Phase 1: start from a dense random graph, then remove edges to make it Q3-free ----
    # Dense start: include each edge with probability p. Then iteratively remove one edge
    # from a detected Q3 until none remain. This yields a near-maximal Q3-free graph.
    def dense_init(p):
        adj = [0]*n
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # remove edges to kill all Q3s
        for _ in range(5000):
            if time.time() > deadline - 5.0:
                break
            # find a Q3 by scanning edges
            found = None
            for u in range(n):
                au = adj[u]
                for v in _bits(au):
                    if v <= u:
                        continue
                    if _edge_makes_q3(adj, n, u, v):
                        found = (u, v)
                        break
                if found:
                    break
            if found is None:
                break
            # remove a random edge from that Q3 to break it: just remove (u,v)
            u, v = found
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        return adj

    # Try a few densities, keep the densest Q3-free result
    best_adj = None
    best_count = -1
    for p in (0.9, 0.8, 0.7, 0.6):
        if time.time() > deadline - 10.0:
            break
        adj = dense_init(p)
        c = _count_edges(adj, n)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    adj = best_adj
    cur_count = best_count

    # ---- Phase 2: greedy fill (add any edge that doesn't create a Q3) ----
    def greedy_fill():
        nonlocal cur_count
        changed = True
        while changed and time.time() < deadline - 8.0:
            changed = False
            for u in range(n):
                au = adj[u]
                for v in _bits(~au & ((1 << n) - 1)):
                    if v <= u:
                        continue
                    if time.time() > deadline - 8.0:
                        break
                    # try add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if _edge_makes_q3(adj, n, u, v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        cur_count += 1
                        changed = True
                if time.time() > deadline - 8.0:
                    break

    greedy_fill()
    best_adj = adj[:]
    best_count = cur_count

    # ---- Phase 3: simulated annealing with edge swaps (constant edge count) ----
    # Propose: remove a random present edge, add a random missing edge.
    # Objective: keep Q3-free; maximize edge count (constant here).
    # To actually gain edges we also allow pure-add moves when beneficial.
    # We track best (max edge count) Q3-free state.

    def make_edge_lists():
        present = []
        missing = []
        for u in range(n):
            au = adj[u]
            for v in _bits(au):
                if v > u:
                    present.append((u, v))
            for v in _bits(~au & ((1 << n) - 1)):
                if v > u:
                    missing.append((u, v))
        return present, missing

    T0 = 1.0
    Tend = 0.02
    T = T0
    total_steps = 0
    # We'll run swaps until near deadline, cooling over time.
    while time.time() < deadline - 6.0:
        present, missing = make_edge_lists()
        if not present or not missing:
            break
        # pick a move
        if rng.random() < 0.5:
            # pure add: try to add a missing edge
            u, v = rng.choice(missing)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _edge_makes_q3(adj, n, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                cur_count += 1
                if cur_count > best_count:
                    best_count = cur_count
                    best_adj = adj[:]
        else:
            # swap: remove one present, add one missing
            u1, v1 = rng.choice(present)
            u2, v2 = rng.choice(missing)
            if (u1, v1) == (u2, v2) or (v1, u1) == (u2, v2):
                continue
            # remove (u1,v1)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            # add (u2,v2)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            if _edge_makes_q3(adj, n, u2, v2):
                # invalid: revert both
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
            else:
                # valid swap; edge count unchanged
                pass
        total_steps += 1
        # cool
        if total_steps % 20000 == 0:
            T = max(Tend, T * 0.99)

    # ---- Phase 4: final greedy fill ----
    greedy_fill()
    if cur_count > best_count:
        best_count = cur_count
        best_adj = adj[:]

    return _edge_list(best_adj, n)
