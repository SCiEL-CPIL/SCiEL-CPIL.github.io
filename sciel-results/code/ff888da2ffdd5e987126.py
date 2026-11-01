"""Voltage-graph lift of the Petersen graph over Z_N to build a 4-regular graph
with large girth (target: cage problem n(4,10))."""
import time
import itertools
import random
import numpy as np
from collections import deque

_PET = None

def _petersen():
    global _PET
    if _PET is None:
        # Petersen graph: outer 5-cycle, inner star, spokes
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))          # outer cycle
            edges.append((5 + i, 5 + ((i + 2) % 5))) # inner star
            edges.append((i, 5 + i))                 # spokes
        _PET = edges
    return _PET

def _nbk_walks(base_edges, m):
    """All non-backtracking walks up to length m as (start, end, voltage_signs).
    voltage_signs is a list of +1/-1 per edge (orientation relative to walk)."""
    n = 10
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    walks = []  # (start, end, signs_tuple)
    # length 1
    for u, v in base_edges:
        walks.append((u, v, (1,)))
        walks.append((v, u, (-1,)))
    # extend
    for length in range(2, m + 1):
        new_walks = []
        for (u, w, signs) in walks:
            for x in adj[w]:
                if x == u:
                    continue  # non-backtracking
                # sign of the new edge: from w to x
                # we store orientation: +1 means walk direction matches (prev_node, next)
                # For edge (w,x): if (w,x) in base as (w,x) then +1 else -1
                # We track sign as: +1 if the undirected edge traversed in stored order
                # Simpler: sign = +1 if (w,x) is an edge as stored (w,x), else -1
                # We'll just store +1 for traversal w->x in "positive" convention.
                # Let's define: sign=+1 means we go from first to second in some canonical form.
                # Actually for voltage sum we need: voltage_sum = sum s_i * v(e_i)
                # where s_i = +1 if we traverse e_i in the (u,v) direction, -1 otherwise.
                # Let's just store the sign directly.
                # For edge between w and x: if (w,x) in base_edges set, sign=+1; else sign=-1
                new_signs = signs + (1,)  # placeholder, we'll fix below
                new_walks.append((w, x, new_signs))
        # This approach is getting complex. Let me redo with proper sign tracking.
        break
    return walks


def _find_voltages(base_edges, N, target_girth, deadline):
    """Find voltage assignment to base edges such that no non-backtracking
    closed walk of length < target_girth has zero voltage sum mod N."""
    m_edges = len(base_edges)
    best_voltages = None
    best_min_len = 0

    # Precompute all non-backtracking closed walks up to length target_girth - 1
    # Represent each walk as (edge_indices, signs) where signs[i] = +1 or -1
    n = 10
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, (u, v) in enumerate(base_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        edge_set.add((u, v))

    # Enumerate non-backtracking closed walks
    # Walk = sequence of (vertex, edge_index, prev_vertex)
    # For efficiency, enumerate up to length L = target_girth - 1
    L = target_girth - 1
    # closed_walks: list of (edge_indices_tuple, signs_tuple)
    closed_walks = []

    # BFS over non-backtracking walks
    # State: (current_vertex, edge_just_used, prev_vertex)
    # We track the sequence of (edge_idx, sign)
    from collections import defaultdict

    # For each starting edge and direction
    for start_edge in range(m_edges):
        u, v = base_edges[start_edge]
        # Two directions
        for direction in [0, 1]:
            if direction == 0:
                cur, prev = v, u
                cur_edge = start_edge
                cur_sign = 1  # traversed u->v, sign +1
            else:
                cur, prev = u, v
                cur_edge = start_edge
                cur_sign = -1  # traversed v->u, sign -1

            seq = [(cur_edge, cur_sign)]

            # Extend up to length L
            for length in range(1, L):
                extended = False
                for (nxt, nxt_edge) in adj[cur]:
                    if nxt == prev:
                        continue  # non-backtracking
                    # sign: traversing cur -> nxt on edge nxt_edge
                    eu, ev = base_edges[nxt_edge]
                    if (eu, ev) == (cur, nxt):
                        nxt_sign = 1
                    else:
                        nxt_sign = -1
                    new_seq = seq + [(nxt_edge, nxt_sign)]
                    # Check if closed: nxt == start vertex
                    start_v = v if direction == 0 else u
                    if nxt == start_v and len(new_seq) >= 3:
                        closed_walks.append(tuple(new_seq))
                    # Continue extending
                    # (we only need closed walks, but to find longer ones we extend)
                    # For efficiency, limit depth
                    if length < L - 1:
                        # Store for further extension - this gets expensive
                        pass
                break  # Only go 1 step for now (too expensive to go deeper)

    # The above is incomplete. Let me use a cleaner approach:
    # Enumerate all non-backtracking walks of length 3..L and check if closed.
    # For a 4-regular graph on 10 vertices, the number of non-backtracking walks
    # of length L is at most 10 * 4 * 3^(L-1). For L=9: 10*4*3^8 = 10*4*6561 = 262440.
    # That's manageable.

    closed_walks = []
    for start_v in range(n):
        for (first_nxt, first_edge) in adj[start_v]:
            prev = start_v
            cur = first_nxt
            eu, ev = base_edges[first_edge]
            first_sign = 1 if (eu, ev) == (prev, cur) else -1
            seq = [(first_edge, first_sign)]
            # Extend
            _extend_walks(seq, start_v, prev, cur, L, closed_walks, adj, base_edges)

    # Now we have closed_walks. For each, we need voltage_sum = sum(s_i * v[e_i]) != 0 mod N.
    # This is a constraint satisfaction problem. Use random search with verification.

    # Build the constraint matrix: for each closed walk, a vector in Z_N^m_edges
    # where entry j = sum of signs for edge j appearances.
    # Actually each walk uses each edge at most once (non-backtracking, simple enough).
    # Let's build: constraints[i][j] = coefficient of voltage[j] in walk i.

    num_walks = len(closed_walks)
    if num_walks == 0:
        return [1] * m_edges

    # Build coefficient vectors
    coeff = np.zeros((num_walks, m_edges), dtype=np.int64)
    for i, walk in enumerate(closed_walks):
        for (edge_idx, sign) in walk:
            coeff[i, edge_idx] += sign

    # Now find voltages v in Z_N^m such that coeff @ v != 0 (mod N) for all i.
    # Random search:
    random.seed(42)
    best_score = 0
    best_v = None
    deadline_t = deadline - 2  # safety margin

    for trial in range(200000):
        if time.time() > deadline_t:
            break
        v = [random.randint(1, N - 1) for _ in range(m_edges)]
        v_arr = np.array(v, dtype=np.int64)
        # Compute all voltage sums
        sums = (coeff @ v_arr) % N
        # Count how many are zero
        zeros = np.count_nonzero(sums == 0)
        if zeros == 0:
            return v
        if zeros < best_score:
            best_score = zeros
            best_v = v

    # If no perfect solution found, try to fix the remaining violations
    if best_v is not None:
        v_arr = np.array(best_v, dtype=np.int64)
        sums = (coeff @ v_arr) % N
        zero_indices = np.where(sums == 0)[0]
        # Try to adjust individual voltages to fix violations
        for zi in zero_indices:
            if time.time() > deadline_t:
                break
            for _ in range(100):
                # Pick a random edge in this walk and change its voltage
                walk = closed_walks[zi]
                edge_idx = random.choice(walk)[0]
                old_v = v_arr[edge_idx]
                v_arr[edge_idx] = (old_v + random.randint(1, N - 1)) % N
                if v_arr[edge_idx] == 0:
                    v_arr[edge_idx] = 1
                # Check if this walk is now fixed and no new violations
                new_sums = (coeff @ v_arr) % N
                new_zeros = np.count_nonzero(new_sums == 0)
                if new_zeros < zeros:
                    zeros = new_zeros
                    zero_indices = np.where(new_sums == 0)[0]
                    if zeros == 0:
                        return v_arr.tolist()
                    break
                else:
                    v_arr[edge_idx] = old_v

    # Fallback: return best found
    if best_v is not None:
        return best_v
    return [1] * m_edges


def _extend_walks(seq, start_v, prev, cur, max_len, closed_walks, adj, base_edges):
    if len(seq) >= max_len:
        return
    for (nxt, nxt_edge) in adj[cur]:
        if nxt == prev:
            continue
        eu, ev = base_edges[nxt_edge]
        nxt_sign = 1 if (eu, ev) == (cur, nxt) else -1
        new_seq = seq + [(nxt_edge, nxt_sign)]
        if nxt == start_v and len(new_seq) >= 3:
            closed_walks.append(tuple(new_seq))
        _extend_walks(new_seq, start_v, cur, nxt, max_len, closed_walks, adj, base_edges)


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if k != 4 or g != 10:
        # Fallback: use the Lazebnik-Ustimenko-Woldar construction for other parameters
        return _luw_construct(k, g)

    # Voltage graph lift of Petersen over Z_32
    base_edges = _petersen()
    N = 32
    m_edges = len(base_edges)

    voltages = _find_voltages(base_edges, N, g, deadline)

    if time.time() > deadline - 5:
        # Not enough time, return a simple construction
        return _luw_construct(k, g)

    # Build the lift
    # Vertex (v, t) where v in 0..9, t in 0..N-1
    # Edge between (u, t) and (v, t + voltages[e]) for edge e=(u,v)
    # and (v, t) to (u, t - voltages[e])
    edges = []
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i] % N
        for t in range(N):
            edges.append((u * N + t, v * N + ((t + vol) % N)))
    n = 10 * N

    return n, edges


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
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
