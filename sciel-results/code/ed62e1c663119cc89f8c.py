"""
Construct a 3-regular graph with girth >= 16 using voltage lifts over
base graphs with good girth properties, combined with systematic
voltage assignment and correct girth verification.

Structural changes from previous version:
1. Correct non-backtracking girth verification using BFS
2. Use base graphs with higher girth (cube, Petersen) to reduce
   the number of short closed walks that constrain voltages
3. Systematic voltage search with early termination
4. Fallback to LUW construction
"""
import time as _time
import random as _random
from collections import deque
import numpy as np

def _verify_girth(adj, n, g):
    """
    Verify girth >= g using non-backtracking BFS from each vertex.
    A cycle of length L < g exists iff there's a non-backtracking walk
    from v back to v of length L >= 3 (but for simple graphs, we need
    to be careful: the non-backtracking walk from v to v of length L
    corresponds to a cycle of length L only if it doesn't repeat vertices
    before the end. However, the standard approach is: if there's a
    non-backtracking walk from v to v of length L, there's a cycle of
    length <= L. So if no non-backtracking walk of length < g returns
    to start, girth >= g.
    
    Actually, the correct statement: the girth is the minimum length of
    a non-backtracking closed walk. So we check: for each start vertex,
    do a non-backtracking BFS and check if we can return to start in
    < g steps.
    """
    for start in range(n):
        # BFS on non-backtracking walks: state = (current, prev)
        visited = {}  # (v, prev) -> distance
        dq = deque()
        dq.append((start, -1))
        visited[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = visited[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    # Found a non-backtracking closed walk of length d+1
                    return False
                state = (w, v)
                if state not in visited:
                    visited[state] = d + 1
                    dq.append(state)
    return True

def _verify_girth_fast(adj, n, g):
    """
    Faster girth verification using bitsets for visited states.
    State space: n * (k+1) where k is max degree. For k=3, that's 4n states.
    We encode state (v, prev) as v * 4 + prev_index where prev_index is 0..3
    (3 means "no prev" / start).
    """
    k = max(len(adj[v]) for v in range(n)) if n > 0 else 0
    # For 3-regular, k=3, so 4 states per vertex
    n_states = n * 4
    for start in range(n):
        # visited[state] = distance, -1 if not visited
        visited = np.full(n_states, -1, dtype=np.int16)
        dq = deque()
        # State (start, -1) encoded as start * 4 + 3
        start_state = start * 4 + 3
        visited[start_state] = 0
        dq.append((start, -1))
        while dq:
            v, prev = dq.popleft()
            d = visited[v * 4 + (prev if prev >= 0 else 3)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state_idx = w * 4 + (v if v >= 0 else 3)
                if visited[state_idx] == -1:
                    visited[state_idx] = d + 1
                    dq.append((w, v))
    return True

def _build_voltage_lift(base_edges, n_base, m, shifts):
    """
    Build a voltage lift graph.
    base_edges: list of (u, v) edges in the base graph
    n_base: number of vertices in base graph
    m: lift size (number of vertices per base vertex)
    shifts: list of voltage shifts for each edge, same length as base_edges
    Returns (n, edges) or None if invalid.
    """
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if a == b:
                return None
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
            else:
                # Multi-edge, invalid for simple graph
                return None
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, edges

def _luw_construct(k, g, d_override=None):
    """LUW construction - algebraic construction for cages."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q
    
    rule = {2: (1,1), 3: (2,1), 4: (1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - (c-5) % 4
        rule[c] = [(1,c0-2), (c0-1,1), (c0,1), (1,c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _search_voltages(base_edges, n_base, m, g, deadline, max_attempts=500):
    """
    Search for voltage assignments that give girth >= g.
    Uses a combination of systematic and random search with
    early girth checking.
    """
    n = n_base * m
    e_count = len(base_edges)
    
    # Precompute all non-backtracking closed walks in the base graph
    # up to length g-1, to know which voltage sums must be non-zero
    closed_walks = []  # list of lists of edge indices
    for start in range(n_base):
        # Non-backtracking DFS
        stack = [(start, -1, [start], [])]
        while stack:
            v, prev, path, edge_path = stack.pop()
            if len(path) > 1 and v == start:
                if len(path) - 1 < g:  # length < g
                    closed_walks.append(list(edge_path))
                continue
            if len(path) - 1 >= g:
                continue
            for w in range(n_base):
                if w == prev:
                    continue
                # Check if w is adjacent to v in base graph
                if (v, w) in base_edges or (w, v) in base_edges:
                    # Find edge index
                    for ei, (u1, v1) in enumerate(base_edges):
                        if (u1 == v and v1 == w) or (u1 == w and v1 == v):
                            stack.append((w, v, path + [w], edge_path + [ei]))
                            break
    
    # Remove duplicates
    seen = set()
    unique_walks = []
    for walk in closed_walks:
        key = tuple(sorted(walk))
        if key not in seen:
            seen.add(key)
            unique_walks.append(walk)
    
    # For each closed walk, the sum of voltages must not be 0 mod m
    # This is a constraint: sum(shifts[ei] for ei in walk) != 0 mod m
    
    # Try to find shifts satisfying all constraints
    # Start with random assignments and check
    
    for attempt in range(max_attempts):
        if _time.time() > deadline:
            break
        
        if attempt < 50:
            # Systematic: arithmetic progression based
            for s in range(min(m, 50)):
                step = m // e_count if e_count > 0 else 1
                shifts = [(s + i * step) % m for i in range(e_count)]
                if all(s != 0 for s in shifts):
                    # Check constraints
                    valid = True
                    for walk in unique_walks:
                        total = sum(shifts[ei] for ei in walk) % m
                        if total == 0:
                            valid = False
                            break
                    if valid:
                        return shifts
        else:
            # Random
            rng = _random.Random(attempt * 7919 + 13)
            shifts = [rng.randint(1, m - 1) for _ in range(e_count)]
            valid = True
            for walk in unique_walks:
                total = sum(shifts[ei] for ei in walk) % m
                if total == 0:
                    valid = False
                    break
            if valid:
                return shifts
    
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    
    if g != 16:
        # For other girths, use LUW
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    
    # For k=3, g=16: target n(3,16) = 936
    
    # Strategy 1: Voltage lift over K_{3,3} with m=156 (n=936)
    k33_edges = [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
    n_base = 6
    m = 156
    n_target = n_base * m  # 936
    
    shifts = _search_voltages(k33_edges, n_base, m, g, deadline, max_attempts=300)
    if shifts is not None:
        result = _build_voltage_lift(k33_edges, n_base, m, shifts)
        if result is not None:
            n, edges = result
            adj = [list(set(e[0] for e in edges if e[1]==v) | set(e[1] for e in edges if e[0]==v)) for v in range(n)]
            # Rebuild adj properly
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            adj_list = [list(s) for s in adj]
            if _verify_girth(adj_list, n, g):
                return n, edges
    
    # Strategy 2: Voltage lift over cube (8 vertices, 12 edges) with m=117 (n=936)
    cube_edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    n_base_cube = 8
    m_cube = 117
    n_target2 = n_base_cube * m_cube  # 936
    
    shifts2 = _search_voltages(cube_edges, n_base_cube, m_cube, g, deadline, max_attempts=200)
    if shifts2 is not None:
        result = _build_voltage_lift(cube_edges, n_base_cube, m_cube, shifts2)
        if result is not None:
            n, edges = result
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            adj_list = [list(s) for s in adj]
            if _verify_girth(adj_list, n, g):
                return n, edges
    
    # Strategy 3: Voltage lift over Petersen (10 vertices, 15 edges) with m=94 (n=940)
    petersen_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(5,7),(7,9),(9,6),(6,8),(8,5),(0,5),(1,6),(2,7),(3,8),(4,9)]
    n_base_p = 10
    m_p = 94
    n_target3 = n_base_p * m_p  # 940
    
    shifts3 = _search_voltages(petersen_edges, n_base_p, m_p, g, deadline, max_attempts=200)
    if shifts3 is not None:
        result = _build_voltage_lift(petersen_edges, n_base_p, m_p, shifts3)
        if result is not None:
            n, edges = result
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            adj_list = [list(s) for s in adj]
            if _verify_girth(adj_list, n, g):
                return n, edges
    
    # Strategy 4: Try larger lifts with smaller base
    # K_{3,3} with m=157 (n=942)
    for m_try in [157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]:
        if _time.time() > deadline - 5:
            break
        n_try = 6 * m_try
        shifts_try = _search_voltages(k33_edges, 6, m_try, g, deadline - 5, max_attempts=100)
        if shifts_try is not None:
            result = _build_voltage_lift(k33_edges, 6, m_try, shifts_try)
            if result is not None:
                n, edges = result
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                adj_list = [list(s) for s in adj]
                if _verify_girth(adj_list, n, g):
                    return n, edges
    
    # Fallback: LUW construction
    n_luw, edges_luw = _luw_construct(k, g)
    return n_luw, edges_luw
