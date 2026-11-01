"""
Constructs a 3-regular graph with girth at least g using voltage lifts of the Petersen graph.
"""
import time
import random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """
    Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    # We check for cycles of length < g.
    # A non-backtracking walk of length L starting at s and ending at s (with L >= 3)
    # implies a cycle of length <= L.
    # We perform BFS on the state space (v, prev).
    # State (v, prev) means we are at v, came from prev.
    # Distance is the length of the walk.
    
    for start in range(n):
        # dist[state] = length of walk
        # state is (v, prev)
        dist = {}
        dq = deque()
        # Initial states: (start, -1) is not a walk, it's the start.
        # From start, we can go to any neighbor w.
        # State (w, start) has length 1.
        for w in adj[start]:
            state = (w, start)
            dist[state] = 1
            dq.append(state)
        
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            
            # If we reach start again, we found a cycle.
            # The walk was start -> ... -> v -> start?
            # No, the state is (v, prev).
            # If v == start, the walk length is d.
            # But we started with (w, start) of length 1.
            # If we see (start, u) with length d, it means a walk of length d+1?
            # Let's trace:
            # (w, start) len 1.
            # Next is (x, w) len 2.
            # If x == start, state is (start, w) len 2.
            # This corresponds to walk start -> w -> start. Length 2.
            # But simple graphs have no cycles of length 2.
            # So we should not count this.
            # We only care about cycles of length >= 3.
            
            if v == start:
                # The walk length is d.
                # But this state (start, prev) was reached from (prev, start)?
                # No, (start, prev) means we are at start, came from prev.
                # The previous state was (prev, something).
                # Actually, if v == start, the walk is start -> ... -> prev -> start.
                # Length is d.
                # If d < g, we have a cycle of length d (or less).
                # However, if d == 2, it's a backtrack.
                # In a simple graph, we don't have 2-cycles.
                # So if d >= 3 and d < g, return False.
                if d < g and d >= 3:
                    return False
                # If d == 1, impossible (no self-loops).
                # If d == 2, backtrack. Ignore.
                continue

            if d >= g - 1:
                # No need to expand further, any cycle found will be >= g
                continue
            
            for w in adj[v]:
                if w == prev:
                    continue
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _build_lift(base_edges, m, shifts):
    """
    Build a voltage graph lift over Z_m.
    base_edges: list of (u,v) on the base graph
    m: lift size
    shifts: list of voltage shifts, one per base edge
    Returns (n, adj_list, edge_list) or None if invalid.
    """
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    
    # Map edge index to shift
    # base_edges must correspond to shifts order
    
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
            else:
                # Multiple edge?
                # This can happen if the base graph has multiple edges or if shifts cause collisions.
                # Petersen is simple, so no multiple edges in base.
                # Collision in lift: (u, i) -- (v, i+vol) and (u, i') -- (v, i'+vol')
                # If u=u', i=i', then v=v', i+vol = i'+vol' => vol=vol'.
                # So no multiple edges unless base has multiple edges.
                pass

    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    adj_list = [list(s) for s in adj]
    return n, adj_list, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    if k != 3:
        # Fallback for other k
        # LUW construction
        return _luw_construct(k, g)

    # Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    e_count = 15

    # Moore bound for k=3, g=16 is 510.
    # n = 10 * m. So m >= 51.
    # We start from m=52 to be safe (n=520).
    
    best_result = None
    best_n = float('inf')
    
    rng = random.Random(42)
    
    # Search m from 52 upwards
    # Allocate more time to smaller m
    for m in range(52, 1000):
        if time.time() > deadline - 5:
            break
            
        n = n_base * m
        if n >= best_n:
            continue
            
        # Determine number of attempts
        # Smaller m is harder to satisfy, so try more?
        # Actually, smaller m has a smaller search space for shifts (m^15).
        # But the constraint is tighter.
        # Let's try 2000 attempts for m=52, decreasing as m increases.
        attempts = 2000 if m < 60 else 500
        
        for _ in range(attempts):
            if time.time() > deadline - 2:
                break
            
            # Random shifts
            # Avoid 0? 0 is fine.
            shifts = [rng.randint(0, m - 1) for _ in range(e_count)]
            
            result = _build_lift(petersen_edges, m, shifts)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

    # If not found, fallback to LUW
    if time.time() < deadline - 10:
        try:
            n_luw, edges_luw = _luw_construct(3, g)
            if n_luw < best_n:
                return n_luw, edges_luw
        except:
            pass

    if best_result is not None:
        return best_result
        
    # Last resort: return a large LUW graph
    return _luw_construct(3, g)

def _luw_construct(k, g, d_override=None):
    """
    LUW construction for k=3.
    """
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
            
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

