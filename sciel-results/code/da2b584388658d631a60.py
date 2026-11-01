"""Structural change: use a voltage graph lift of the Petersen graph over Z_m
with Thue-Morse-derived voltage assignments to structurally minimize low-order
polynomial collisions in closed walk sums. This is a fundamentally different
design from the previous K_3,3-based lifts and random search approaches.

The Petersen graph has girth 5 and 15 edges. A lift over Z_m gives 10*m vertices.
For girth 16, we need m >= 51 (since 10*51 = 510 >= Moore bound 510).
The Thue-Morse sequence provides a deterministic, high-dispersion pattern that
avoids arithmetic structure that could create short cycles.
"""
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _thue_morse(n):
    """Generate the first n bits of the Thue-Morse sequence."""
    result = []
    for i in range(n):
        result.append(bin(i).count('1') % 2)
    return result

def _thue_morse_shifts(m, e_count):
    """Generate e_count voltage shifts in Z_m using Thue-Morse-inspired dispersion.
    
    The key idea: use the Thue-Morse sequence to create a deterministic
    pseudo-random-looking sequence of shifts that avoids arithmetic progressions
    and quadratic patterns which are prone to creating short cycles.
    """
    # Generate a longer Thue-Morse sequence and use it to index into a permutation
    tm = _thue_morse(4 * m)
    
    # Create a permutation of 1..m-1 using Thue-Morse ordering
    # This gives a well-spread, non-arithmetic sequence
    perm = []
    for i in range(1, m):
        perm.append((tm[i % (4*m)] * m + i) % (m - 1) + 1)
    
    # Shuffle the permutation using another Thue-Morse pass for more dispersion
    # Use a different phase of the Thue-Morse sequence
    tm2 = _thue_morse(4 * m)
    perm2 = []
    for i in range(len(perm)):
        perm2.append((tm2[(i + m) % (4*m)] * len(perm) + i) % len(perm))
    
    perm_final = [perm[perm2[i]] for i in range(len(perm))]
    
    # Select e_count shifts from this well-dispersed permutation
    step = max(1, len(perm_final) // e_count)
    shifts = [perm_final[i * step % len(perm_final)] for i in range(e_count)]
    
    return shifts

def _thue_morse_shifts_v2(m, e_count):
    """Alternative Thue-Morse based shift generation with different structure."""
    tm = _thue_morse(8 * m)
    
    # Use Thue-Morse to create pairs of shifts that sum to specific values
    # This helps avoid certain types of collisions
    shifts = []
    for i in range(e_count):
        # Use a different segment of the Thue-Morse sequence for each edge
        offset = i * 7 % (8 * m)
        base_val = tm[offset] * m // 2 + i * (m // e_count)
        shifts.append(base_val % (m - 1) + 1)
    
    return shifts

def _thue_morse_shifts_v3(m, e_count):
    """Third variant: use Thue-Morse to determine sign and magnitude separately."""
    tm = _thue_morse(16 * m)
    
    shifts = []
    for i in range(e_count):
        # Magnitude from one part of Thue-Morse
        mag_idx = i * 13 % (16 * m)
        magnitude = (tm[mag_idx] * m // 3 + i * (m // (e_count * 2))) % (m - 1) + 1
        
        # Sign/direction from another part
        dir_idx = (i * 19 + 5) % (16 * m)
        direction = tm[dir_idx]
        
        if direction:
            shifts.append(magnitude)
        else:
            shifts.append((m - magnitude) % (m - 1) + 1)
    
    return shifts

def _petersen_edges():
    """Return the 15 edges of the Petersen graph (10 vertices)."""
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    return [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # outer cycle
        (5,7), (7,9), (9,6), (6,8), (8,5),  # inner pentagram
        (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
    ]

def _build_lift(base_edges, m, shifts):
    """Build a voltage lift of the base graph over Z_m.
    
    Returns (n, edges) if the lift is a valid simple 3-regular graph, else None.
    """
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
    n = n_base * m
    e_count = len(base_edges)
    
    adj = [set() for _ in range(n)]
    edges = []
    valid = True
    
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if a == b:
                valid = False
                break
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
            else:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, edges

def _thue_morse_petersen_lift(m):
    """Try Thue-Morse voltage assignments on the Petersen graph lift."""
    base_edges = _petersen_edges()
    e_count = len(base_edges)
    
    shift_generators = [
        _thue_morse_shifts,
        _thue_morse_shifts_v2,
        _thue_morse_shifts_v3,
    ]
    
    # Also try with offset variations
    for gen in shift_generators:
        for offset in range(5):
            # Create a modified generator with offset
            def shifted_gen(m, e_count, _gen=gen, _off=offset):
                shifts = _gen(m, e_count)
                # Apply offset
                return [(s + _off) % (m - 1) + 1 for s in shifts]
            
            shifts = shifted_gen(m, e_count)
            result = _build_lift(base_edges, m, shifts)
            if result is not None:
                n, edges = result
                adj = [list(set()) for _ in range(n)]
                adj_sets = [set() for _ in range(n)]
                for u, v in edges:
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
                adj_list = [list(s) for s in adj_sets]
                if _verify_girth_fast(adj_list, n, 16):
                    return n, edges
    
    return None

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
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

def _random_3reg(n, deadline):
    """Try to generate a random 3-regular graph on n vertices and check girth."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Fisher-Yates style random 3-regular graph generation
    # Use the configuration model with rejection
    for attempt in range(50):
        if _time.time() > deadline - 15:
            break
        
        rng = _random.Random(12345 + attempt)
        
        # Create stubs: each vertex has 3 stubs
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
        # Randomly pair stubs
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if not valid:
            continue
        
        # Check degree
        degree_ok = all(len(adj[i]) == 3 for i in range(n))
        if not degree_ok:
            continue
        
        adj_list = [list(s) for s in adj]
        if _verify_girth_fast(adj_list, n, 16):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
    
    return None

def _voltage_lift_search(k, g, deadline):
    """Search for a valid voltage lift with various base graphs and lift sizes."""
    if k != 3:
        return None
    
    base_graphs = [
        ("petersen", _petersen_edges(), 10),
        ("k33", [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)], 6),
        ("cube", [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)], 8),
    ]
    
    moore = 510  # for g=16, k=3
    
    for name, base_edges, n_base in base_graphs:
        e_count = len(base_edges)
        
        m_min = max(1, (moore + n_base - 1) // n_base)
        
        for m in range(m_min, m_min + 300):
            if _time.time() > deadline - 10:
                return None
            
            n = n_base * m
            
            # Try Thue-Morse shifts first
            for gen in [_thue_morse_shifts, _thue_morse_shifts_v2, _thue_morse_shifts_v3]:
                shifts = gen(m, e_count)
                result = _build_lift(base_edges, m, shifts)
                if result is not None:
                    n_r, edges = result
                    adj_sets = [set() for _ in range(n_r)]
                    for u, v in edges:
                        adj_sets[u].add(v)
                        adj_sets[v].add(u)
                    adj_list = [list(s) for s in adj_sets]
                    if _verify_girth_fast(adj_list, n_r, g):
                        return n_r, edges
            
            # Try some random shifts
            for attempt in range(20):
                if _time.time() > deadline - 5:
                    break
                rng = _random.Random(hash((name, m, attempt)))
                shifts = [rng.randint(1, m - 1) for _ in range(e_count)]
                result = _build_lift(base_edges, m, shifts)
                if result is not None:
                    n_r, edges = result
                    adj_sets = [set() for _ in range(n_r)]
                    for u, v in edges:
                        adj_sets[u].add(v)
                        adj_sets[v].add(u)
                    adj_list = [list(s) for s in adj_sets]
                    if _verify_girth_fast(adj_list, n_r, g):
                        return n_r, edges
    
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    
    # For k=3, g=16: try Thue-Morse Petersen lift first
    if g == 16:
        # Moore bound for (3,16) is 2*(2^8 - 1) = 510
        # Petersen lift: 10*m vertices, so m >= 51
        # Try m from 51 upward with Thue-Morse shifts
        for m in range(51, 120):
            if _time.time() > deadline - 20:
                break
            result = _thue_morse_petersen_lift(m)
            if result is not None:
                n, edges = result
                return n, edges
        
        # Try K_3,3 and cube base graphs with Thue-Morse
        result = _voltage_lift_search(k, g, deadline)
        if result is not None:
            return result
        
        # Try random 3-regular graphs
        for n_try in range(510, 600, 2):
            if _time.time() > deadline - 15:
                break
            result = _random_3reg(n_try, deadline)
            if result is not None:
                return result
    
    # Final fallback: LUW
    n_luw, edges_luw = _luw_construct(k, g)
    return n_luw, edges_luw
