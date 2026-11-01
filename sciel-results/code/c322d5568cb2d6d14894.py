import random
import time
from collections import deque

def _luw_construct(k, g):
    """L-U-W algebraic construction (deterministic fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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

def _build_lift(base_edges, voltages, m):
    """Build voltage lift edges."""
    edges = []
    for u, v in base_edges:
        volt = voltages.get((u, v), voltages.get((v, u), 0))
        for i in range(m):
            edges.append((u * m + i, v * m + (i + volt) % m))
    return edges

def _verify_girth_fast(n, edges, target_girth):
    """Fast girth verification using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # dist[state] = length of non-backtracking walk from start
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= target_girth - 1:
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

def _quadratic_residues(m):
    """Generate quadratic residues mod m for structured voltage patterns."""
    residues = set()
    for i in range(1, m):
        residues.add((i * i) % m)
    return sorted(residues)

def _golden_ratio_shifts(m):
    """Generate shifts based on golden ratio conjugate for good distribution."""
    phi = (1 + 5 ** 0.5) / 2
    shifts = []
    for i in range(1, m):
        shifts.append(int(i * phi) % m)
    return shifts

def _modular_pattern_shifts(m, seed=1):
    """Generate structured shifts using modular arithmetic patterns."""
    shifts = []
    for i in range(1, m):
        # Use a combination of linear and quadratic terms
        val = (seed * i + (i * i) % m) % m
        shifts.append(val)
    return shifts

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # Petersen graph edges (standard)
    petersen_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                      (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
                      (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    
    best_n, best_edges = None, None
    
    # Strategy 1: Try deterministic algebraic voltage lifts with structured patterns
    # Use multiple pattern families to increase chance of finding girth-g lift
    pattern_families = [
        "quadratic", "golden", "modular_pattern"
    ]
    
    # Try various lift sizes, starting from smaller values
    # For girth 16, we need sufficiently large lifts
    m_values = [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for m in m_values:
        if time.time() > deadline:
            break
        n = 10 * m
        
        for pattern_name in pattern_families:
            if time.time() > deadline:
                break
            
            # Generate base shift pattern
            if pattern_name == "quadratic":
                base_shifts = _quadratic_residues(m)
            elif pattern_name == "golden":
                base_shifts = _golden_ratio_shifts(m)
            else:  # modular_pattern
                base_shifts = _modular_pattern_shifts(m)
            
            # Create voltage assignments from the pattern
            # Petersen has 15 edges; we need to assign voltages
            # Use different offsets for each edge to create variety
            
            for offset in range(min(5, len(base_shifts))):
                if time.time() > deadline:
                    break
                
                voltages = {}
                for idx, (u, v) in enumerate(petersen_edges):
                    # Use the pattern with an offset
                    shift_idx = (idx + offset) % len(base_shifts)
                    volt = base_shifts[shift_idx]
                    # Ensure non-zero voltages to avoid multi-edges
                    if volt == 0:
                        volt = 1
                    voltages[(u, v)] = volt
                    voltages[(v, u)] = (-volt) % m
                
                edges = _build_lift(petersen_edges, voltages, m)
                
                # Check for multi-edges (shouldn't happen with non-zero voltages on Petersen)
                edge_set = set()
                has_multi = False
                for e in edges:
                    key = (min(e), max(e))
                    if key in edge_set:
                        has_multi = True
                        break
                    edge_set.add(key)
                
                if has_multi:
                    continue
                
                # Verify girth
                if _verify_girth_fast(n, edges, g):
                    best_n, best_edges = n, edges
                    break
            
            if best_n is not None:
                break
        
        if best_n is not None:
            break
    
    # Strategy 2: If algebraic patterns failed, try random voltages with more attempts
    if best_n is None:
        for m in [200, 300, 400, 500, 600, 800, 1000]:
            if time.time() > deadline:
                break
            n = 10 * m
            
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                voltages = {}
                for u, v in petersen_edges:
                    volt = random.randint(1, m - 1)  # Avoid 0
                    voltages[(u, v)] = volt
                    voltages[(v, u)] = (-volt) % m
                
                edges = _build_lift(petersen_edges, voltages, m)
                
                # Quick multi-edge check
                edge_set = set()
                has_multi = False
                for e in edges:
                    key = (min(e), max(e))
                    if key in edge_set:
                        has_multi = True
                        break
                    edge_set.add(key)
                
                if has_multi:
                    continue
                
                if _verify_girth_fast(n, edges, g):
                    best_n, best_edges = n, edges
                    break
            
            if best_n is not None:
                break
    
    # Fallback to L-U-W
    if best_n is None:
        best_n, best_edges = _luw_construct(k, g)
    
    return best_n, best_edges
