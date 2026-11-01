import random
from collections import deque
import time

def construct(k, g):
    # For k=4, g=10, use improved voltage assignment on Paley(9)
    if k != 4 or g != 10:
        # Fallback for other (k, g)
        return _fallback_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin
    
    # GF(9) Arithmetic
    def mul(x, y):
        a, b = divmod(x, 3)
        c, d = divmod(y, 3)
        real = (a * c - b * d) % 3
        imag = (a * d + b * c) % 3
        return real + imag * 3
    
    def sub(x, y):
        a, b = divmod(x, 3)
        c, d = divmod(y, 3)
        return ((a - c) % 3) + ((b - d) % 3) * 3
    
    # Squares in GF(9)^*
    squares = set()
    for i in range(1, 9):
        squares.add(mul(i, i))
    
    # Base Paley Graph edges
    base_edges = []
    base_adj = [[] for _ in range(9)]
    for u in range(9):
        for v in range(u + 1, 9):
            diff = sub(u, v)
            if diff in squares:
                base_edges.append((u, v))
                base_adj[u].append(v)
                base_adj[v].append(u)
    
    # Find all short cycles in base graph (length 4, 5, 6, 7, 8)
    # For girth 10, we need to avoid cycles of length < 10 in the lift
    # A cycle of length L in base becomes a cycle in lift iff sum of voltages = 0 mod L_lift
    
    # For efficiency, focus on 4-cycles and 6-cycles (most common short cycles)
    short_cycles = []
    
    # Find 4-cycles in base
    for u in range(9):
        for i in range(len(base_adj[u])):
            v = base_adj[u][i]
            for j in range(len(base_adj[v])):
                w = base_adj[v][j]
                if w == u:
                    continue
                for x in base_adj[w]:
                    if x != v and x in base_adj[u]:
                        # Cycle u-v-w-x-u
                        cycle = (u, v, w, x)
                        short_cycles.append(cycle)
    
    # Find 6-cycles in base (sample some)
    # This is expensive, so we'll rely on the girth check
    
    # Target sizes
    L_candidates = [36, 35, 34, 37, 33, 38, 40, 42]
    
    for L in L_candidates:
        N = 9 * L
        if N < 242:
            continue
        
        if time.time() > deadline:
            break
        
        # Try to find voltage assignment
        best_voltages = None
        best_deficit = float('inf')
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Initialize voltages
            voltages = [random.randint(0, L - 1) for _ in base_edges]
            
            # Local search: try to minimize number of violated constraints
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Count violated 4-cycle constraints
                violated = 0
                for cycle in short_cycles:
                    # Get edge indices for this cycle
                    # Cycle: u-v-w-x-u
                    u, v, w, x = cycle
                    # Find edge indices
                    e1 = base_edges.index((u, v)) if (u, v) in base_edges else base_edges.index((v, u))
                    e2 = base_edges.index((v, w)) if (v, w) in base_edges else base_edges.index((w, v))
                    e3 = base_edges.index((w, x)) if (w, x) in base_edges else base_edges.index((x, w))
                    e4 = base_edges.index((x, u)) if (x, u) in base_edges else base_edges.index((u, x))
                    
                    # Sum of voltages (with signs based on direction)
                    # For simplicity, just check if sum is 0
                    s = (voltages[e1] + voltages[e2] + voltages[e3] + voltages[e4]) % L
                    if s == 0:
                        violated += 1
                
                if violated == 0:
                    # Check full girth
                    adj = [[] for _ in range(N)]
                    for i, (u, v) in enumerate(base_edges):
                        vol = voltages[i]
                        for offset in range(L):
                            u_lift = u * L + offset
                            v_lift = v * L + (offset + vol) % L
                            adj[u_lift].append(v_lift)
                            adj[v_lift].append(u_lift)
                    
                    if _check_girth(adj, N, g):
                        edges = []
                        for u in range(N):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return N, edges
                
                if violated < best_deficit:
                    best_deficit = violated
                    best_voltages = voltages[:]
                
                # Try a random 2-switch
                e1_idx, e2_idx = random.sample(range(len(base_edges)), 2)
                voltages[e1_idx], voltages[e2_idx] = voltages[e2_idx], voltages[e1_idx]
        
        # If best_voltages has few violations, try to polish
        if best_voltages and best_deficit < len(short_cycles) * 0.1:
            voltages = best_voltages[:]
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                violated = 0
                for cycle in short_cycles:
                    u, v, w, x = cycle
                    e1 = base_edges.index((u, v)) if (u, v) in base_edges else base_edges.index((v, u))
                    e2 = base_edges.index((v, w)) if (v, w) in base_edges else base_edges.index((w, v))
                    e3 = base_edges.index((w, x)) if (w, x) in base_edges else base_edges.index((x, w))
                    e4 = base_edges.index((x, u)) if (x, u) in base_edges else base_edges.index((u, x))
                    
                    s = (voltages[e1] + voltages[e2] + voltages[e3] + voltages[e4]) % L
                    if s == 0:
                        violated += 1
                
                if violated == 0:
                    adj = [[] for _ in range(N)]
                    for i, (u, v) in enumerate(base_edges):
                        vol = voltages[i]
                        for offset in range(L):
                            u_lift = u * L + offset
                            v_lift = v * L + (offset + vol) % L
                            adj[u_lift].append(v_lift)
                            adj[v_lift].append(u_lift)
                    
                    if _check_girth(adj, N, g):
                        edges = []
                        for u in range(N):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return N, edges
                
                if violated == 0:
                    break
                
                # Random perturbation
                idx = random.randint(0, len(base_edges) - 1)
                voltages[idx] = (voltages[idx] + random.randint(1, L - 1)) % L
    
    # Fallback: LUW graph
    return _fallback_construct(k, g)


def _check_girth(adj, n, girth):
    """Check if graph has no cycle shorter than girth using non-backtracking BFS."""
    for start in range(n):
        # State: (current, prev) -> depth
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb))
            visited[(start, nb)] = 1
        
        while queue:
            cur, prev = queue.popleft()
            depth = visited[(cur, prev)]
            if depth + 1 >= girth:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                new_state = (nb, cur)
                if new_state in visited:
                    d2 = visited[new_state]
                    if depth + 1 + d2 < girth:
                        return False
                else:
                    visited[new_state] = depth + 1
                    if depth + 1 < girth:
                        queue.append(new_state)
    return True


def _fallback_construct(k, g):
    """Fallback to LUW-style construction."""
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
