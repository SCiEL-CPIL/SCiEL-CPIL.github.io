"""Direct construction of a 4-regular graph with girth at least g using
stochastic local search (Simulated Annealing) with 2-switch moves to
eliminate short cycles, falling back to LUW construction if needed."""
import time
import sys
import random
import numpy as np
from collections import deque

_DEADLINE = 115.0  # safety margin below the 120s evaluator kill


def construct(k, g):
    t_start = time.time()

    if k != 4:
        # Fallback: use the LUW construction for any k, g
        return _luw(k, g)

    # Try stochastic local search for k=4, g=10
    # First get the LUW baseline
    luw_n, luw_edges = _luw(k, g)
    
    # Try to find a smaller graph via SA
    best_n, best_edges = _sa_search(k, g, luw_n, t_start)
    
    if best_n < luw_n:
        return best_n, best_edges
    
    return luw_n, luw_edges


def _compute_girth(adj, n, max_girth):
    """Compute the girth of a graph by BFS from each vertex.
    Returns the length of the shortest cycle, or max_girth if none found shorter."""
    for start in range(n):
        # BFS to find shortest cycle back to start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < max_girth:
                        return cycle_len
    return max_girth


def _count_short_cycles(adj, n, max_len):
    """Count cycles of length < max_len using non-backtracking walks."""
    count = 0
    for start in range(n):
        # Non-backtracking BFS up to depth floor(max_len/2)
        # Track (current, prev) pairs
        seen = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start))
            seen.add((nb, start))
        depth = 1
        while queue and depth < max_len:
            new_queue = deque()
            while queue:
                u, prev = queue.popleft()
                if u == start and depth >= 3:
                    count += 1
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if (v, u) in seen:
                        continue
                    seen.add((v, u))
                    new_queue.append((v, u))
            queue = new_queue
            depth += 1
    return count


def _random_4reg(n, seed=42):
    """Generate a random 4-regular graph on n vertices using configuration model."""
    rng = random.Random(seed)
    # Create stubs: 4 per vertex
    stubs = []
    for i in range(n):
        stubs.extend([i] * 4)
    rng.shuffle(stubs)
    
    adj = [[] for _ in range(n)]
    i = 0
    while i < len(stubs):
        u = stubs[i]
        j = i + 1
        # Find a different vertex
        while stubs[j] == u or (u, stubs[j]) in adj[u] or (stubs[j], u) in adj[stubs[j]]:
            j += 1
            if j >= len(stubs):
                # Restart
                stubs = []
                for k in range(n):
                    stubs.extend([k] * 4)
                rng.shuffle(stubs)
                i = 0
                break
        if i == 0 and j >= len(stubs):
            # Failed, try again
            return _random_4reg(n, seed + 1)
        v = stubs[j]
        adj[u].append(v)
        adj[v].append(u)
        i = j + 1
    
    # Verify it's 4-regular
    for i in range(n):
        if len(adj[i]) != 4:
            return _random_4reg(n, seed + 1)
    
    return adj


def _sa_search(k, g, max_n, t_start):
    """Simulated annealing to find k-regular graph with girth >= g on fewer than max_n vertices."""
    rng = random.Random(12345)
    deadline = _DEADLINE - 2.0  # safety margin
    
    # Try several candidate sizes from Moore bound upward
    # Moore bound for (4,10): 246
    moore = 2 * (k - 1) + 2 * sum((k - 1) ** i for i in range(1, (g - 1) // 2 + 1))
    moore = max(moore, k + 2)
    
    # Try sizes from moore to max_n-1
    candidates = []
    for n in range(moore, max_n):
        if n % 2 == 0:  # 4-regular graph needs even number of edges, so n*4/2 = 2n edges
            candidates.append(n)
    
    if not candidates:
        return max_n, []
    
    # Start with smaller candidates first
    for n in candidates:
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this size
        best_girth = 0
        best_adj = None
        
        for restart in range(3):
            if time.time() > deadline:
                break
            
            adj = _random_4reg(n, seed=42 + restart * 1000)
            if adj is None:
                continue
            
            # Compute initial girth
            girth = _compute_girth(adj, n, g)
            
            if girth >= g:
                # Found a valid graph!
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            
            # SA: try to increase girth via 2-switches
            T = 10.0
            T_min = 0.1
            cooling = 0.995
            
            current_girth = girth
            current_adj = [list(neigh) for neigh in adj]
            
            steps = 0
            while time.time() < deadline and T > T_min:
                steps += 1
                
                # Pick a random 2-switch: remove edges (a,b) and (c,d), add (a,c) and (b,d)
                # or (a,d) and (b,c)
                a = rng.randint(0, n - 1)
                b = rng.choice(current_adj[a])
                
                # Find c != a,b with d in adj[c] such that (c,d) and (a,b) are disjoint
                c = rng.randint(0, n - 1)
                if c == a or c == b:
                    continue
                d = rng.choice(current_adj[c])
                if d == a or d == b:
                    continue
                
                # Check that new edges don't already exist
                if (c in current_adj[a]) or (d in current_adj[a]) or (b in current_adj[c]) or (d in current_adj[c]):
                    continue
                if (a in current_adj[c]) or (b in current_adj[c]):
                    continue
                
                # Apply 2-switch
                current_adj[a].remove(b)
                current_adj[b].remove(a)
                current_adj[c].remove(d)
                current_adj[d].remove(c)
                
                current_adj[a].append(c)
                current_adj[c].append(a)
                current_adj[b].append(d)
                current_adj[d].append(b)
                
                # Compute new girth
                new_girth = _compute_girth(current_adj, n, g)
                
                delta = new_girth - current_girth
                
                if delta >= 0 or rng.random() < np.exp(delta / T):
                    current_girth = new_girth
                    if current_girth >= g:
                        # Found valid graph
                        edges = []
                        for i in range(n):
                            for j in current_adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
                else:
                    # Reject: undo
                    current_adj[a].remove(c)
                    current_adj[c].remove(a)
                    current_adj[b].remove(d)
                    current_adj[d].remove(b)
                    
                    current_adj[a].append(b)
                    current_adj[b].append(a)
                    current_adj[c].append(d)
                    current_adj[d].append(c)
                
                T *= cooling
        
        # Keep track of best found
        if best_girth < current_girth:
            best_girth = current_girth
            best_adj = current_adj
    
    return max_n, []


def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    # girth = d + 5 for odd d, so d = g - 5 for odd g, g - 4 for even g
    # Actually the paper says girth is at least d + 5.
    # For g=10, we want d+5 >= 10, so d >= 5. Use d=5.
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Build the rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Use BFS to find the connected component
    from collections import deque as dq
    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
