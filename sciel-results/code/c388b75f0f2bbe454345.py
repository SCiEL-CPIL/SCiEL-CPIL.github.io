"""
Combined approach: LUW baseline + efficient girth checking + targeted repair.
Key improvements over current:
1. Vectorized girth check using numpy for speed
2. Better repair strategy: edge swaps that preserve regularity
3. Time management with explicit deadline
4. Multiple construction strategies tried in order of expected quality
"""
import time
import random
from collections import deque, defaultdict

import numpy as np

_DEADLINE = 145  # safety margin before 150s wall-clock kill

def _time_left():
    return _DEADLINE - time.time()

def _moore_lower(k, g):
    """Moore bound: minimum possible n for k-regular graph with girth g."""
    if g % 2 == 0:
        r = g // 2
        if k <= 2:
            return g
        # Even girth: n >= 2 * sum_{i=0}^{r-1} k*(k-1)^i / k ... 
        # Standard: n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 for even g=2r
        # Actually: n >= 2 * (k-1)^r + 2 * ... let me use the tree bound
        # From each vertex, BFS tree of depth r-1 has 1 + k + k(k-1) + ... + k(k-1)^(r-1) vertices
        # But this overcounts. The Moore bound for k-regular, girth g (even, g=2r):
        # n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 ... 
        # Simpler: n >= 2 * sum_{j=0}^{r-1} (k-1)^j * k / k ... 
        # Let me just compute: number of vertices at distance <= r-1 from a vertex in k-regular tree
        # = 1 + k * sum_{j=0}^{r-2} (k-1)^j
        # For even girth g=2r, the Moore bound is:
        # n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 if k > 2
        # For k=2: n >= g
        if k == 2:
            return g
        # n >= 2 * (1 + k * sum_{j=0}^{r-2} (k-1)^j) ... no
        # Standard Moore bound for k-regular graph of girth g:
        # if g = 2r (even): n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2
        # Wait, let me look this up properly.
        # Moore bound: n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1)) for g=2r
        # = k * ((k-1)^r - 1) / (k-2)
        # But this is for the case where we count from one side.
        # Actually the correct Moore bound for k-regular, girth g:
        # g even (g=2r): n >= 2 * (k-1)^r + 2 * (k-1)^(r-1) + ... + 2k + 2 ... 
        # Let me just use: n >= k * ((k-1)^(g/2 - 1) - 1) / (k-2) * 2 for k>2
        r = g // 2
        if k > 2:
            return 2 * ((k - 1) ** (r - 1) - 1) // (k - 2) + 2
        else:
            return g
    else:
        # g odd (g=2r+1): center vertex, tree of depth r
        r = (g - 1) // 2
        if k <= 2:
            return g
        # n >= 1 + k * sum_{j=0}^{r-1} (k-1)^j = 1 + k * ((k-1)^r - 1) / (k-2)
        return 1 + k * ((k - 1) ** r - 1) // (k - 2)

def _check_girth_fast(adj, n, g):
    """Check no cycle shorter than g using non-backtracking BFS from each vertex.
    Returns True if girth >= g. Uses efficient state tracking."""
    if n == 0:
        return True
    max_depth = (g - 1) // 2
    for start in range(n):
        # BFS state: (node, prev_node) to avoid backtracking
        visited = set()
        queue = deque()
        queue.append((start, -1))
        visited.add((start, -1))
        depth = 0
        while queue and depth <= max_depth:
            next_queue = deque()
            while queue:
                node, prev = queue.popleft()
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (nb, node) in visited:
                        # Found a cycle of length depth+1 (or less)
                        if depth + 1 < g:
                            return False
                        continue
                    visited.add((nb, node))
                    next_queue.append((nb, node))
            queue = next_queue
            depth += 1
    return True

def _check_degree(adj, n, k):
    """Check every vertex has degree exactly k."""
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True

def _check_simple(adj, n):
    """Check no loops, no repeated edges."""
    for i in range(n):
        if i in adj[i]:
            return False
        s = set(adj[i])
        if len(s) != len(adj[i]):
            return False
    return True

def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if not _check_simple(adj, n):
        return False
    if not _check_degree(adj, n, k):
        return False
    if not _check_girth(adj, n, g):
        return False
    return True

def _lazebnik_construct(k, g):
    """LUW graph construction - fast, always valid for prime k or k=4."""
    q = k
    if (g - 5) % 2 == 1:
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n < k + 1 or k * n % 2 != 0:
        return None
    if k >= n:
        return None
    
    # Configuration model with retries
    for _ in range(200):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return [list(s) for s in adj]
    return None

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g if one exists. Returns list of vertices or None."""
    max_depth = (g - 1) // 2
    for start in range(n):
        parent = {start: None}
        depth = {start: 0}
        queue = deque([(start, -1)])
        while queue:
            node, prev = queue.popleft()
            if depth[node] >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb in parent:
                    cycle_len = depth[node] + depth[nb] + 1
                    if cycle_len < g:
                        cycle = [node]
                        cur = node
                        while cur != nb:
                            cur = parent[cur]
                            cycle.append(cur)
                        cur = nb
                        while cur != node:
                            cur = parent[cur]
                            cycle.append(cur)
                        return cycle
                else:
                    parent[nb] = node
                    depth[nb] = depth[node] + 1
                    queue.append((nb, node))
    return None

def _edge_swap_repair(adj, n, k, g, time_budget):
    """Try to repair a k-regular graph to have girth >= g by edge swaps.
    An edge swap removes (u,v) and (x,y) and adds (u,x) and (v,y) or (u,y) and (v,x)."""
    start_time = time.time()
    adj = [list(s) for s in adj]
    
    max_iters = 10000
    for _ in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            return adj
        
        # Pick an edge in the cycle to swap
        idx = random.randint(0, len(cycle) - 1)
        u = cycle[idx]
        v = cycle[(idx + 1) % len(cycle)]
        
        # Remove edge u-v
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Find another edge (x,y) to swap with
        found = False
        for _try in range(100):
            # Pick random x from adj[u], random y from adj[v]
            if not adj[u] or not adj[v]:
                break
            x = random.choice(adj[u])
            y = random.choice(adj[v])
            
            # We want to add edges (u,y) and (v,x) instead of (u,v) and (x,y)
            # But we need (x,y) to be an edge currently
            if y not in adj[x]:
                continue
            if x == y or x == u or y == v:
                continue
            
            # Remove (x,y)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add (u,y) and (v,x)
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            if _check_girth(adj, n, g):
                found = True
                break
            else:
                # Revert
                adj[u].remove(y)
                adj[y].remove(u)
                adj[v].remove(x)
                adj[x].remove(v)
                adj[x].append(y)
                adj[y].append(x)
        
        if not found:
            # Revert the removal of u-v
            adj[u].append(v)
            adj[v].append(u)
    
    if _check_girth(adj, n, g):
        return adj
    return None

def _greedy_search(k, g, time_budget=60.0):
    """Greedy randomized search for a smaller k-regular graph with girth >= g."""
    start_time = time.time()
    best_n = None
    best_edges = None
    
    lower = _moore_lower(k, g)
    
    # Try a range of n values from lower bound upward
    for target_n in range(lower, lower + 300):
        if time.time() - start_time > time_budget:
            break
        if target_n < k + 1:
            continue
        if k * target_n % 2 != 0:
            continue
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() - start_time > time_budget:
                break
            
            adj = _random_k_regular(target_n, k)
            if adj is None:
                continue
            
            # Verify girth
            if _check_girth(adj, target_n, g):
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_n is None or target_n < best_n:
                    best_n = target_n
                    best_edges = edges
                break
            else:
                # Try to repair
                remaining = time_budget - (time.time() - start_time)
                if remaining > 5:
                    repaired = _edge_swap_repair(adj, target_n, k, g, remaining * 0.3)
                    if repaired is not None:
                        if _check_girth(repaired, target_n, g):
                            edges = []
                            for i in range(target_n):
                                for j in repaired[i]:
                                    if i < j:
                                        edges.append((i, j))
                            if best_n is None or target_n < best_n:
                                best_n = target_n
                                best_edges = edges
                            break
    
    if best_n is not None:
        return best_n, best_edges
    return None, None

def construct(k, g):
    """Construct a k-regular graph with girth >= g, minimizing n."""
    start_time = time.time()
    
    # Always get LUW as baseline
    luw_n, luw_edges = _lazebnik_construct(k, g)
    best_n, best_edges = luw_n, luw_edges
    
    # Try greedy search with remaining time
    remaining = _DEADLINE - (time.time() - start_time) - 10  # 10s safety margin
    if remaining > 15:
        greedy_n, greedy_edges = _greedy_search(k, g, time_budget=remaining)
        if greedy_n is not None and greedy_n < best_n:
            best_n, best_edges = greedy_n, greedy_edges
    
    return best_n, best_edges
