import time
import random
from collections import deque
from itertools import combinations

_DEADLINE = 115.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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

def _random_4regular(n, rng):
    """Generate a random 4-regular graph on n vertices using configuration model."""
    if n < 2 or n % 2 != 0:
        return None
    # Create stubs: each vertex has 4 stubs
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    
    # Try multiple random pairings
    for attempt in range(500):
        rng.shuffle(stubs)
        edges = []
        edge_set = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append(key)
        if valid and len(edges) == 2 * n:
            return edges
    return None

def _build_adj(edges, n):
    """Build adjacency list from edges."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _girth_check(adj, g):
    """Check if graph has girth at least g using BFS with non-backtracking constraint.
    Returns (True, None) if girth >= g, or (False, cycle_length) if a shorter cycle exists."""
    n = len(adj)
    # For each edge, check for cycles of length < g starting from that edge
    # Use BFS tracking (node, prev_node) pairs
    for start_u in range(n):
        for start_v in adj[start_u]:
            if start_v <= start_u:
                continue  # Avoid double counting
            # BFS from (start_u, start_v)
            visited = set()
            queue = deque()
            queue.append((start_v, start_u, 1))
            visited.add((start_v, start_u))
            while queue:
                node, prev, depth = queue.popleft()
                if depth >= g - 1:
                    continue
                for neighbor in adj[node]:
                    if neighbor == prev:
                        continue
                    if neighbor == start_u and depth + 1 >= 3:
                        # Found a cycle
                        if depth + 1 < g:
                            return False, depth + 1
                    new_state = (neighbor, node)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((neighbor, node, depth + 1))
    return True, None

def _find_short_cycle(adj, g):
    """Find a short cycle (length < g) if one exists. Returns list of vertices or None."""
    n = len(adj)
    for start_u in range(n):
        for start_v in adj[start_u]:
            if start_v <= start_u:
                continue
            # BFS from (start_u, start_v) tracking path
            visited = {}
            queue = deque()
            queue.append((start_v, start_u, [start_u, start_v]))
            visited[(start_v, start_u)] = [start_u, start_v]
            while queue:
                node, prev, path = queue.popleft()
                depth = len(path) - 1
                if depth >= g - 1:
                    continue
                for neighbor in adj[node]:
                    if neighbor == prev:
                        continue
                    if neighbor == start_u and depth + 1 >= 3:
                        if depth + 1 < g:
                            return path + [start_u]
                    new_state = (neighbor, node)
                    if new_state not in visited:
                        visited[new_state] = path + [neighbor]
                        queue.append((neighbor, node, path + [neighbor]))
    return None

def _apply_2switch_inplace(edges, adj, e1, e2, rng):
    """Apply a 2-switch in-place. Returns True if successful, False otherwise."""
    a, b = e1
    c, d = e2
    # Try both possible reconnections
    for new_e1, new_e2 in [((min(a,c), max(a,c)), (min(b,d), max(b,d))), 
                           ((min(a,d), max(a,d)), (min(b,c), max(b,c)))]:
        if new_e1 == e1 or new_e1 == e2 or new_e2 == e1 or new_e2 == e2:
            continue
        if new_e1 == new_e2:
            continue
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        # Check no duplicate edges
        if new_e1[1] in adj[new_e1[0]] or new_e2[1] in adj[new_e2[0]]:
            continue
        
        # Apply switch in-place
        # Remove old edges from adj
        adj[a].discard(b)
        adj[b].discard(a)
        adj[c].discard(d)
        adj[d].discard(c)
        
        # Add new edges to adj
        adj[new_e1[0]].add(new_e1[1])
        adj[new_e1[1]].add(new_e1[0])
        adj[new_e2[0]].add(new_e2[1])
        adj[new_e2[1]].add(new_e2[0])
        
        # Update edges list
        edges.remove(e1)
        edges.remove(e2)
        edges.append(new_e1)
        edges.append(new_e2)
        
        return True
    return False

def construct(k, g):
    global _START
    _START = time.time()
    
    if k != 4 or g != 10:
        return _luw(k, g)
    
    # Moore bound for k=4, g=10: 1 + 4*sum_{i=0}^{3} 3^i = 1 + 4*(1+3+9+27) = 1 + 4*40 = 161
    # But actual lower bound is 243
    # For even girth g, Moore bound is 1 + k * sum_{i=0}^{g/2-2} (k-1)^i
    moore_bound = 1 + k * sum((k-1)**i for i in range(g//2 - 1))
    
    # Known best: 320 vertices. Lower bound: 243.
    # Try to find a graph with n < 320 vertices.
    
    rng = random.Random(42)
    best_result = None
    best_n = float('inf')
    
    # Try various n values from lower bound upward
    min_n = max(moore_bound, 243)
    
    for n in range(min_n, min(min_n + 80, 320)):
        if _remaining() < 10:
            break
        
        # Generate random 4-regular graph
        edges = _random_4regular(n, rng)
        if edges is None:
            continue
        
        # Build adjacency
        adj = _build_adj(edges, n)
        
        # Check girth
        ok, cycle_len = _girth_check(adj, g)
        if ok:
            best_n = n
            best_result = (n, edges)
            break  # Found smallest n
        
        # If not ok, try local search to fix
        if _remaining() < 5:
            continue
        
        # Local search: find short cycles and try to break them with 2-switches
        for iteration in range(200):
            if _remaining() < 5:
                break
            
            cycle = _find_short_cycle(adj, g)
            if cycle is None:
                # No short cycles found
                ok, _ = _girth_check(adj, g)
                if ok:
                    best_n = n
                    best_result = (n, edges)
                    break
                continue
            
            # Get edges in the cycle
            cycle_edges = []
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                e = (min(u, v), max(u, v))
                cycle_edges.append(e)
            
            # Try to break the cycle with a 2-switch involving one cycle edge and one non-cycle edge
            broke = False
            for _ in range(10):
                if _remaining() < 5:
                    break
                # Pick a random cycle edge
                e1 = rng.choice(cycle_edges)
                # Pick a random edge not in the cycle
                non_cycle_edges = [e for e in edges if e not in cycle_edges]
                if not non_cycle_edges:
                    break
                e2 = rng.choice(non_cycle_edges)
                
                if _apply_2switch_inplace(edges, adj, e1, e2, rng):
                    broke = True
                    break
            
            if not broke:
                break
        
        if best_n == n:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to LUW
    return _luw(k, g)
