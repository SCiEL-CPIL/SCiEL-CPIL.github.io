import time
import random
import heapq
import numpy as np
from collections import deque

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _generate_random_k_regular(n, k, time_limit=5.0):
    """Generate a random k-regular simple graph using configuration model + 2-switches."""
    start_time = time.time()
    
    # Configuration model: create half-edges
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    # Randomly pair up half-edges
    adjacency = [set() for _ in range(n)]
    
    def try_make_simple(max_attempts=1000):
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                return None
            
            # Create random perfect matching of half-edges
            perm = list(range(len(half_edges)))
            random.shuffle(perm)
            
            adjacency = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(perm), 2):
                u = half_edges[perm[i]]
                v = half_edges[perm[i+1]]
                if u == v:
                    valid = False
                    break
                if v in adjacency[u]:
                    valid = False
                    break
                adjacency[u].add(v)
                adjacency[v].add(u)
            
            if not valid:
                continue
            
            # Check if all degrees are exactly k
            for i in range(n):
                if len(adjacency[i]) != k:
                    valid = False
                    break
            
            if valid:
                return adjacency
        
        return None
    
    adjacency = try_make_simple()
    if adjacency is None:
        return None
    
    # Use 2-switches to ensure simplicity (already done above) and randomize
    # Perform some random 2-switches to get a more random-looking graph
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        u1 = random.randint(0, n-1)
        u2 = random.randint(0, n-1)
        if u1 == u2:
            continue
        if len(adjacency[u1]) < k or len(adjacency[u2]) < k:
            continue
        # Pick random neighbors
        v1_list = list(adjacency[u1])
        v2_list = list(adjacency[u2])
        v1 = random.choice(v1_list)
        v2 = random.choice(v2_list)
        # Check if switch is valid: u1-v2 and u2-v1 must not exist, and u1!=v2, u2!=v1
        if v2 == u1 or v1 == u2:
            continue
        if v2 in adjacency[u1] or v1 in adjacency[u2]:
            continue
        # Perform switch
        adjacency[u1].discard(v1)
        adjacency[u1].add(v2)
        adjacency[v1].discard(u1)
        adjacency[v1].add(u2)
        adjacency[u2].discard(v2)
        adjacency[u2].add(v1)
        adjacency[v2].discard(u2)
        adjacency[v2].add(u1)
    
    return adjacency

def _compute_girth(adjacency, n, max_girth=100):
    """Compute the girth of the graph using BFS from each vertex."""
    girth = max_girth
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_girth:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] == dist[u] + 1:
                    # Found a cycle of length dist[u] + 1 + dist[v] - dist[u] = dist[v] + 1
                    # Actually, cycle length = dist[u] + 1 + (dist[v] - dist[u]) = dist[v] + 1
                    # Wait, that's not right. Let me think again.
                    # If we found v at distance dist[v] from start, and u is at distance dist[u],
                    # and u-v is an edge, then we have a cycle of length dist[u] + 1 + dist[v]
                    # if the paths from start to u and start to v are disjoint except at start.
                    # This is a lower bound on the cycle length through this edge.
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < girth:
                        girth = cycle_len
    return girth

def _find_short_cycle(adjacency, n, max_len=15):
    """Find a cycle of length at most max_len, return the cycle or None."""
    for start in range(n):
        # BFS with parent tracking
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_len:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] == dist[u] + 1:
                    # Found a cycle
                    # Reconstruct cycle
                    cycle = []
                    cur = u
                    while cur != v:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(v)
                    cycle.append(u)
                    # Remove duplicates if any
                    if len(cycle) <= max_len:
                        return cycle
    return None

def _greedy_switch_improve(adjacency, n, k, target_girth, time_limit=30.0):
    """Improve girth using 2-switches."""
    start_time = time.time()
    current_girth = _compute_girth(adjacency, n, max_girth=target_girth + 1)
    
    if current_girth >= target_girth:
        return adjacency, current_girth
    
    # Try to improve girth using 2-switches
    max_iterations = 50000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if current_girth >= target_girth:
            break
        
        # Find a short cycle
        cycle = _find_short_cycle(adjacency, n, max_len=current_girth)
        if cycle is None:
            break
        
        # Try to break this cycle using 2-switches
        # For each edge in the cycle, try to switch it with another edge
        improved = False
        cycle_edges = []
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            if u < v:
                cycle_edges.append((u, v))
            else:
                cycle_edges.append((v, u))
        
        for u, v in cycle_edges:
            if time.time() - start_time > time_limit:
                break
            
            # Try to find a 2-switch involving edge (u, v)
            # Pick random u' from neighbors of u (not v), v' from neighbors of v (not u)
            u_neighbors = [x for x in adjacency[u] if x != v]
            v_neighbors = [x for x in adjacency[v] if x != u]
            
            if not u_neighbors or not v_neighbors:
                continue
            
            for _ in range(10):  # Try a few random switches
                u_prime = random.choice(u_neighbors)
                v_prime = random.choice(v_neighbors)
                
                # Check if switch is valid
                if u_prime == v or v_prime == u:
                    continue
                if v_prime in adjacency[u] or u_prime in adjacency[v]:
                    continue
                if u_prime == v_prime:
                    continue
                
                # Perform tentative switch
                adjacency[u].discard(v)
                adjacency[u].add(v_prime)
                adjacency[v].discard(u)
                adjacency[v].add(u_prime)
                adjacency[v_prime].discard(u)
                adjacency[v_prime].add(u)
                adjacency[u_prime].discard(v)
                adjacency[u_prime].add(v)
                
                # Check if this improves girth
                new_girth = _compute_girth(adjacency, n, max_girth=target_girth + 1)
                if new_girth > current_girth:
                    current_girth = new_girth
                    improved = True
                    break
                else:
                    # Revert
                    adjacency[u].discard(v_prime)
                    adjacency[u].add(v)
                    adjacency[v].discard(u_prime)
                    adjacency[v].add(u)
                    adjacency[v_prime].discard(u)
                    adjacency[v_prime].add(v)
                    adjacency[u_prime].discard(v)
                    adjacency[u_prime].add(u)
            
            if improved:
                break
        
        if not improved:
            # Try random 2-switches
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                u1 = random.randint(0, n-1)
                u2 = random.randint(0, n-1)
                if u1 == u2:
                    continue
                
                v1_list = list(adjacency[u1])
                v2_list = list(adjacency[u2])
                if not v1_list or not v2_list:
                    continue
                
                v1 = random.choice(v1_list)
                v2 = random.choice(v2_list)
                
                # Check if switch is valid
                if v2 == u1 or v1 == u2:
                    continue
                if v2 in adjacency[u1] or v1 in adjacency[u2]:
                    continue
                
                # Perform tentative switch
                adjacency[u1].discard(v1)
                adjacency[u1].add(v2)
                adjacency[v1].discard(u1)
                adjacency[v1].add(u2)
                adjacency[u2].discard(v2)
                adjacency[u2].add(v1)
                adjacency[v2].discard(u2)
                adjacency[v2].add(u1)
                
                # Check if this improves girth
                new_girth = _compute_girth(adjacency, n, max_girth=target_girth + 1)
                if new_girth > current_girth:
                    current_girth = new_girth
                    improved = True
                    break
                else:
                    # Revert
                    adjacency[u1].discard(v2)
                    adjacency[u1].add(v1)
                    adjacency[v1].discard(u2)
                    adjacency[v1].add(u1)
                    adjacency[u2].discard(v1)
                    adjacency[u2].add(v2)
                    adjacency[v2].discard(u1)
                    adjacency[v2].add(u2)
            
            if not improved:
                # Stuck, try a few more random switches
                for _ in range(50):
                    u1 = random.randint(0, n-1)
                    u2 = random.randint(0, n-1)
                    if u1 == u2:
                        continue
                    v1_list = list(adjacency[u1])
                    v2_list = list(adjacency[u2])
                    if not v1_list or not v2_list:
                        continue
                    v1 = random.choice(v1_list)
                    v2 = random.choice(v2_list)
                    if v2 == u1 or v1 == u2:
                        continue
                    if v2 in adjacency[u1] or v1 in adjacency[u2]:
                        continue
                    adjacency[u1].discard(v1)
                    adjacency[u1].add(v2)
                    adjacency[v1].discard(u1)
                    adjacency[v1].add(u2)
                    adjacency[u2].discard(v2)
                    adjacency[u2].add(v1)
                    adjacency[v2].discard(u2)
                    adjacency[v2].add(u1)
    
    return adjacency, current_girth

def _switching_construct(k, g, n, time_limit=100.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    # Generate random k-regular graph
    adjacency = _generate_random_k_regular(n, k, time_limit=10.0)
    if adjacency is None:
        return None
    
    # Verify it's valid
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
    
    # Improve girth using switching
    adjacency, girth = _greedy_switch_improve(adjacency, n, k, g, time_limit=time_limit - 15.0)
    
    if girth < g:
        return None
    
    # Verify girth exactly
    final_girth = _compute_girth(adjacency, n, max_girth=g + 1)
    if final_girth < g:
        return None
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges

def construct(k, g):
    # For the specific case k=5, g=9, try switching approach
    if k == 5 and g == 9:
        # Try a few sizes starting from best known
        target_ns = [1116, 1120, 1150, 1200]
        
        total_time = 120.0
        time_per_attempt = total_time / len(target_ns) - 10  # Leave 10s for fallback
        
        for n in target_ns:
            if time.time() - 0 > total_time - 10:
                break
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
    
    # Fallback to LUW
    return _luw_construct(k, g)
