"""Search for small 3-regular graphs of girth 16 via random generation + local repair."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # For k=3, g=16: Moore bound is 2*(1 + (k-1) + (k-1)^2 + ... + (k-1)^7)
    # = 2*(1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 2*255 = 510
    # So n must be >= 512 (even, and divisible by 2 for 3-regular)
    # Actually for cubic graphs, n must be even. Moore bound gives 510, so n >= 512.
    # Known best is 936, lower bound 512.
    
    if k != 3 or g != 16:
        # Fallback: use the Lazebnik-Ustimenko-Woldar construction
        return _luw_construct(k, g)
    
    # Strategy: Try to construct a 3-regular graph with girth >= 16
    # Use random 3-regular graph generation and check girth
    # If girth is too small, try local modifications (edge swaps)
    
    # Start with n = 512 (the Moore bound, rounded up to even)
    # Actually, let's start a bit higher to have more room, then try to shrink
    # But for now, let's just find ANY valid graph and minimize n
    
    # Approach: 
    # 1. Generate random 3-regular graphs on n vertices
    # 2. Check if girth >= 16
    # 3. If not, try to improve via 2-switches or other local operations
    # 4. Also try to reduce n by checking if a smaller n works
    
    # For efficiency, let's use BFS-based girth checking
    # A 3-regular graph has girth >= g iff no non-backtracking walk of length < g returns to start
    
    def check_girth(adj, n, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        for u in range(n):
            # BFS tracking (vertex, previous_vertex) to avoid backtracking
            visited = set()
            queue = deque()
            # Start: go to each neighbor
            for v in adj[u]:
                visited.add((u, v))
                queue.append((v, u, 1))
            while queue:
                curr, prev, dist = queue.popleft()
                if dist >= min_girth:
                    continue
                if dist == min_girth - 1:
                    # We've reached distance min_girth-1, need to check if any edge goes back to u
                    # Actually, we need to check if we can reach u in one more step without backtracking
                    for w in adj[curr]:
                        if w == prev:
                            continue
                        if w == u:
                            return False  # Found cycle of length dist+1 < min_girth
                        if dist + 1 < min_girth:
                            if (curr, w) not in visited:
                                visited.add((curr, w))
                                queue.append((w, curr, dist + 1))
                else:
                    for w in adj[curr]:
                        if w == prev:
                            continue
                        if (curr, w) not in visited:
                            visited.add((curr, w))
                            queue.append((w, curr, dist + 1))
        return True
    
    def generate_random_3_regular(n):
        """Generate a random 3-regular graph on n vertices using configuration model."""
        # Create stubs: each vertex has 3 stubs
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        
        # Randomly pair up stubs
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        used = [False] * n
        
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                # Self-loop, retry
                return None
            if v in adj[u]:
                # Multi-edge, retry
                return None
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify degree
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        
        return adj
    
    def compute_girth(adj, n):
        """Compute the exact girth of the graph."""
        min_girth = float('inf')
        for u in range(n):
            # BFS to find shortest cycle through u
            # Track (vertex, parent) to avoid backtracking
            dist = {}
            parent = {}
            queue = deque()
            for v in adj[u]:
                dist[(u, v)] = 1
                parent[(u, v)] = None
                queue.append((v, u))
            
            while queue:
                curr, prev = queue.popleft()
                d = dist[(prev, curr)]
                if d >= min_girth:
                    continue
                for w in adj[curr]:
                    if w == prev:
                        continue
                    if w == u:
                        min_girth = min(min_girth, d + 1)
                    elif (curr, w) not in dist:
                        dist[(curr, w)] = d + 1
                        parent[(curr, w)] = (prev, curr)
                        queue.append((w, curr))
        return min_girth if min_girth < float('inf') else n  # No cycle found
    
    def local_repair(adj, n, target_girth, max_iters=1000):
        """Try to increase girth via 2-switches (edge swaps)."""
        adj = [list(x) for x in adj]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Find a short cycle
            cycle = find_short_cycle(adj, n, target_girth)
            if cycle is None:
                return adj  # Girth is good enough
            
            # Try to break the cycle via a 2-switch
            # Pick two edges from the cycle and swap them
            if len(cycle) < 4:
                break
            
            # Try random 2-switches
            improved = False
            for _ in range(50):
                # Pick two random edges from the cycle
                i1, i2 = random.sample(range(len(cycle)), 2)
                u1, v1 = cycle[i1], cycle[(i1+1) % len(cycle)]
                u2, v2 = cycle[i2], cycle[(i2+1) % len(cycle)]
                
                # Check if we can swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
                # or (u1,v2) and (v1,u2)
                for new_edges in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
                    (a, b), (c, d) = new_edges
                    if a == b or c == d:
                        continue
                    # Check no multi-edges
                    if b in adj[a] or d in adj[c]:
                        continue
                    if a in adj[b] or c in adj[d]:
                        continue
                    
                    # Perform swap
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    adj[a].append(c)
                    adj[c].append(a)
                    adj[b].append(d)
                    adj[d].append(b)
                    
                    # Check if girth improved
                    new_girth = compute_girth(adj, n)
                    if new_girth > target_girth or (cycle is not None and new_girth >= len(cycle)):
                        improved = True
                        break
                    else:
                        # Revert
                        adj[a].remove(c)
                        adj[c].remove(a)
                        adj[b].remove(d)
                        adj[d].remove(b)
                        adj[a].append(b)
                        adj[b].append(a)
                        adj[c].append(d)
                        adj[d].append(c)
                
                if improved:
                    break
            
            if not improved:
                # Try a random 2-switch anywhere
                for _ in range(20):
                    u = random.randint(0, n-1)
                    v = random.choice(adj[u])
                    w = random.randint(0, n-1)
                    if w == u or w == v:
                        continue
                    x = random.choice(adj[w])
                    if x == u or x == v:
                        continue
                    # Swap (u,v) and (w,x) -> (u,w) and (v,x) or (u,x) and (v,w)
                    for new_edges in [((u, w), (v, x)), ((u, x), (v, w))]:
                        (a, b), (c, d) = new_edges
                        if a == b or c == d:
                            continue
                        if b in adj[a] or d in adj[c]:
                            continue
                        if a in adj[b] or c in adj[d]:
                            continue
                        
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].remove(d)
                        adj[d].remove(c)
                        adj[a].append(c)
                        adj[c].append(a)
                        adj[b].append(d)
                        adj[d].append(b)
                        
                        new_girth = compute_girth(adj, n)
                        if new_girth >= target_girth:
                            improved = True
                            break
                        else:
                            # Revert
                            adj[a].remove(c)
                            adj[c].remove(a)
                            adj[b].remove(d)
                            adj[d].remove(b)
                            adj[a].append(b)
                            adj[b].append(a)
                            adj[c].append(d)
                            adj[d].append(c)
                        
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return adj
    
    def find_short_cycle(adj, n, target_girth):
        """Find a cycle shorter than target_girth."""
        for u in range(n):
            # BFS
            visited = {}
            queue = deque()
            for v in adj[u]:
                visited[(u, v)] = (None, None, 1)
                queue.append((v, u))
            
            while queue:
                curr, prev = queue.popleft()
                d = visited[(prev, curr)][2]
                if d >= target_girth:
                    continue
                for w in adj[curr]:
                    if w == prev:
                        continue
                    if w == u:
                        # Found cycle of length d+1
                        if d + 1 < target_girth:
                            # Reconstruct cycle
                            cycle = []
                            node = (prev, curr)
                            while node is not None:
                                p, c = node
                                cycle.append(c)
                                node = visited[node][0]
                            cycle.append(u)
                            cycle.reverse()
                            return cycle
                    elif (curr, w) not in visited:
                        visited[(curr, w)] = (visited[(prev, curr)][0], (prev, curr), d + 1)
                        queue.append((w, curr))
        return None
    
    # Main search
    best_adj = None
    best_n = None
    
    # Try increasing n from Moore bound
    # Moore bound for (3,16): 2*sum_{i=0}^{7} 2^i = 2*(2^8 - 1) = 510
    # So n >= 512
    
    # Start with larger n where random graphs are more likely to have good girth
    # Then try to shrink
    
    # For cubic graphs, girth grows slowly with n
    # A random cubic graph on n vertices has girth ~ (1/2) log_2(n)
    # For girth 16, we need roughly 2^32 vertices for random, which is too large
    # So we need a smarter approach
    
    # Alternative: Use the known construction from the literature
    # The smallest known (3,16)-cage has 936 vertices
    
    # Let's try to construct a graph with n close to 936
    # Using a voltage graph approach or a known construction
    
    # For now, let's try random generation with local repair
    # Start with n = 936 (known to work)
    
    n_target = 936
    found = False
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
        
        adj = generate_random_3_regular(n_target)
        if adj is None:
            continue
        
        girth = compute_girth(adj, n_target)
        if girth >= 16:
            best_adj = adj
            best_n = n_target
            found = True
            break
        
        # Try local repair
        repaired = local_repair(adj, n_target, 16, max_iters=100)
        girth2 = compute_girth(repaired, n_target)
        if girth2 >= 16:
            best_adj = repaired
            best_n = n_target
            found = True
            break
    
    if not found:
        # Fallback to LUW construction
        return _luw_construct(3, 16)
    
    # Convert adjacency list to edge list
    edges = []
    for u in range(best_n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    return best_n, edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (fallback)."""
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
# EVOLVE-BLOCK-END
