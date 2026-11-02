import time
import random
import math

def construct(k, g):
    start_time = time.time()
    DEADLINE = start_time + 135.0  # 15s margin for safety

    # If not the specific target case, use L-U-W which is valid for all
    if not (k == 3 and g == 16):
        return _luw_construct(k, g)

    # Moore bound for k=3, g=16 is 1 + 3*(1 + 2 + 4 + 8 + 16 + 32 + 64) = 1 + 3*127 = 382?
    # Wait, Moore bound for even girth g:
    # n >= k * (k-1)^(g/2 - 1) / 2 ?
    # For k=3, g=16:
    # n >= 3 * 2^7 / 2 = 3 * 64 = 192?
    # Let's check the formula.
    # For even girth g, the Moore bound is:
    # n >= k * (k-1)^(g/2 - 1) / 2 ?
    # Actually, for k-regular graph of even girth g:
    # n >= k * (k-1)^(g/2 - 1) / 2  is for trees?
    # The Moore bound is:
    # n >= 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i  (for odd girth?)
    # For even girth g, the bound is derived from the bipartite double cover?
    # Standard Moore Bound for k-regular, girth g:
    # If g is even: n >= k * (k-1)^(g/2 - 1) / 2  ... No.
    # Let's use the standard bound:
    # n >= k * (k-1)^(g/2 - 1) / 2 is for the number of vertices at distance g/2?
    # Actually, the bound is:
    # n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i for odd g.
    # For even g, n >= k * (k-1)^(g/2 - 1) / 2 is not the full bound.
    # The bound is: n >= k * (k-1)^(g/2 - 1) / 2 * 2?
    # Let's just use a safe lower bound.
    # For k=3, g=16, known lower bound is 512.
    
    moore_bound = 512
    
    # We will try to find a graph with n close to the known best 936, 
    # or even smaller if possible.
    # The random search might struggle to find girth 16 for n < 600.
    # Let's try n from 600 to 1000.
    
    best_n = None
    best_edges = None
    
    # Try specific sizes
    target_ns = [600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800]
    
    for n in target_ns:
        if time.time() > DEADLINE:
            break
            
        # Try a few random instances
        for attempt in range(5):
            if time.time() > DEADLINE:
                break
                
            edges = _generate_random_cubic_graph(n)
            if not edges:
                continue
                
            # Check girth
            girth = _get_girth(n, edges)
            
            if girth >= g:
                return n, edges
            
            # If girth is close, try to improve
            if girth >= g - 4:
                edges = _improve_girth(n, edges, g, max_steps=20, time_limit=DEADLINE)
                if edges:
                    girth = _get_girth(n, edges)
                    if girth >= g:
                        return n, edges
    
    # Fallback to L-U-W
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
    q = k
    # d is the parameter for the construction.
    # Girth is at least d + 5.
    # We need girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # d must be odd for the standard construction to work nicely?
    # The code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _generate_random_cubic_graph(n):
    """
    Generate a random simple cubic graph using configuration model with repair.
    """
    if n < 4 or n % 2 != 0: # Cubic graphs must have even number of vertices
        return None
        
    # Stubs
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    
    random.shuffle(stubs)
    
    # Pair stubs
    adj = [[] for _ in range(n)]
    edges = []
    
    # We need to resolve self-loops and multi-edges.
    # Simple approach: Retry pairing if conflict.
    # More efficient: Local repair.
    
    # For simplicity and speed in this context, we use a rejection sampling 
    # with a limit, or a local repair.
    
    # Let's try a simple local repair.
    # 1. Randomly pair.
    # 2. Check for loops/multi-edges.
    # 3. If found, swap stubs to fix.
    
    # This is complex to implement correctly in short time.
    # Alternative: Use a known algorithm for random regular graphs.
    # "Random Cubic Graph" can be generated by:
    # 1. Create 2n/3 edges? No.
    
    # Let's use a simpler method:
    # Start with a cycle (0-1-2...-n-1).
    # Add n/2 chords? No, that changes degree.
    
    # Best bet: 
    # 1. Shuffle stubs.
    # 2. Pair them.
    # 3. If self-loop or multi-edge, swap one of the stubs with a random other stub.
    
    pairs = [(stubs[i], stubs[i+1]) for i in range(0, len(stubs), 2)]
    
    # Convert to adj list and check validity
    # We'll represent edges as a set of frozensets for checking multi-edges
    edge_set = set()
    valid = True
    
    for u, v in pairs:
        if u == v:
            # Self loop
            # Find another stub to swap with
            # Pick a random pair index j != i
            # Swap v with one of the stubs in pair j
            # This is getting complicated.
            
            # Let's just reject and restart if we find a loop/multi-edge 
            # and we haven't done too many attempts.
            valid = False
            break
        if (v, u) in edge_set:
            valid = False
            break
        edge_set.add((u, v))
    
    if not valid:
        return None
        
    # Build adj
    for u, v in pairs:
        adj[u].append(v)
        adj[v].append(u)
        
    return pairs

def _get_girth(n, edges):
    """
    Calculate the girth of a graph.
    Returns infinity if no cycle (tree), but cubic graphs have cycles.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # But we need the shortest one.
                    # This BFS finds the shortest cycle passing through start?
                    # No, standard BFS girth check:
                    # If we visit a node already visited, and it's not the parent, 
                    # we have a cycle.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth == 3:
                        return 3
                        
    return min_girth

def _improve_girth(n, edges, target_girth, max_steps, time_limit):
    """
    Try to increase girth using 2-switches.
    """
    # This is a placeholder for a more complex optimization.
    # Given time constraints, we might skip this if _get_girth is slow.
    # For now, return None to indicate no improvement.
    return None
