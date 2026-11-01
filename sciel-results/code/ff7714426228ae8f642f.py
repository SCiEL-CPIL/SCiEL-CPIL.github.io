import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a hybrid approach:
    1. For specific small parameters (k=3, g=16), attempts a randomized local search 
       to find a graph smaller than the known cage (936 vertices).
    2. Falls back to the Lazebnik-Ustimenko-Woldar (L-U-W) construction if the 
       search fails or for other parameters.
    """
    # Handle the specific case of the task: k=3, g=16
    if k == 3 and g == 16:
        # Time budget: finish well under 120s, leave room for fallback.
        start_time = time.time()
        timeout = 110.0

        # Try increasing N from a reasonable lower bound up to just below known cage.
        # 3-regular graphs need even N.
        for N in range(512, 936, 2):
            if time.time() - start_time > timeout:
                break

            # More attempts for larger N where random graphs have better girth.
            # But keep per-attempt cost low.
            attempts = 40 if N < 700 else 25
            for attempt in range(attempts):
                if time.time() - start_time > timeout:
                    break

                edges = generate_random_3_regular(N)
                if edges is None:
                    continue

                # Fast check: if girth >= 16, return.
                girth = compute_girth(N, edges, target=16)
                if girth >= 16:
                    return N, edges

                # If girth is close, try local repair.
                if girth >= 14:
                    repaired_edges = local_repair(N, edges, target_girth=16, time_limit=0.8)
                    if repaired_edges:
                        final_girth = compute_girth(N, repaired_edges, target=16)
                        if final_girth >= 16:
                            return N, repaired_edges

        # If random search failed, fall back to L-U-W
        return luw_construct(3, 16)
    else:
        # General case: use L-U-W construction
        return luw_construct(k, g)

def generate_random_3_regular(N):
    """Generates a random 3-regular graph on N vertices."""
    if N % 2 != 0:
        return None

    # Configuration model with rejection for loops/multi-edges.
    # For moderate N, this is fast.
    stubs = [i for i in range(N) for _ in range(3)]
    random.shuffle(stubs)

    edges = []
    seen = set()
    for i in range(0, N * 3, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if (u, v) in seen or (v, u) in seen:
            return None
        seen.add((u, v))
        seen.add((v, u))
        edges.append((u, v))

    return edges

def compute_girth(N, edges, target=None):
    """
    Computes the girth of the graph using BFS.
    If target is given, we can stop early if we find a cycle < target.
    Returns the girth (or N if none found, i.e., tree, but 3-reg is connected cycles exist).
    """
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(N):
        # BFS from start
        visited = [-1] * N
        parent = [-1] * N
        visited[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # In undirected BFS, an edge to a visited node (not parent) indicates a cycle.
                    # The cycle length is depth[u] + depth[v] + 1.
                    cycle_len = visited[u] + visited[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if target is not None and min_girth < target:
                            # We only care if girth < target. If we found one, return early.
                            return min_girth
        
        # Early exit if we already found a very small cycle
        if target is not None and min_girth < target:
            return min_girth

    return min_girth if min_girth != float('inf') else N

def local_repair(N, edges, target_girth, time_limit):
    """
    Attempts to increase the girth by breaking short cycles via 2-switches (edge swaps).
    Returns new edges if successful, else None.
    """
    start_time = time.time()
    
    # Build adjacency and edge set for quick lookup
    adj = [[] for _ in range(N)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
    
    # We want to break cycles of length < target_girth.
    # Strategy: Find a short cycle, then try to swap an edge in it with a non-adjacent edge to break the cycle without creating shorter ones.
    # A simpler heuristic: Try random 2-switches and check if girth improves.
    
    # To be efficient, we only do this if girth is close (14 or 15).
    # Let's try a limited number of random swaps.
    
    current_girth = compute_girth(N, edges, target=target_girth)
    if current_girth >= target_girth:
        return edges
    
    # Try up to 50 random swaps
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random edge (u, v)
        u, v = random.choice(edges)
        
        # Find a neighbor w of u (w != v) and a neighbor x of v (x != u)
        # such that (w, x) is not an edge and w != x.
        # Then swap (u,v) and (w,x) with (u,x) and (w,v).
        
        candidates = []
        for w in adj[u]:
            if w == v:
                continue
            for x in adj[v]:
                if x == u or x == w:
                    continue
                if (min(w, x), max(w, x)) not in edge_set:
                    candidates.append((w, x))
        
        if not candidates:
            continue
        
        w, x = random.choice(candidates)
        
        # Perform swap
        new_edges = []
        for e in edges:
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                continue
            new_edges.append(e)
        new_edges.append((u, x))
        new_edges.append((w, v))
        
        # Check girth
        new_girth = compute_girth(N, new_edges, target=target_girth)
        if new_girth >= target_girth:
            return new_edges
        elif new_girth > current_girth:
            # Accept if it improves girth, even if not yet target
            edges = new_edges
            current_girth = new_girth
            # Update adj and edge_set for next iteration
            adj = [[] for _ in range(N)]
            edge_set = set()
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
                edge_set.add((min(a, b), max(a, b)))
            
    return None

def luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    """
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
