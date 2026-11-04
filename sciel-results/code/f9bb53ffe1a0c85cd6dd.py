import time
import random
from collections import deque

def _moore_bound(k, g):
    if k <= 1:
        return 0
    if g % 2 == 1:
        # Odd girth
        # n >= 2 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        m = (g - 3) // 2
        s = 0
        term = 1
        for i in range(m + 1):
            s += term
            term *= (k - 1)
        return 2 + k * s
    else:
        # Even girth
        # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        m = g // 2 - 1
        s = 0
        term = 1
        for i in range(m + 1):
            s += term
            term *= (k - 1)
        return 2 * s

def _seed_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
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

def _generate_random_k_regular(n, k):
    """
    Generates a random k-regular simple graph on n vertices.
    Returns adjacency list (list of lists) or None if failed.
    """
    if n < k:
        return None
    if (n * k) % 2 != 0:
        return None
    
    # Configuration model
    # Each vertex has k stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Try to pair stubs to form a simple graph
    for _ in range(50): # 50 retries
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edge_set or (v, u) in edge_set:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))
            edge_set.add((v, u))
        if valid:
            return adj
    return None

def _check_girth_limited(adj, g):
    """
    Checks if girth is at least g.
    Returns (True, None) if girth >= g.
    Returns (False, cycle) if girth < g, where cycle is a list of nodes.
    Uses BFS limited to depth D = (g-1)//2.
    """
    n = len(adj)
    D = (g - 1) // 2
    if D == 0:
        return True, None
        
    # Try a few random starts to find a short cycle quickly
    # If we don't find one in limited BFS from random nodes, it's likely large
    # But to be rigorous, we should check all? 
    # For n=500, checking all is fast enough if D is small.
    # Let's check all nodes to be safe, but break early if found.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= D:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < D: # Only enqueue if we can go deeper
                        q.append(v)
                elif v != parent[u]:
                    # Found a back edge
                    # Cycle length = dist[u] + dist[v] + 1
                    # But wait, if dist[v] > D, we wouldn't have processed u's neighbors?
                    # We only process u if dist[u] < D.
                    # If dist[v] is large, the cycle is long.
                    # We only care if cycle < g.
                    # Cycle length L. If L < g, then the meeting point depth is < D.
                    # Specifically, if L < g, then min(dist[u], dist[v]) <= (L-1)/2 < (g-1)/2 = D.
                    # So at least one of u,v has dist < D.
                    # Since we process u (dist[u] < D), if v is visited, dist[v] is set.
                    # If dist[v] + dist[u] + 1 < g, we found a short cycle.
                    
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [u]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cycle.append(cur)
                        # Append path from v to LCA?
                        # Actually, the cycle is u -> ... -> LCA -> ... -> v -> u?
                        # No, in BFS tree, parent pointers go to root.
                        # The cycle is u -> parent[u] -> ... -> LCA -> ... -> parent[v] -> v -> u (back edge).
                        # We just need the nodes to pick an edge.
                        # Let's just return the edge (u, v) as part of the cycle.
                        # The local search can just use the edge (u, v).
                        return False, (u, v, cycle_len)
    return True, None

def _local_search(adj, k, g, max_swaps=2000):
    """
    Attempts to increase girth to at least g via edge swaps.
    Returns True if successful, False otherwise.
    Modifies adj in place.
    """
    n = len(adj)
    # Maintain edge set for O(1) lookup
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
                
    # Pre-compute neighbors sets for faster swap validation?
    # Adj is list of lists. Converting to sets is O(E).
    adj_sets = [set(neigh) for neigh in adj]
    
    for _ in range(max_swaps):
        ok, info = _check_girth_limited(adj, g)
        if ok:
            return True
            
        u, v, cycle_len = info
        # Pick a random edge from the cycle?
        # We only have (u, v) from the check. 
        # (u, v) is the back edge. The rest of the cycle is in the tree.
        # Swapping the back edge (u, v) is a good candidate.
        
        # Find a swap partner for (u, v)
        # Swap (u, v) with (w, x) -> (u, w) and (v, x)
        # Conditions:
        # 1. w, x distinct from u, v.
        # 2. (u, w) not in edge_set, (v, x) not in edge_set.
        # 3. (u, x) not in edge_set, (v, w) not in edge_set (to avoid creating multi-edges in the other config? 
        #    Actually, we are replacing (u,v) and (w,x).
        #    New edges: (u,w) and (v,x).
        #    We must ensure (u,w) != (v,x) and no loops.
        #    Also ensure (u,w) and (v,x) don't already exist.
        #    And that we don't create parallel edges.
        
        attempts = 0
        while attempts < 100:
            attempts += 1
            w = random.randrange(n)
            x = random.randrange(n)
            if w == u or w == v or x == u or x == v:
                continue
            if w == x:
                continue
            # Check if (w, x) is an edge
            if (w, x) not in edge_set and (x, w) not in edge_set:
                continue
            
            # Check if new edges are valid
            if (u, w) in edge_set or (w, u) in edge_set:
                continue
            if (v, x) in edge_set or (x, v) in edge_set:
                continue
            if (u, x) in edge_set or (x, u) in edge_set:
                # This would mean we create (u,x) which exists? No, we are creating (u,w) and (v,x).
                # We just need to ensure (u,w) and (v,x) are new.
                # We also need to ensure we don't create multi-edges.
                # Since graph is simple, if (u,w) doesn't exist, it's fine.
                pass
            if (v, w) in edge_set or (w, v) in edge_set:
                # This check is not strictly necessary for the new edges, 
                # but if (v,w) exists, and we add (v,x), it's fine.
                # The only constraint is that new edges don't exist.
                pass
            
            # Perform swap
            # Remove (u, v) and (w, x)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj_sets[u].remove(v)
            adj_sets[v].remove(u)
            adj_sets[w].remove(x)
            adj_sets[x].remove(w)
            
            edge_set.discard((u, v))
            edge_set.discard((v, u))
            edge_set.discard((w, x))
            edge_set.discard((x, w))
            
            # Add (u, w) and (v, x)
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            adj_sets[u].add(w)
            adj_sets[w].add(u)
            adj_sets[v].add(x)
            adj_sets[x].add(v)
            
            edge_set.add((u, w))
            edge_set.add((w, u))
            edge_set.add((v, x))
            edge_set.add((x, v))
            
            break # Swap done, break while loop
        else:
            # Could not find a valid swap, restart local search?
            # Or just break and return False
            break
            
    return False

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_min = _moore_bound(k, g)
    # Ensure n_min is even? No, but for k-regular, n*k must be even.
    # If n_min * k is odd, n_min must be incremented.
    if (n_min * k) % 2 != 0:
        n_min += 1
        
    # Search range
    # We try increasing n.
    # Limit the search to reasonable bounds to avoid infinite loop if no solution found quickly.
    # For k=5, g=9, n_min=428. Solution is likely < 1000.
    max_n = n_min + 1000 
    
    for n in range(n_min, max_n):
        if time.time() > deadline:
            break
        # Try a few random restarts for this n
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            adj = _generate_random_k_regular(n, k)
            if adj is None:
                continue
                
            # Local search
            if _local_search(adj, k, g, max_swaps=1000):
                # Convert adj to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    # Fallback to seed
    return _seed_construct(k, g)
