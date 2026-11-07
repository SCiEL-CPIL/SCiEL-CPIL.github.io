import time
import random

def _check_valid_fast(adj, n, deg):
    """Check if graph satisfies constraint. adj is list of sets, deg is list of degrees."""
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u:
                        continue
                    if s == t:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 15
    best_adj = None
    best_count = 0
    
    # Helper to count edges
    def count_edges(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    # Strategy: Greedy add with different orderings
    # Try natural order, reverse order, and random orders
    
    orders = []
    # Natural
    orders.append([(u, v) for u in range(n) for v in range(u+1, n)])
    # Reverse
    orders.append(list(reversed([(u, v) for u in range(n) for v in range(u+1, n)])))
    # Several random orders
    for _ in range(50):
        if time.time() > deadline:
            break
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        orders.append(edges)
    
    for order in orders:
        if time.time() > deadline:
            break
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in order:
            # Try adding edge
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            if _check_valid_fast(adj, n, deg):
                pass
            else:
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
    
    # Try local search on the best solution
    if best_adj is not None:
        adj = [set(s) for s in best_adj]
        deg = [len(adj[i]) for i in range(n)]
        
        # Try to add back removed edges
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u+1, n):
                    if v in adj[u]:
                        continue
                    # Try adding edge (u,v)
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    if _check_valid_fast(adj, n, deg):
                        improved = True
                    else:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        deg[u] -= 1
                        deg[v] -= 1
            if not improved:
                break
    
    if best_adj is None:
        best_adj = [set() for _ in range(n)]
        for v in range(1, n):
            best_adj[0].add(v)
            best_adj[v].add(0)
    
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([int(u), int(v)])
    
    return result
