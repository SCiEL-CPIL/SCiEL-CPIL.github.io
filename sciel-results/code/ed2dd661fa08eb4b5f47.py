import time
import random

def construct():
    start_time = time.time()
    deadline = 100.0
    
    n = 20
    
    def count_bad_pairs(G):
        bad = 0
        for u in range(n):
            for v in range(u + 1, n):
                if len(G[u] & G[v]) >= 3:
                    bad += 1
        return bad

    # Helper to create circulant C_n(1, 2, 3)
    def create_circulant():
        G = [set() for _ in range(n)]
        for i in range(n):
            for k in [1, 2, 3]:
                G[i].add((i + k) % n)
                G[i].add((i - k) % n)
        return G

    # Local search for regular graph
    def local_search(G, max_time):
        current_bad = count_bad_pairs(G)
        best_graph = [s.copy() for s in G]
        best_bad = current_bad
        
        # Generate edge list for swaps
        def get_edges():
            edges = []
            for u in range(n):
                for v in G[u]:
                    if u < v:
                        edges.append((u, v))
            return edges

        while time.time() - start_time < max_time:
            if current_bad == 0:
                break
                
            edges = get_edges()
            if len(edges) < 2:
                break
                
            improved = False
            for _ in range(50): # 50 attempts per iteration
                e1, e2 = random.sample(edges, 2)
                u, v = e1
                x, y = e2
                
                if len({u, v, x, y}) < 4:
                    continue
                
                if x in G[u] or y in G[v]:
                    continue
                
                # Apply swap
                G[u].remove(v)
                G[v].remove(u)
                G[x].remove(y)
                G[y].remove(x)
                
                G[u].add(x)
                G[x].add(u)
                G[v].add(y)
                G[y].add(v)
                
                new_bad = count_bad_pairs(G)
                
                if new_bad < current_bad:
                    current_bad = new_bad
                    improved = True
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_graph = [s.copy() for s in G]
                else:
                    # Revert
                    G[u].remove(x)
                    G[x].remove(u)
                    G[v].remove(y)
                    G[y].remove(v)
                    
                    G[u].add(v)
                    G[v].add(u)
                    G[x].add(y)
                    G[y].add(x)
            
            if not improved and current_bad == best_bad:
                # Stuck, try random restart? Or just break?
                # For small n, break is fine, we have fallback.
                pass

        return best_graph, best_bad

    # Attempt 1: 6-regular
    G6 = create_circulant()
    best_G6, bad6 = local_search(G6, 40.0)
    
    if bad6 == 0:
        return [[u, v] for u in range(n) for v in best_G6[u] if u < v]
        
    # Attempt 2: 5-regular
    # Take C_20(1,2,3), find perfect matching, remove it.
    G5 = create_circulant()
    # Find perfect matching in G5
    # Greedy matching
    matched = [False] * n
    matching = []
    for u in range(n):
        if matched[u]:
            continue
        for v in G5[u]:
            if not matched[v]:
                matching.append((u, v))
                matched[u] = True
                matched[v] = True
                break
    
    # Remove matching edges
    for u, v in matching:
        G5[u].remove(v)
        G5[v].remove(u)
        
    best_G5, bad5 = local_search(G5, 40.0)
    
    if bad5 == 0:
        return [[u, v] for u in range(n) for v in best_G5[u] if u < v]
        
    # Fallback: Seed
    edges = []
    for k in range(5):
        edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
    return edges
