import random
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Fallback: Groetzsch graph
    groetzsch_edges = []
    for i in range(5):
        groetzsch_edges.append([i, (i + 1) % 5])
    for i in range(5):
        u = i
        w = (i + 1) % 5
        groetzsch_edges.append([u, 5 + w])
        groetzsch_edges.append([w, 5 + u])
    for i in range(5):
        groetzsch_edges.append([5 + i, 10])
        
    best_edges = len(groetzsch_edges)
    best_graph = groetzsch_edges
    
    def is_3_colorable(adj):
        color = [-1] * n
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(i):
            if i == n:
                return True
            v = order[i]
            for c in range(3):
                conflict = False
                nb = adj[v]
                while nb:
                    lsb = nb & -nb
                    u = lsb.bit_length() - 1
                    if color[u] == c:
                        conflict = True
                        break
                    nb ^= lsb
                if not conflict:
                    color[v] = c
                    if backtrack(i + 1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack(0)

    def generate_random_maximal_tf():
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        adj = [0] * n
        count = 0
        for u, v in edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        return adj, count

    def local_search(adj, count):
        # Try to improve edge count by swapping edges
        # This is a simple hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pick a random edge to remove
            edges_in_adj = [(u, v) for u in range(n) for v in range(u+1, n) if (adj[u] & (1 << v))]
            if not edges_in_adj:
                break
            u, v = random.choice(edges_in_adj)
            # Remove edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            count -= 1
            
            # Try to add edges to make it maximal again
            # We can just re-run the maximalization on the remaining edges?
            # Or try to add random edges.
            # Let's try to add edges that don't create triangles
            added = 0
            # Shuffle all possible edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for x, y in all_edges:
                if not (adj[x] & (1 << y)): # Not already present
                    if (adj[x] & adj[y]) == 0:
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        count += 1
                        added += 1
                        if count > best_edges and not is_3_colorable(adj):
                            # Found better
                            best_edges = count
                            best_graph = [[a, b] for a in range(n) for b in range(a+1, n) if (adj[a] & (1 << b))]
                            improved = True
                            # Break out? Or continue?
                            # Let's break to restart from this new best
                            return adj, count
            # If we didn't find a better one, we are stuck in a local optimum
            # This local search is expensive. Maybe skip it?
            pass
        return adj, count

    # Main loop
    while time.time() < deadline:
        adj, count = generate_random_maximal_tf()
        
        if not is_3_colorable(adj):
            if count > best_edges:
                best_edges = count
                best_graph = [[u, v] for u in range(n) for v in range(u+1, n) if (adj[u] & (1 << v))]
                
        # Optional: Local search
        # If count is close to best, try local search
        # But local search is slow.
        # Let's just do random restarts.

    return best_graph
