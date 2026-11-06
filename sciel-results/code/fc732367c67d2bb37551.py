import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 145  # Safety margin
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Adjacency as list of sets
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
    
    def has_c6_with_edge(u, v):
        """Check if adding edge (u,v) would create a C6.
        A C6 through (u,v): u-v-x-y-z-w-u
        Need path of length 4 from v to u: v-x-y-z-w-u
        """
        # For each x in N(v):
        for x in adj[v]:
            if x == u:
                continue
            # For each y in N(x), y != u, y != v
            for y in adj[x]:
                if y == u or y == v or y == x:
                    continue
                # For each z in N(y), z != u, z != v, z != x
                for z in adj[y]:
                    if z == u or z == v or z == x or z == y:
                        continue
                    # Check if there exists w in N(z) ∩ N(u), w not in {u,v,x,y}
                    for w in adj[z]:
                        if w == u or w == v or w == x or w == y or w == z:
                            continue
                        if w in adj[u]:
                            return True
        return False
    
    def is_c6_free():
        """Verify the entire graph is C6-free."""
        edge_list = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edge_list.append((i, j))
        for u, v in edge_list:
            # Temporarily remove edge
            adj[u].discard(v)
            adj[v].discard(u)
            result = has_c6_with_edge(u, v)
            # Restore edge
            adj[u].add(v)
            adj[v].add(u)
            if result:
                return False
        return True
    
    def edge_count():
        return sum(len(adj[i]) for i in range(n)) // 2
    
    # Phase 1: Greedy construction
    random.seed(42)
    
    # Try multiple random orderings, keep best
    best_edges = 0
    best_adj = None
    
    for trial in range(50):
        if time_left() < 5:
            break
        
        adj2 = [set() for _ in range(n)]
        
        # Random permutation of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time_left() < 5:
                break
            # Check if adding (u,v) creates C6
            # We need to check in the context of adj2
            # Temporarily use adj2 for the check
            # We'll inline the check for speed
            creates_c6 = False
            for x in adj2[v]:
                if x == u:
                    continue
                for y in adj2[x]:
                    if y == u or y == v or y == x:
                        continue
                    for z in adj2[y]:
                        if z == u or z == v or z == x or z == y:
                            continue
                        for w in adj2[z]:
                            if w == u or w == v or w == x or w == y or w == z:
                                continue
                            if w in adj2[u]:
                                creates_c6 = True
                                break
                        if creates_c6:
                            break
                    if creates_c6:
                        break
                if creates_c6:
                    break
            if not creates_c6:
                adj2[u].add(v)
                adj2[v].add(u)
        
        ec = sum(len(adj2[i]) for i in range(n)) // 2
        if ec > best_edges:
            best_edges = ec
            best_adj = [set(s) for s in adj2]
    
    # Use best found so far
    if best_adj is None:
        # Fallback: 8 copies of K5
        adj = [set() for _ in range(n)]
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    adj[5*k+a].add(5*k+b)
                    adj[5*k+b].add(5*k+a)
    else:
        adj = best_adj
    
    # Phase 2: Local search - try to add more edges
    for trial in range(200):
        if time_left() < 5:
            break
        
        improved = False
        for u in range(n):
            if time_left() < 5:
                break
            for v in range(u+1, n):
                if time_left() < 5:
                    break
                if v in adj[u]:
                    continue
                # Check if we can add (u,v)
                creates_c6 = False
                for x in adj[v]:
                    if x == u:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == x:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == x or z == y:
                                continue
                            for w in adj[z]:
                                if w == u or w == v or w == x or w == y or w == z:
                                    continue
                                if w in adj[u]:
                                    creates_c6 = True
                                    break
                            if creates_c6:
                                break
                        if creates_c6:
                            break
                    if creates_c6:
                        break
                if not creates_c6:
                    adj[u].add(v)
                    adj[v].add(u)
                    improved = True
        
        if not improved:
            break
    
    # Phase 3: Edge swap local search
    for trial in range(100):
        if time_left() < 10:
            break
        
        current_ec = edge_count()
        improved = False
        
        # Try removing one edge and adding another
        for u in range(n):
            if time_left() < 10:
                break
            for v in list(adj[u]):
                if time_left() < 10:
                    break
                if v < u:
                    continue
                # Try removing (u,v)
                adj[u].discard(v)
                adj[v].discard(u)
                
                # Try adding all possible edges
                for a in range(n):
                    if time_left() < 10:
                        break
                    for b in range(a+1, n):
                        if time_left() < 10:
                            break
                        if b in adj[a]:
                            continue
                        # Check if adding (a,b) creates C6
                        creates_c6 = False
                        for x in adj[b]:
                            if x == a:
                                continue
                            for y in adj[x]:
                                if y == a or y == b or y == x:
                                    continue
                                for z in adj[y]:
                                    if z == a or z == b or z == x or z == y:
                                        continue
                                    for w in adj[z]:
                                        if w == a or w == b or w == x or w == y or w == z:
                                            continue
                                        if w in adj[a]:
                                            creates_c6 = True
                                            break
                                    if creates_c6:
                                        break
                                if creates_c6:
                                    break
                            if creates_c6:
                                break
                        if not creates_c6:
                            adj[a].add(b)
                            adj[b].add(a)
                            new_ec = edge_count()
                            if new_ec > current_ec:
                                improved = True
                            else:
                                # Revert
                                adj[a].discard(b)
                                adj[b].discard(a)
                            # Restore (u,v)
                            adj[u].add(v)
                            adj[v].add(u)
                            break
                    if improved:
                        break
                if improved:
                    break
                # Restore (u,v)
                adj[u].add(v)
                adj[v].add(u)
            if improved:
                break
        
        if not improved:
            break
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    
    return edges
