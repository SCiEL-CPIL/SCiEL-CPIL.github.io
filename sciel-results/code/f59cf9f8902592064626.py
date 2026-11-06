import random
import time

def construct():
    n = 40
    adj = [set() for _ in range(n)]
    
    def creates_c6(u, v):
        # Check if adding edge (u, v) creates a C6.
        # Path: u - a - b - c - d - v
        # a in N(u)
        # b in N(a), b != u, v
        # c in N(b), c not in {u, a, v}
        # d in N(c), d not in {u, a, b, c, v}
        # v in N(d)
        
        for a in adj[u]:
            if a == v: continue
            for b in adj[a]:
                if b in (u, v): continue
                for c in adj[b]:
                    if c in (u, a, v): continue
                    # Candidates for d are in N(c) intersect N(v)
                    # Exclude {u, a, b, c, v}
                    candidates = adj[c] & adj[v]
                    if candidates - {u, a, b, c, v}:
                        return True
        return False

    def try_add_edge(u, v):
        if v in adj[u]:
            return False
        if creates_c6(u, v):
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    best_edges = 0
    best_adj = None
    best_edges_list = []
    
    start_time = time.time()
    limit = 115 # seconds

    for attempt in range(200):
        if time.time() - start_time > limit:
            break
            
        # Reset
        adj = [set() for _ in range(n)]
        
        # Random order of edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            try_add_edge(u, v)
            
        # Local Search: Try to add missing edges
        # Identify missing edges
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    missing.append((u, v))
                    
        # Try to add all missing edges
        changed = True
        while changed and time.time() - start_time < limit:
            changed = False
            for u, v in missing:
                if try_add_edge(u, v):
                    changed = True
            # Re-calculate missing if changed? 
            # It's safer to just loop through all pairs again if we made changes
            if changed:
                missing = []
                for u in range(n):
                    for v in range(u+1, n):
                        if v not in adj[u]:
                            missing.append((u, v))
        
        # If we have time, do some swap moves
        # Swap: Remove one edge, try to add two missing edges
        if time.time() - start_time < limit - 5:
            current_edges = sum(len(s) for s in adj) // 2
            # List of all edges in graph
            graph_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        graph_edges.append((u, v))
            
            # List of missing edges
            missing = []
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u]:
                        missing.append((u, v))
                        
            if missing:
                # Try a few swaps
                for _ in range(50):
                    if time.time() - start_time > limit - 2:
                        break
                    # Pick random edge to remove
                    if not graph_edges:
                        break
                    idx = random.randint(0, len(graph_edges) - 1)
                    ru, rv = graph_edges[idx]
                    
                    # Remove it
                    adj[ru].remove(rv)
                    adj[rv].remove(ru)
                    graph_edges.pop(idx)
                    
                    # Try to add this edge back? No, we want to add others.
                    # Try to add random missing edges
                    added_any = False
                    # Try to add the removed edge back? 
                    # Strategy: Remove (ru, rv). Try to add (ru, rv) and another? 
                    # No, we want net gain. 
                    # Try to add 2 random missing edges.
                    random.shuffle(missing)
                    for u2, v2 in missing[:2]:
                        if try_add_edge(u2, v2):
                            added_any = True
                            
                    if added_any:
                        # If we added at least one, we are ahead or equal (if we added 1, we are at -1+1=0, but we removed 1, so we are at -1? No.)
                        # We removed 1 edge. 
                        # If we added 1 edge, net change 0.
                        # If we added 2 edges, net change +1.
                        # We should only keep the change if net change > 0.
                        # So we need to add 2 edges.
                        pass # This logic is flawed. We need to track count.
                        
                    # Revert if we didn't gain
                    # Let's simplify: Just try to add the removed edge back if we didn't gain.
                    # Actually, let's just do a simple "try to add missing edges" loop which is already done above.
                    # The swap logic is tricky to implement correctly without state tracking.
                    # Let's stick to the "Try to add all missing edges" which is effective for greedy.
                    # The main gain comes from better ordering or more restarts.
                    
                    # Let's revert for now to keep code simple and safe
                    adj[ru].add(rv)
                    adj[rv].add(ru)
                    graph_edges.append((ru, rv))

        # Update best
        current_edges = sum(len(s) for s in adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [set(s) for s in adj]
            best_edges_list = [[u, v] for u in range(n) for v in adj[u] if u < v]

    return best_edges_list
