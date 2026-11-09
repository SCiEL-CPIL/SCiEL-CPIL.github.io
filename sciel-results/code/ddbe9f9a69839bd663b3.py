import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    
    # Construct Clebsch Graph (Folded 5-cube)
    def get_clebsch_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                diff = u ^ v
                dist = bin(diff).count('1')
                if dist == 1 or dist == 4:
                    edges.append([u, v])
        return edges

    base_edges = get_clebsch_edges()
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def is_triangle_free(adj, n):
        for i in range(n):
            for j in adj[i]:
                if i > j:
                    continue
                if adj[i] & adj[j]:
                    return False
        return True

    def is_not_3_colorable(adj, n):
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        # Note: The original code used a simple backtrack(0) which is correct 
        # but slower. We can optimize by using the order, but for n=16 
        # simple backtrack is often fast enough if we use bitmasks or similar.
        # Let's stick to the simple one for clarity, or implement a faster one.
        # For n=16, 3^16 is huge, but backtracking prunes heavily.
        # Let's use a slightly optimized backtracking with degree ordering.
        
        # Actually, let's just use the provided one, it works.
        return not backtrack(0)

    # Validation
    if not is_triangle_free(adj, n) or not is_not_3_colorable(adj, n):
        # Fallback
        pass

    current_edges = len(base_edges)
    best_edges = current_edges
    best_adj = [s.copy() for s in adj]

    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Local Search
    # Strategy: Try to add edges. If stuck, try removing one edge and adding multiple.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Try simple additions
        addable = []
        for u, v in all_possible_edges:
            if v in adj[u]:
                continue
            if not (adj[u] & adj[v]):
                addable.append((u, v))
                
        random.shuffle(addable)
        for u, v in addable:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            
            if is_not_3_colorable(adj, n):
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [s.copy() for s in adj]
                improved = True
            else:
                adj[u].discard(v)
                adj[v].discard(u)

        # 2. If no simple addition worked, try edge swaps (remove 1, add 1+)
        if not improved:
            # Try removing a random edge and adding as many as possible
            if current_edges > 0:
                # Pick a random edge to remove
                rem_u, rem_v = random.choice(base_edges) # Use base_edges as a proxy for current? 
                # Better: pick from current adj
                current_edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            current_edge_list.append((u, v))
                
                if current_edge_list:
                    rem_u, rem_v = random.choice(current_edge_list)
                    
                    # Remove it
                    adj[rem_u].discard(rem_v)
                    adj[rem_v].discard(rem_u)
                    current_edges -= 1
                    
                    # Try to add all possible edges
                    addable_after_removal = []
                    for u, v in all_possible_edges:
                        if v in adj[u]:
                            continue
                        if not (adj[u] & adj[v]):
                            addable_after_removal.append((u, v))
                    
                    random.shuffle(addable_after_removal)
                    added_count = 0
                    for u, v in addable_after_removal:
                        if time.time() > deadline:
                            break
                        if v in adj[u]:
                            continue
                        
                        adj[u].add(v)
                        adj[v].add(u)
                        
                        if is_not_3_colorable(adj, n):
                            added_count += 1
                            current_edges += 1
                        else:
                            adj[u].discard(v)
                            adj[v].discard(u)
                    
                    # If we added more than we removed, we improved
                    if added_count > 1:
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = [s.copy() for s in adj]
                        improved = True
                    else:
                        # Revert to state before removal? 
                        # We lost rem_u, rem_v. We added some.
                        # If current_edges < best_edges, we might want to revert to best_adj?
                        # For now, we keep the current state if it's valid, 
                        # but we should probably restart from best_adj if we get stuck.
                        # Let's just revert the specific removal if it didn't help much?
                        # Simpler: if we didn't improve, revert the removal and try another.
                        # But we already added some edges. 
                        # Let's just accept the state if it's valid.
                        # If it's worse than best, we will eventually restart or timeout.
                        # To be safe, let's restart from best_adj if we are stuck.
                        if current_edges < best_edges:
                            adj = [s.copy() for s in best_adj]
                            current_edges = best_edges
                            improved = True # Force loop to continue
                        else:
                            # Keep current state
                            pass

    # Final validation and return
    if not is_triangle_free(best_adj, n) or not is_not_3_colorable(best_adj, n):
         # Fallback to base
         return base_edges
         
    final_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                final_edges.append([u, v])
                
    return final_edges
