import random
import time

def construct():
    n = 40
    adj = [set() for _ in range(n)]
    
    def creates_c6(u, v):
        # Check if adding edge (u, v) creates a C6.
        # Path: u - a - b - c - d - v
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

    def count_edges():
        return sum(len(s) for s in adj) // 2

    start_time = time.time()
    limit = 115 # seconds
    
    best_edges = 0
    best_adj = None
    best_edges_list = []

    # We will run multiple restarts of ILS
    for restart in range(50):
        if time.time() - start_time > limit:
            break
            
        # Reset
        adj = [set() for _ in range(n)]
        
        # 1. Greedy Initialization
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            try_add_edge(u, v)
            
        # 2. Local Search: Add all possible missing edges
        changed = True
        while changed and time.time() - start_time < limit:
            changed = False
            # To avoid bias, shuffle missing edges each iteration? 
            # Or just iterate. Shuffling helps escape order biases.
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            random.shuffle(missing)
            for u, v in missing:
                if try_add_edge(u, v):
                    changed = True
                    # If we added one, the list of missing is stale, 
                    # but we just loop again. 
                    # Breaking and restarting the while loop is cleaner.
                    break
            # If we added one, restart the while loop to re-evaluate
            # Actually, the `while changed` with `break` inside works:
            # If we add, changed=True, break inner for.
            # Next iteration of while, changed is True, so we loop again.
            # If no add, changed remains False, inner for finishes, while ends.
            
        # 3. Iterated Local Search with Swaps
        # We try to improve by removing 1 edge and adding 2 (net +1)
        # or removing 1 and adding 1 (net 0, but changes structure)
        
        current_edges = count_edges()
        
        # List of existing edges for removal
        existing_edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        
        # We will do a limited number of swap attempts per restart
        swap_attempts = 0
        max_swaps_per_restart = 2000
        
        while swap_attempts < max_swaps_per_restart and time.time() - start_time < limit - 2:
            swap_attempts += 1
            
            # Identify missing edges
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            if not missing:
                break # Complete graph (unlikely for C6-free)
                
            random.shuffle(missing)
            
            # Strategy: Pick a random existing edge to remove
            if not existing_edges:
                break
            idx = random.randint(0, len(existing_edges) - 1)
            ru, rv = existing_edges[idx]
            
            # Remove it
            adj[ru].remove(rv)
            adj[rv].remove(ru)
            existing_edges.pop(idx)
            
            # Try to add edges from missing list
            added_count = 0
            added_edges = []
            
            # Try to add up to 2 edges
            # We prioritize adding edges that don't conflict with each other?
            # Just try randomly.
            for u, v in missing[:10]: # Try first 10 random missing edges
                if try_add_edge(u, v):
                    added_count += 1
                    added_edges.append((u, v))
                    existing_edges.append((u, v))
                    if added_count == 2:
                        break
                        # If we added 2, net gain is +1. We keep this state.
                        # If we added 1, net gain is 0. We might keep to explore.
                        # If we added 0, we revert.
                    
            if added_count == 2:
                # Net gain +1. Keep state.
                current_edges += 1
                # Update best
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [set(s) for s in adj]
                    best_edges_list = [[u, v] for u in range(n) for v in adj[u] if u < v]
                
            elif added_count == 1:
                # Net gain 0. 
                # Accept with some probability to move to new local optimum?
                # Or always accept to escape?
                # Let's accept to explore.
                pass
                
            else:
                # Net gain -1. Revert.
                adj[ru].add(rv)
                adj[rv].add(ru)
                existing_edges.append((ru, rv))
                # Revert added edges (none added)
                
        # End of swap phase for this restart
        
        # Update best one last time
        current_edges = count_edges()
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [set(s) for s in adj]
            best_edges_list = [[u, v] for u in range(n) for v in adj[u] if u < v]

    return best_edges_list
