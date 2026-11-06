<<<<<<< SEARCH
    def local_search(adj, max_time=1.0):
        """Local search with kick mechanism."""
        ls_start = time.time()
        current_edges = sum(x.bit_count() for x in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        # First, greedy fill
        adj, current_edges = greedy_fill(adj)
        best_adj = adj[:]
        best_edges = current_edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if not missing:
            return best_adj, best_edges
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - ls_start < max_time and no_improve_count < max_no_improve:
            if time.time() > deadline - 3:
                break
                
            if not missing:
                break
                
            # Try to add a random missing edge
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                missing.pop(idx)
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Kick: remove 1-3 random edges, then greedy fill
                kick_size = random.choice([1, 1, 1, 2, 2, 3])
                existing = get_edge_list(adj)
                if not existing:
                    break
                    
                removed = []
                for _ in range(min(kick_size, len(existing))):
                    if not existing:
                        break
                    idx2 = random.randint(0, len(existing) - 1)
                    a, b = existing[idx2]
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    current_edges -= 1
                    removed.append((a, b))
                    existing.pop(idx2)
                
                # Greedy fill from current state
                adj, current_edges = greedy_fill(adj)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # Revert to best if we're stuck
                    if no_improve_count >= max_no_improve:
                        adj = best_adj[:]
                        current_edges = best_edges
                        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                        no_improve_count = 0
                        if not missing:
                            break
                        continue
                        # Actually, let's just break and restart
                        break
                
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                
        return best_adj, best_edges
=======
    def local_search(adj, max_time=1.0):
        """Local search with kick mechanism."""
        ls_start = time.time()
        current_edges = sum(x.bit_count() for x in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        # First, greedy fill
        adj, current_edges = greedy_fill(adj, max_attempts=5, time_limit=deadline - 5)
        best_adj = adj[:]
        best_edges = current_edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if not missing:
            return best_adj, best_edges
        
        no_improve_count = 0
        max_no_improve = 30
        
        while time.time() - ls_start < max_time and no_improve_count < max_no_improve:
            if time.time() > deadline - 3:
                break
                
            if not missing:
                break
                
            # Try to add a random missing edge
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                missing.pop(idx)
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Kick: remove 1-4 random edges, then greedy fill
                kick_size = random.choice([1, 1, 2, 2, 3, 3, 4])
                existing = get_edge_list(adj)
                if not existing:
                    break
                    
                removed = []
                for _ in range(min(kick_size, len(existing))):
                    if not existing:
                        break
                    idx2 = random.randint(0, len(existing) - 1)
                    a, b = existing[idx2]
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    current_edges -= 1
                    removed.append((a, b))
                    existing.pop(idx2)
                
                # Greedy fill from current state
                adj, current_edges = greedy_fill(adj, max_attempts=3, time_limit=deadline - 5)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # Revert to best if we're stuck
                    if no_improve_count >= max_no_improve:
                        adj = best_adj[:]
                        current_edges = best_edges
                        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                        no_improve_count = 0
                        if not missing:
                            break
                        break
                
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                
        return best_adj, best_edges
>>>>>>> REPLACE

