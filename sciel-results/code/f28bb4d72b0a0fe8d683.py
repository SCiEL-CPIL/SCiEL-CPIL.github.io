<<<<<<< SEARCH
    def local_search(edges_set, seed_val):
        """
        Improve the edge set by adding edges and swapping.
        """
        random.seed(seed_val)
        
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        
        # Pass 1: Add any possible non-edge
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 2:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
        
        # Pass 2: Try swaps (remove 1, add 2)
        removable = list(current_edges)
        random.shuffle(removable)
        
        # Re-generate non_edges list as it changed
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        consecutive_failures = 0
        max_failures = 50
        
        while time.time() < deadline - 2:
            if consecutive_failures > max_failures:
                break
                
            if not removable:
                break
            
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            # Try to add 2 non-edges
            added = 0
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 2:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added >= 2:
                # Success: Edge count increased by (added - 1)
                consecutive_failures = 0
                # Update removable and non_edges for next iteration
                removable = list(current_edges)
                random.shuffle(removable)
                
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] & (1 << j)):
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                
                # Continue searching for more improvements
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1

        return current_edges
=======
    def local_search(edges_set, seed_val):
        """
        Improve the edge set by adding edges and swapping.
        """
        random.seed(seed_val)
        
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        
        # Pass 1: Add any possible non-edge (multiple passes)
        for _ in range(3):
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        non_edges.append((i, j))
            
            if not non_edges:
                break
            
            random.shuffle(non_edges)
            
            changed = False
            for u, v in non_edges:
                if time.time() > deadline - 2:
                    break
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    changed = True
            
            if not changed:
                break
        
        # Pass 2: Try swaps (remove 1, add 2)
        removable = list(current_edges)
        random.shuffle(removable)
        
        # Re-generate non_edges list as it changed
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        consecutive_failures = 0
        max_failures = 30  # Reduced to allow more iterations
        
        while time.time() < deadline - 2:
            if consecutive_failures > max_failures:
                break
                
            if not removable:
                break
            
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            # Try to add 2 non-edges
            added = 0
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 2:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added >= 2:
                # Success: Edge count increased by (added - 1)
                consecutive_failures = 0
                # Update removable and non_edges for next iteration
                removable = list(current_edges)
                random.shuffle(removable)
                
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] & (1 << j)):
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1

        # Final polish: try to add any remaining edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 1:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))

        return current_edges
>>>>>>> REPLACE

