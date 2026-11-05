    # Phase 2: Local Search on Best Graph
    if best_edges and time.time() < deadline - 5:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        current_count = best_count
        current_adj = adj[:]
        
        # Multiple passes of greedy addition
        while time.time() < deadline - 2:
            improved = False
            missing = [(u, v) for u, v in all_edges if not ((current_adj[u] >> v) & 1)]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 2:
                    break
                # Try to add
                if (current_adj[u] & current_adj[v]) == 0:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    if not is_3_colorable(current_adj):
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_edges = edges_from_adj(current_adj)
                        improved = True
                    else:
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
            
            if not improved:
                # If no improvement, break or try a kick?
                # Let's try a kick: remove a random edge, then continue
                if current_count > best_count - 5: # Only kick if we are close to best
                    existing = [(u, v) for u, v in all_edges if (current_adj[u] >> v) & 1]
                    if existing:
                        u, v = random.choice(existing)
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                        current_count -= 1
                        # Continue loop to try to add others
                    else:
                        break
                else:
                    break

    return best_edges
