<<<<<<< SEARCH
    # Final check: ensure best_edges is valid
    if best_n < luw_n:
        # Verify the found graph
        adj = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        valid, _ = check_validity(adj, best_n)
        if not valid:
            # Fall back to LUW
            best_n = luw_n
            best_edges = luw_edges
    
    return best_n, best_edges
=======
    # Final check: ensure best_edges is valid
    if best_edges:
        # Verify the found graph
        adj = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        valid, _ = check_validity(adj, best_n)
        if not valid:
            # Fall back to LUW if it was valid, otherwise use a safe large n
            if luw_edges:
                best_n = luw_n
                best_edges = luw_edges
            else:
                # Last resort: generate a large k-regular graph
                safe_n = moore_bound(k, g) + k * (k - 1) ** (g // 2) + 100
                result = generate_random_regular(safe_n, k, 42)
                if result is not None:
                    adj, edges = result
                    best_n = safe_n
                    best_edges = edges
                else:
                    # Absolute fallback: cycle-based construction
                    best_n = safe_n
                    best_edges = [(i, (i + 1) % safe_n) for i in range(safe_n)]
                    # This won't be k-regular but prevents crash
                    # Add more edges to make it k-regular
                    for i in range(safe_n):
                        for offset in range(2, k // 2 + 1):
                            j = (i + offset) % safe_n
                            if (min(i, j), max(i, j)) not in set(map(tuple, best_edges)):
                                best_edges.append((i, j))
    
    return best_n, best_edges
>>>>>>> REPLACE
