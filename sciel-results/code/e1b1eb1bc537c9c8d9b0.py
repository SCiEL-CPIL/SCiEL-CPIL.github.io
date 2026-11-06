    def construct():
        n = 40
        m = 401
        deadline = time.time() + 115.0
        
        def random_graph():
            # Generate random graph with m edges
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_edges)
            return set(all_edges[:m])

        def compute_books(A):
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            return A2

        best_edges = None
        best_max_b = float('inf')
        
        while time.time() < deadline:
            # Start with random graph
            edges = random_graph()
            A = np.zeros((n, n), dtype=np.int32)
            for u, v in edges:
                A[u, v] = 1
                A[v, u] = 1
            
            A2 = compute_books(A)
            max_b = np.max(A2)
            
            if max_b < best_max_b:
                best_max_b = max_b
                best_edges = set(edges)
            
            # Local Search
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Find worst edge
                idx = np.argmax(A2)
                u, v = np.unravel_index(idx, A2.shape)
                if A2[u, v] == 0: break # No edges left? or max is 0
                
                worst_edge = (min(u, v), max(u, v))
                current_book = A2[u, v]
                
                best_swap = None
                best_swap_book = current_book
                best_swap_sum = np.sum(A2**2)
                
                # Try a few random swaps for this worst edge
                for _ in range(50):
                    # Pick random non-edge
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y: continue
                    new_edge = (min(x, y), max(x, y))
                    if new_edge in edges: continue
                    if new_edge == worst_edge: continue
                    
                    # Simulate swap
                    # Remove (u,v), Add (x,y)
                    A[u, v] = 0; A[v, u] = 0
                    A[x, y] = 1; A[y, x] = 1
                    
                    A2_new = compute_books(A)
                    max_new = np.max(A2_new)
                    sum_new = np.sum(A2_new**2)
                    
                    # Revert
                    A[u, v] = 1; A[v, u] = 1
                    A[x, y] = 0; A[y, x] = 0
                    
                    if max_new < best_swap_book or (max_new == best_swap_book and sum_new < best_swap_sum):
                        best_swap_book = max_new
                        best_swap_sum = sum_new
                        best_swap = new_edge
                        best_swap_A2 = A2_new.copy() # Store A2 to avoid recompute?
                        # Actually, storing A2 is expensive. Just store the edge and recompute if accepted.
                        # But we need to know if it's better.
                        # Let's just store the edge and the metrics.
                        # If we accept, we update A and A2.
                    
                if best_swap:
                    # Apply best swap
                    u, v = worst_edge
                    x, y = best_swap
                    edges.discard(worst_edge)
                    edges.add(best_swap)
                    
                    A[u, v] = 0; A[v, u] = 0
                    A[x, y] = 1; A[y, x] = 1
                    
                    A2 = compute_books(A)
                    max_b = np.max(A2)
                    sum_b = np.sum(A2**2)
                    
                    if max_b < best_max_b or (max_b == best_max_b and sum_b < np.sum(np.array(best_edges) @ np.array(best_edges) ** 2)): # This check is expensive, skip sum check for global best?
                         # Actually, we just track max_b for global best.
                         if max_b < best_max_b:
                             best_max_b = max_b
                             best_edges = set(edges)
                    improved = True
                else:
                    # Try random SA move if stuck on worst edge?
                    # Or just break local search and restart.
                    break
                    
        return [[u, v] for u, v in best_edges]
    