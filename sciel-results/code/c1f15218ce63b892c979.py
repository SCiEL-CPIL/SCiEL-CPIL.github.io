import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
        # adj is a list of integers (bitmasks)
        # 1. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 2. Check constraints on neighbors of u
        # For w in N(u), u becomes a common neighbor of v and w.
        # So we need current common(v, w) <= 1.
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            # Calculate common neighbors of v and w
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 3. Check constraints on neighbors of v
        # For w in N(v), v becomes a common neighbor of u and w.
        # So we need current common(u, w) <= 1.
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_fill(order):
        adj = [0] * n
        count = 0
        for u, v in order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    def local_search(adj, count):
        # Simple local search: try to add edges that were skipped or swap
        # For now, just try to add any missing edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find all missing edges
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        missing.append((i, j))
            
            if not missing:
                break
                
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
                    # Continue to pack more
                    if time.time() > deadline:
                        return count
            
        return count

    def two_opt_swap(adj, count):
        """
        Try to remove one edge and add two new edges to increase density.
        This is a heuristic local search move.
        """
        if time.time() > deadline:
            return count
        
        # Find all present edges
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        
        if not present:
            return count

        random.shuffle(present)
        
        # Try a few random removals
        for _ in range(20):
            if time.time() > deadline:
                break
                
            # Pick a random edge to remove
            rem_edge = random.choice(present)
            u, v = rem_edge
            remove_edge(adj, u, v)
            temp_count = count - 1
            
            # Now try to add as many edges as possible
            improved = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                # Find missing edges
                missing = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] >> j) & 1:
                            missing.append((i, j))
                
                if not missing:
                    break
                
                random.shuffle(missing)
                added_any = False
                for a, b in missing:
                    if is_valid_add(adj, a, b):
                        add_edge(adj, a, b)
                        temp_count += 1
                        added_any = True
                        break
                
                if not added_any:
                    break
            
            if temp_count > count:
                count = temp_count
                improved = True
            else:
                # Revert
                add_edge(adj, u, v)
                count = temp_count + 1 # Restore original count logic? No, temp_count was count-1 + additions.
                # Actually, if we didn't improve, we should revert to original state.
                # We removed (u,v) so we add it back.
                # The count is currently temp_count.
                # Original count was count.
                # If we revert, count should be original count.
                # But we modified adj.
                # Let's just return if no improvement to keep it simple and safe.
                # Actually, the logic above is tricky. Let's simplify:
                # If temp_count <= count, revert.
                add_edge(adj, u, v)
                count = temp_count + 1 # This is wrong if temp_count != count-1.
                # Let's restart the function or just break if no improvement found in this batch.
                # To be safe, let's just do simple addition local search and maybe a different swap strategy.
                # The current two_opt_swap is risky. Let's stick to the proven local search and maybe add a "shrink-expand" if time permits.
                # For now, let's just use the local search.
                break
        
        return count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 5:
        random.shuffle(all_edges)
        adj, count = greedy_fill(all_edges)
        
        # Local search
        count = local_search(adj, count)
        
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_edges.append([i, j])
                    mask ^= lsb
                    
        restarts += 1
        
        # If we are running out of time, break early
        if time.time() > deadline - 10:
            break

    return best_edges
