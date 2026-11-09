import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n = 16
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
            
    # Map edges to indices for bitmask representation?
    # Or just use the set of edges.
    
    best_edges = set()
    best_count = 0

    # Precompute all 8-sets
    all_8sets = list(itertools.combinations(range(n), 8))
    
    # Precompute splits for each 8-set
    # splits_cache[combo] = list of (A_indices, B_indices)
    splits_cache = {}
    for combo in all_8sets:
        vlist = list(combo)
        splits = []
        for i in range(8):
            for j in range(i + 1, 8):
                for k in range(j + 1, 8):
                    for l in range(k + 1, 8):
                        A_idx = (i, j, k, l)
                        B_idx = tuple(x for x in range(8) if x not in A_idx)
                        splits.append((A_idx, B_idx))
        splits_cache[combo] = splits

    def check_q3_fast(adj):
        """Fast check: return True if Q3 exists, else False."""
        # Use local references for speed
        adj_local = adj
        for combo in all_8sets:
            # Quick edge count
            ec = 0
            # Unroll loop for speed? No, keep it simple but use local vars
            for i in range(8):
                u = combo[i]
                au = adj_local[u]
                for j in range(i + 1, 8):
                    v = combo[j]
                    if (au >> v) & 1:
                        ec += 1
            if ec < 12:
                continue
            
            for A_idx, B_idx in splits_cache[combo]:
                A = [combo[i] for i in A_idx]
                B = [combo[i] for i in B_idx]
                
                rows_used = 0
                cols_used = 0
                missing_count = 0
                valid = True
                for ri in range(4):
                    a = A[ri]
                    aa = adj_local[a]
                    for ci in range(4):
                        b = B[ci]
                        if not ((aa >> b) & 1):
                            if (rows_used >> ri) & 1 or (cols_used >> ci) & 1:
                                valid = False
                                break
                            rows_used |= (1 << ri)
                            cols_used |= (1 << ci)
                            missing_count += 1
                            if missing_count > 4:
                                valid = False
                                break
                    if not valid:
                        break
                if valid and missing_count <= 4:
                    return True
        return False

    def find_q3_edge_to_remove(adj):
        """Find a Q3 and return an edge in it to remove. Returns None if no Q3."""
        adj_local = adj
        for combo in all_8sets:
            ec = 0
            for i in range(8):
                u = combo[i]
                au = adj_local[u]
                for j in range(i + 1, 8):
                    v = combo[j]
                    if (au >> v) & 1:
                        ec += 1
            if ec < 12:
                continue
            
            for A_idx, B_idx in splits_cache[combo]:
                A = [combo[i] for i in A_idx]
                B = [combo[i] for i in B_idx]
                
                rows_used = 0
                cols_used = 0
                missing_count = 0
                valid = True
                for ri in range(4):
                    a = A[ri]
                    aa = adj_local[a]
                    for ci in range(4):
                        b = B[ci]
                        if not ((aa >> b) & 1):
                            if (rows_used >> ri) & 1 or (cols_used >> ci) & 1:
                                valid = False
                                break
                            rows_used |= (1 << ri)
                            cols_used |= (1 << ci)
                            missing_count += 1
                            if missing_count > 4:
                                valid = False
                                break
                    if not valid:
                        break
                if valid and missing_count <= 4:
                    # Collect existing edges in A×B
                    existing = []
                    for a in A:
                        aa = adj_local[a]
                        for b in B:
                            if (aa >> b) & 1:
                                existing.append((min(a,b), max(a,b)))
                    if existing:
                        return random.choice(existing)
        return None

    def try_add_edge(adj, edges, u, v):
        """Try to add edge (u,v). Returns True if added successfully."""
        e = (min(u,v), max(u,v))
        if e in edges:
            return True
        edges.add(e)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not check_q3_fast(adj):
            return True
        else:
            edges.remove(e)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    # Multiple restarts with different strategies
    random.seed(42)
    
    # Strategy 1: Random dense init (0.75 - 0.95)
    # Strategy 2: Full init (1.0)
    
    for trial in range(10): # Increased trials
        if time.time() > deadline - 30:
            break
        
        if trial < 5:
            density = 0.75 + 0.05 * trial
        else:
            density = 1.0 # Start with K16 for some trials
        
        edges = set()
        adj = [0] * n
        for e in all_edges:
            if random.random() < density:
                edges.add(e)
                u, v = e
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Phase 1: Remove edges to make Q3-free
        removals = 0
        while time.time() < deadline - 25 and removals < 100: # Increased removal limit
            e_to_remove = find_q3_edge_to_remove(adj)
            if e_to_remove is None:
                break
            edges.remove(e_to_remove)
            u, v = e_to_remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            removals += 1
        
        # Verify Q3-free
        if check_q3_fast(adj):
            # Still has Q3, try more removals
            while time.time() < deadline - 20:
                e_to_remove = find_q3_edge_to_remove(adj)
                if e_to_remove is None:
                    break
                edges.remove(e_to_remove)
                u, v = e_to_remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Phase 2: Try to add edges
        added_any = True
        while added_any and time.time() < deadline - 15:
            added_any = False
            non_edges = [e for e in all_edges if e not in edges]
            random.shuffle(non_edges)
            for (u, v) in non_edges:
                if try_add_edge(adj, edges, u, v):
                    added_any = True
                    break
        
        # Update best
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)

    # Final polish on best
    if best_edges and time.time() < deadline - 10:
        adj = [0] * n
        for (u, v) in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Try adding more edges
        added_any = True
        while added_any and time.time() < deadline - 5:
            added_any = False
            non_edges = [e for e in all_edges if e not in best_edges]
            random.shuffle(non_edges)
            for (u, v) in non_edges:
                e = (min(u,v), max(u,v))
                if e not in best_edges:
                    best_edges.add(e)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not check_q3_fast(adj):
                        added_any = True
                        break
                    else:
                        best_edges.remove(e)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

    return [list(e) for e in best_edges]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
