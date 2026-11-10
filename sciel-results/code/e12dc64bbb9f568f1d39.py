import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    # cc[u][v] is the number of common neighbors of u and v
    cc = [[0] * n for _ in range(n)]

    def popcount(x):
        return bin(x).count('1')

    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if cc[u][v] > 2:
            return False
        # Check neighbors of v
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                return False
            mask_v ^= lsb
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                return False
            mask_u ^= lsb
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        # Update cc
        # For each neighbor w of v (excluding u), cc[u][w] increases by 1
        # We need to iterate over neighbors of v BEFORE adding u?
        # Actually, add_edge sets adj first.
        # Let's be careful.
        # When adding (u,v):
        # For any w in N(v) \ {u}, u and w now share v as a common neighbor. So cc[u][w] += 1.
        # For any w in N(u) \ {v}, v and w now share u as a common neighbor. So cc[v][w] += 1.
        
        # Current adj[v] includes u.
        # So N(v) \ {u} is adj[v] with bit u cleared.
        old_mask_v = adj[v] ^ (1 << u)
        m = old_mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] += 1
            cc[w][u] += 1
            m ^= lsb

        old_mask_u = adj[u] ^ (1 << v)
        m = old_mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] += 1
            cc[w][v] += 1
            m ^= lsb

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

        # Update cc
        # For any w in N(v) (after removal, so excluding u), cc[u][w] decreases by 1.
        # Note: adj[v] no longer has u. So adj[v] is N(v) \ {u}.
        m = adj[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] -= 1
            cc[w][u] -= 1
            m ^= lsb

        # For any w in N(u) (after removal, so excluding v), cc[v][w] decreases by 1.
        m = adj[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] -= 1
            cc[w][v] -= 1
            m ^= lsb

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    random.seed(42)

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    all_edges_set = set(all_edges)

    best_edges = []
    best_count = 0

    # Strategy:
    # 1. Greedy construction with multiple random orders to get a good starting point.
    # 2. Local search: 1-flip (add/remove) and 2-flip (swap) moves.
    #    Since we want to maximize edges, "add" moves are strictly beneficial if valid.
    #    "Swap" moves (remove one, add one) keep edge count constant but can enable future adds.
    #    "Remove" moves are only useful if they allow adding two or more edges later,
    #    which is hard to predict. We can simulate this by allowing temporary removals
    #    with a penalty, or just relying on swaps.
    #    Actually, a common technique is to try to add an edge. If blocked, try removing
    #    one of the blocking edges and see if it opens up more.

    # Let's implement a robust local search.
    
    def local_search(initial_adj, initial_cc, initial_count):
        nonlocal best_edges, best_count
        
        # Copy state
        adj_ls = initial_adj[:]
        cc_ls = [row[:] for row in initial_cc]
        count_ls = initial_count

        # Helper functions that operate on adj_ls, cc_ls
        def ls_can_add(u, v):
            if (adj_ls[u] >> v) & 1:
                return False
            if cc_ls[u][v] > 2:
                return False
            mask_v = adj_ls[v]
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                if cc_ls[u][w] > 1:
                    return False
                mask_v ^= lsb
            mask_u = adj_ls[u]
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                if cc_ls[v][w] > 1:
                    return False
                mask_u ^= lsb
            return True

        def ls_add_edge(u, v):
            adj_ls[u] |= (1 << v)
            adj_ls[v] |= (1 << u)
            
            old_mask_v = adj_ls[v] ^ (1 << u)
            m = old_mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cc_ls[u][w] += 1
                cc_ls[w][u] += 1
                m ^= lsb
                
            old_mask_u = adj_ls[u] ^ (1 << v)
            m = old_mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cc_ls[v][w] += 1
                cc_ls[w][v] += 1
                m ^= lsb

        def ls_remove_edge(u, v):
            adj_ls[u] &= ~(1 << v)
            adj_ls[v] &= ~(1 << u)
            
            m = adj_ls[v]
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cc_ls[u][w] -= 1
                cc_ls[w][u] -= 1
                m ^= lsb
                
            m = adj_ls[u]
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cc_ls[v][w] -= 1
                cc_ls[w][v] -= 1
                m ^= lsb

        def ls_count_edges():
            c = 0
            for u in range(n):
                c += popcount(adj_ls[u])
            return c // 2

        def ls_get_edges():
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_ls[u] >> v) & 1:
                        edges.append([u, v])
            return edges

        T = 1.0
        cooling = 0.999
        no_improve = 0
        
        while time.time() < deadline - 2.0 and T > 0.01:
            # Try to find a valid add move
            # Pick random non-existing edge
            non_existing = [(u, v) for u, v in all_edges if not ((adj_ls[u] >> v) & 1)]
            
            if non_existing:
                u, v = random.choice(non_existing)
                if ls_can_add(u, v):
                    ls_add_edge(u, v)
                    count_ls += 1
                    no_improve = 0
                    if count_ls > best_count:
                        best_count = count_ls
                        best_edges = ls_get_edges()
                    T *= cooling
                    continue
            
            # If no add possible, try a swap
            # Pick random existing edge to remove
            existing = [(u, v) for u, v in all_edges if (adj_ls[u] >> v) & 1]
            if not existing:
                break
                
            u1, v1 = random.choice(existing)
            ls_remove_edge(u1, v1)
            count_ls -= 1
            
            # Try to add a random non-existing edge
            # We removed (u1, v1), so it's now non-existing.
            # We want to add an edge that might have been blocked by (u1, v1).
            # Simply pick a random non-existing edge.
            non_existing_after = [(u, v) for u, v in all_edges if not ((adj_ls[u] >> v) & 1)]
            
            added = False
            # To increase efficiency, we can prioritize edges that share vertices with (u1, v1)
            # or are likely to be unblocked.
            # For now, random selection from a subset.
            candidates = random.sample(non_existing_after, min(10, len(non_existing_after)))
            for u2, v2 in candidates:
                if ls_can_add(u2, v2):
                    ls_add_edge(u2, v2)
                    count_ls += 1
                    added = True
                    break
                    
            if not added:
                # Revert
                ls_add_edge(u1, v1)
                count_ls += 1
                T *= cooling
                no_improve += 1
            else:
                # Swap successful, count unchanged
                # Check if this enables further adds?
                # The loop will try adds next iteration.
                no_improve += 1
                T *= cooling
            
            if no_improve > 200:
                if T < 0.1:
                    break

        return count_ls

    # Main loop: Multiple restarts
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Reset state
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        
        # Greedy construction
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add(u, v):
                add_edge(u, v)
                
        current_count = count_edges()
        
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edges()
            
        # Local search
        # Pass copies
        local_search(adj, cc, current_count)
        
        # If local search improved, update best
        # (local_search updates best_edges/best_count directly if it finds something better)
        
        # Optional: Try to improve the best solution found so far with local search
        # This is expensive, so maybe only do it if we have time.
        
    # Final polish on the best solution found
    # Reconstruct best_adj, best_cc from best_edges
    if best_edges:
        best_adj = [0] * n
        best_cc = [[0] * n for _ in range(n)]
        for u, v in best_edges:
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            # Recalculate cc is expensive, but we can just run local search which uses its own state.
            # Wait, local_search takes initial_adj, initial_cc.
            # We need to compute cc for best_adj.
            # Let's just compute it.
            for w in range(n):
                if (best_adj[u] >> w) & 1 and w != v:
                    best_cc[u][w] += 1
                    best_cc[w][u] += 1
                if (best_adj[v] >> w) & 1 and w != u:
                    best_cc[v][w] += 1
                    best_cc[w][v] += 1
        # Actually, the above cc calculation is wrong because it double counts or misses interactions.
        # Better to just run local_search on a fresh state built from best_edges.
        # But local_search modifies its own copy.
        # Let's just run local_search on the current best state.
        
        # Rebuild cc correctly
        best_cc = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v: continue
                # Common neighbors of u and v
                common = best_adj[u] & best_adj[v]
                best_cc[u][v] = popcount(common)
                
        local_search(best_adj, best_cc, best_count)

    return best_edges

# Test
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
    # Verify K2,3-free
    n = 20
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if bin(common).count('1') > 2:
                valid = False
                print(f"Invalid: vertices {u} and {v} have {bin(common).count('1')} common neighbors")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
