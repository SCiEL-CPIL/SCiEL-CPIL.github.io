import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge indices for faster lookup if needed, but bit masks are fast enough
    # adj[i] is a bitmask of neighbors of i

    def is_valid_add(adj, u, v):
        # Check if edge (u,v) is already present
        if (adj[u] >> v) & 1:
            return False
        
        # Check if u and v have > 2 common neighbors
        common_uv = (adj[u] & adj[v]).bit_count()
        if common_uv > 2:
            return False
            
        # Check if adding (u,v) creates a K2,3 with any existing neighbor w of u
        # For any w in N(u) \ {v}, N(u) \cap N(w) must not contain v (which it doesn't if edge not added)
        # Actually, the condition is: for any w in N(u), the common neighbors of u and w must be <= 2.
        # If we add (u,v), v becomes a neighbor of u.
        # So for any w in N(u) (old), v is now in N(u). 
        # We need |N(u) \cap N(w)| <= 2.
        # Since v is in N(u) (new), if v is also in N(w), then v is in N(u) \cap N(w).
        # The old intersection didn't have v. So new intersection size = old intersection size + (1 if v in N(w) else 0).
        # So if v in N(w), we require old |N(u) \cap N(w)| <= 1.
        # If v not in N(w), we require old |N(u) \cap N(w)| <= 2 (which is already true for valid graph).
        
        # Similarly for w in N(v).
        
        temp = adj[u]
        # We only need to check neighbors w of u where w != v. Since edge not added, v is not in adj[u] yet.
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # w is a neighbor of u.
            # If v is a neighbor of w, then v is a common neighbor of u and w (after adding u-v).
            # The other common neighbors are those in (adj[u] & adj[w]).
            # Note: adj[u] does NOT contain v yet.
            common_ow = (adj[u] & adj[w]).bit_count()
            if (adj[w] >> v) & 1:
                # v is neighbor of w, so v is a new common neighbor
                if common_ow + 1 > 2:
                    return False
            else:
                if common_ow > 2:
                    return False
            temp ^= lsb
            
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # w is a neighbor of v.
            # If u is a neighbor of w, then u is a new common neighbor of v and w.
            common_ow = (adj[v] & adj[w]).bit_count()
            if (adj[w] >> u) & 1:
                if common_ow + 1 > 2:
                    return False
            else:
                if common_ow > 2:
                    return False
            temp ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def polish(adj):
        """Try to add any missing edge that is valid."""
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        improved = True
        return adj

    def local_search(adj, rng, time_budget):
        """
        Robust local search with 1-swap and 2-swap.
        """
        start_ls = time.time()
        present = set()
        missing = set()
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                present.add((u, v))
            else:
                missing.add((u, v))
        
        # First, polish
        polish(adj)
        
        # Rebuild sets after polish
        present = set()
        missing = set()
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                present.add((u, v))
            else:
                missing.add((u, v))
                
        improved = True
        while improved and (time.time() - start_ls < time_budget):
            improved = False
            
            # Try 1-swap
            if present and missing:
                present_list = list(present)
                missing_list = list(missing)
                # Limit tries to avoid spending too much time on one step
                tries = min(100, len(present_list) * len(missing_list))
                for _ in range(tries):
                    if time.time() - start_ls > time_budget:
                        return adj
                    u1, v1 = rng.choice(present_list)
                    u2, v2 = rng.choice(missing_list)
                    
                    remove_edge(adj, u1, v1)
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        present.discard((u1, v1))
                        present.add((u2, v2))
                        missing.add((u1, v1))
                        missing.discard((u2, v2))
                        improved = True
                        break
                    else:
                        add_edge(adj, u1, v1)
                        
            # Try 2-swap
            if improved:
                continue # Restart loop to re-evaluate
                
            if len(present) >= 2 and len(missing) >= 2:
                present_list = list(present)
                missing_list = list(missing)
                tries = min(50, len(present_list) * len(missing_list))
                for _ in range(tries):
                    if time.time() - start_ls > time_budget:
                        return adj
                    # Pick 2 present, 2 missing
                    # Using sample for unique pairs
                    try:
                        p1, p2 = rng.sample(present_list, 2)
                        m1, m2 = rng.sample(missing_list, 2)
                    except ValueError:
                        break
                        
                    u1, v1 = p1
                    u2, v2 = p2
                    u3, v3 = m1
                    u4, v4 = m2
                    
                    remove_edge(adj, u1, v1)
                    remove_edge(adj, u2, v2)
                    
                    ok1 = is_valid_add(adj, u3, v3)
                    if ok1:
                        add_edge(adj, u3, v3)
                        ok2 = is_valid_add(adj, u4, v4)
                        if ok2:
                            add_edge(adj, u4, v4)
                            # Update sets
                            present.discard(p1)
                            present.discard(p2)
                            present.add(m1)
                            present.add(m2)
                            missing.add(p1)
                            missing.add(p2)
                            missing.discard(m1)
                            missing.discard(m2)
                            improved = True
                            break
                        else:
                            remove_edge(adj, u3, v3)
                    # Re-add removed edges if failed
                    if not (ok1 and is_valid_add(adj, u4, v4) if ok1 else True):
                         # This logic is getting messy, simpler to just restore if not improved
                         pass 
                    
                    # Restore if not improved
                    if not improved:
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                        
        return polish(adj)

    def greedy_init(rng):
        adj = [0] * n
        edges = all_edges[:]
        rng.shuffle(edges)
        for u, v in edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        return polish(adj)

    def structured_init(rng):
        adj = [0] * n
        # Try a few circulant structures
        # Circulant with steps 1, 2
        for i in range(n):
            for d in [1, 2]:
                j1 = (i + d) % n
                j2 = (i - d) % n
                for j in [j1, j2]:
                    if j != i and not ((adj[i] >> j) & 1):
                        if is_valid_add(adj, i, j):
                            add_edge(adj, i, j)
        return polish(adj)

    def random_structured_init(rng):
        adj = [0] * n
        # Start with random 4-regular-ish graph
        for i in range(n):
            candidates = [j for j in range(n) if j != i and not ((adj[i] >> j) & 1)]
            rng.shuffle(candidates)
            for j in candidates[:4]:
                if is_valid_add(adj, i, j):
                    add_edge(adj, i, j)
        return polish(adj)

    best_count = 0
    best_adj = [0] * n
    
    rng = random.Random(42)
    
    # 1. Try structured inits
    for _ in range(5):
        adj = structured_init(rng)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            
    # 2. Try random structured inits
    for _ in range(10):
        adj = random_structured_init(rng)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            
    # 3. Run local search on best candidates and random inits
    # We will run local search on the current best and some random ones
    
    # Allocate time budget for local search
    # 115s total. Spent ~1s on inits. Remaining ~110s.
    # Let's run local search on the best found so far, and then on new random inits if time permits.
    
    # Run local search on the best_adj
    time_per_ls = 2.0
    adj_work = best_adj[:]
    adj_work = local_search(adj_work, rng, time_per_ls)
    cnt = count_edges(adj_work)
    if cnt > best_count:
        best_count = cnt
        best_adj = adj_work[:]
        
    # Run local search on several random inits
    while time.time() < deadline - 5:
        adj_init = greedy_init(rng)
        cnt_init = count_edges(adj_init)
        # Only run LS if it's promising or we have time
        # To maximize diversity, we run LS on every init if time allows
        # But LS takes ~2s. 110/2 = 55 restarts.
        
        adj_ls = local_search(adj_init, rng, time_per_ls)
        cnt_ls = count_edges(adj_ls)
        if cnt_ls > best_count:
            best_count = cnt_ls
            best_adj = adj_ls[:]
            
    # Final polish on best
    best_adj = polish(best_adj)
    best_count = count_edges(best_adj)
    
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    print("Valid" if valid else "Invalid")
