import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_edges_list(adj_state):
        edges = []
        for i in range(n):
            mask = adj_state[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def is_k23_free(adj):
        """Check if graph is K2,3-free: no two vertices share 3+ common neighbors."""
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common.bit_count() >= 3:
                    return False
        return True

    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it keeps graph K2,3-free."""
        if u == v or (adj[u] >> v) & 1:
            return False
        # Check: for each w in N(u), common(w,v) would increase by 1
        # common(w,v) = popcount(adj[w] & adj[v])
        # After adding (u,v): for w in N(u)\{v}, new common(w,v) = popcount(adj[w] & (adj[v]|(1<<u)))
        # = popcount((adj[w] & adj[v]) | (adj[w] & (1<<u)))
        # Since w != u and w in N(u), adj[w] & (1<<u) = 1 if u in N(w) which is true since w in N(u)
        # So new common(w,v) = popcount(adj[w] & adj[v]) + 1 (if u not already in N(v))
        # Wait, we need to be more careful.
        # Adding edge (u,v): adj[u] gains v, adj[v] gains u.
        # For pair (w, v) where w in N(u): common(w,v) = popcount(adj[w] & adj[v])
        # After: adj[v] has u added. So new common(w,v) = popcount(adj[w] & (adj[v] | (1<<u)))
        # = popcount((adj[w] & adj[v]) | (adj[w] & (1<<u)))
        # If u in adj[w] (which is true since w in N(u)), then adj[w] & (1<<u) = 1<<u
        # So new = popcount((adj[w] & adj[v]) | (1<<u))
        # = popcount(adj[w] & adj[v]) + 1 if u not in adj[w] & adj[v], else same
        # But u not in adj[w] & adj[v] means u is not a common neighbor of w and v
        # Since we're checking before adding, u may or may not be in adj[v]
        # Actually simpler: just check the constraint directly
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[w] >> u) & 1:  # w is neighbor of u
                # After adding (u,v), common(w,v) = popcount(adj[w] & (adj[v] | (1<<u)))
                new_common = (adj[w] & (adj[v] | (1 << u))).bit_count()
                if new_common >= 3:
                    return False
        for x in range(n):
            if x == u or x == v:
                continue
            if (adj[x] >> v) & 1:  # x is neighbor of v
                # After adding (u,v), common(u,x) = popcount((adj[u] | (1<<v)) & adj[x])
                new_common = ((adj[u] | (1 << v)) & adj[x]).bit_count()
                if new_common >= 3:
                    return False
        return True

    def do_add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def do_remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    best_adj = [0] * n
    best_count = 0

    # Phase 1: Greedy initialization with multiple random orderings
    for restart in range(200):
        if time.time() > deadline - 30:
            break
        adj = [0] * n
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if try_add_edge(adj, u, v):
                do_add_edge(adj, u, v)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Phase 2: Local search with edge swaps from best solution
    if time.time() < deadline - 5:
        adj = best_adj[:]
        current_count = best_count
        
        # Try to add any missing edges
        improved = True
        max_iterations = 500
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            improved = False
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        do_add_edge(adj, u, v)
                        current_count += 1
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
            if not improved:
                break
        
        # Edge swap: remove one edge, try to add two
        for swap_iter in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Find an edge to remove
            existing = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        existing.append((i, j))
                    mask ^= lsb
            
            if not existing:
                break
            
            u, v = random.choice(existing)
            do_remove_edge(adj, u, v)
            current_count -= 1
            
            # Try to add edges
            added = 0
            random.shuffle(all_pairs)
            for a, b in all_pairs:
                if not ((adj[a] >> b) & 1):
                    if try_add_edge(adj, a, b):
                        do_add_edge(adj, a, b)
                        current_count += 1
                        added += 1
                        if added >= 2:
                            break
            
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            
            # If we didn't gain, put the edge back
            if current_count <= best_count - 1:
                do_add_edge(adj, u, v)
                current_count += 1
            elif added == 0:
                # No progress, revert
                do_add_edge(adj, u, v)
                current_count += 1

    return get_edges_list(best_adj)
