import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    N = 20
    TOTAL_EDGES = N * (N - 1) // 2  # 190
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Bitmask utilities for fast operations
    # Python 3.10+ has int.bit_count(), which is much faster than bin().count('1')
    if hasattr(int, 'bit_count'):
        def count_bits(x):
            return x.bit_count()
    else:
        def count_bits(x):
            return bin(x).count('1')

    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added without violating K2,3 constraint."""
        if adj[u] & (1 << v):
            return True  # Already present
        
        # 1. Check pair (u, v): common neighbours must be <= 2
        if count_bits(adj[u] & adj[v]) >= 3:
            return False
        
        # 2. For each w in N(u) \ {v}, check |N(v) ∩ N(w)| <= 1
        # We only need to check neighbors of u that are NOT v
        # Note: If w is in N(u), then |N(u) ∩ N(w)| is not the constraint for the new edge directly,
        # but adding (u,v) creates new pairs (v,w) and (u,w) is existing? No.
        # The constraint is: For ANY pair of vertices (x,y), |N(x) ∩ N(y)| <= 2.
        # Adding (u,v) affects:
        # - Pair (u,v): |N(u) ∩ N(v)| must be <= 2. (Checked above)
        # - Pair (v,w) for w in N(u): The common neighbors of v and w are N(v) ∩ N(w).
        #   Before adding (u,v), u was not in N(v). Now u IS in N(v).
        #   So N(v) gains u.
        #   Does u become a common neighbor of v and w?
        #   u is in N(v) (new). Is u in N(w)? Yes, because w in N(u) implies u in N(w).
        #   So u is a common neighbor of v and w.
        #   So |N(v) ∩ N(w)| increases by 1 if u was not already in N(v) ∩ N(w) (which it wasn't, as u not in N(v)).
        #   So we need |N(v) ∩ N(w)|_old + 1 <= 2 => |N(v) ∩ N(w)|_old <= 1.
        #   So for all w in N(u) \ {v}, we need count_bits(adj[v] & adj[w]) <= 1.
        
        mask_u = adj[u] & ~(1 << v)
        w = mask_u
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if count_bits(adj[v] & adj[idx]) >= 2:
                return False
        
        # 3. For each w in N(v) \ {u}, check |N(u) ∩ N(w)| <= 1
        # Similarly, v becomes a new common neighbor of u and w.
        # We need |N(u) ∩ N(w)|_old <= 1.
        
        mask_v = adj[v] & ~(1 << u)
        w = mask_v
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if count_bits(adj[u] & adj[idx]) >= 2:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        """Add edge (u,v) to bitmask representation."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        """Remove edge (u,v) from bitmask representation."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def count_edges(adj):
        """Count edges from bitmask representation."""
        return sum(count_bits(adj[u]) for u in range(N)) // 2
    
    def bitmask_to_edge_list(adj):
        """Convert bitmask representation to edge list."""
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    def greedy_build(seed):
        """Greedy construction using bitmask representation."""
        rng = random.Random(seed)
        adj = [0] * N
        
        edge_indices = list(range(TOTAL_EDGES))
        rng.shuffle(edge_indices)
        
        for idx in edge_indices:
            u, v = all_edges[idx]
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        
        return adj
    
    def local_search(adj, seed, time_limit):
        """Local search: try to add edges, and if stuck, try swapping."""
        rng = random.Random(seed)
        adj = [b for b in adj]  # Copy
        
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = [b for b in adj]
        
        start_time_ls = time.time()
        
        while time.time() - start_time_ls < time_limit:
            if time.time() > deadline:
                break
            improved = False
            
            # Try to add random missing edges
            # To speed up, we don't build the full missing list every time if we can avoid it,
            # but for N=20, 190 edges is small.
            missing = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
            if missing:
                rng.shuffle(missing)
                for u, v in missing[:20]:  # Limit checks per iteration
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = [b for b in adj]
                        break
            
            if not improved:
                # Try a swap: remove a random edge, then try to add edges
                present = [(u, v) for u, v in all_edges if (adj[u] & (1 << v))]
                if present:
                    rng.shuffle(present)
                    for u_rm, v_rm in present[:5]:  # Limit swap attempts
                        if time.time() > deadline:
                            break
                        
                        # Save state
                        adj_backup = [b for b in adj]
                        
                        remove_edge(adj, u_rm, v_rm)
                        current_edges -= 1
                        
                        added = 0
                        missing_now = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
                        rng.shuffle(missing_now)
                        for u_add, v_add in missing_now[:20]:
                            if can_add_edge(adj, u_add, v_add):
                                add_edge(adj, u_add, v_add)
                                current_edges += 1
                                added += 1
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = [b for b in adj]
                                if added >= 2:
                                    break
                        
                        if added < 2:
                            # Restore if we didn't gain net edges
                            adj = adj_backup
                            current_edges += 1
                        else:
                            improved = True
                            break
            
            if not improved:
                # Try 2-edge swap: remove 2 edges, add 3 edges
                present = [(u, v) for u, v in all_edges if (adj[u] & (1 << v))]
                if len(present) >= 2:
                    rng.shuffle(present)
                    for i in range(min(3, len(present))):  # Limit 2-edge swap attempts
                        if time.time() > deadline:
                            break
                        u1, v1 = present[i]
                        u2, v2 = present[(i + 1) % len(present)]
                        
                        # Avoid removing the same edge twice if list is small
                        if (u1, v1) == (u2, v2): continue
                        
                        adj_backup = [b for b in adj]
                        
                        remove_edge(adj, u1, v1)
                        remove_edge(adj, u2, v2)
                        current_edges -= 2
                        
                        added = 0
                        missing_now = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
                        rng.shuffle(missing_now)
                        for u_add, v_add in missing_now[:30]:
                            if can_add_edge(adj, u_add, v_add):
                                add_edge(adj, u_add, v_add)
                                current_edges += 1
                                added += 1
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = [b for b in adj]
                                if added >= 3:
                                    break
                        
                        if added < 3:
                            # Restore
                            adj = adj_backup
                            current_edges += 2
                        else:
                            improved = True
                            break
        
        return best_adj, best_edges
    
    # Multiple restarts with different strategies
    best_adj = [0] * N
    best_count = 0
    
    # We will use a mix of random seeds.
    # Given the tight time limit, we prioritize fast greedy + short local search.
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Simple random greedy is fast and effective
        adj = greedy_build(restart)
        
        cur_count = count_edges(adj)
        
        # Apply local search with a small time budget per restart
        # Total time ~ 115s. 50 restarts. ~2s each.
        adj, cur_count = local_search(adj, restart, time_limit=1.5)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = [b for b in adj]
    
    # Final polish: try to add any remaining edges
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not (best_adj[u] & (1 << v)):
            if can_add_edge(best_adj, u, v):
                add_edge(best_adj, u, v)
                best_count += 1
    
    return bitmask_to_edge_list(best_adj)
