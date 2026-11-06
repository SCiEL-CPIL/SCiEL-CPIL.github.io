import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    N = 20
    TOTAL_EDGES = N * (N - 1) // 2  # 190
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    edge_index.update({(v, u): i for i, (u, v) in enumerate(all_edges)})
    
    # Bitmask utilities for fast operations
    def count_bits(x):
        return bin(x).count('1')
    
    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added without violating K2,3 constraint."""
        if adj[u] & (1 << v):
            return True  # Already present
        
        # Check pair (u, v): common neighbours must be <= 2
        if count_bits(adj[u] & adj[v]) >= 3:
            return False
        
        # For each w in N(v), check |N(u) ∩ N(w)| <= 1
        for w in range(N):
            if w == u or w == v:
                continue
            if adj[v] & (1 << w):
                if count_bits(adj[u] & adj[w]) >= 2:
                    return False
        
        # For each w in N(u), check |N(v) ∩ N(w)| <= 1
        for w in range(N):
            if w == u or w == v:
                continue
            if adj[u] & (1 << w):
                if count_bits(adj[v] & adj[w]) >= 2:
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
    
    def is_valid(adj):
        """Check no two vertices have 3+ common neighbours."""
        for u in range(N):
            for v in range(u + 1, N):
                if count_bits(adj[u] & adj[v]) >= 3:
                    return False
        return True
    
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
            missing = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
            if missing:
                rng.shuffle(missing)
                for u, v in missing[:50]:
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
                    for u_rm, v_rm in present[:10]:
                        if time.time() > deadline:
                            break
                        
                        # Save state
                        adj_backup = [b for b in adj]
                        
                        remove_edge(adj, u_rm, v_rm)
                        current_edges -= 1
                        
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
                    for i in range(min(5, len(present))):
                        if time.time() > deadline:
                            break
                        u1, v1 = present[i]
                        u2, v2 = present[(i + 1) % len(present)]
                        
                        adj_backup = [b for b in adj]
                        
                        remove_edge(adj, u1, v1)
                        remove_edge(adj, u2, v2)
                        current_edges -= 2
                        
                        added = 0
                        missing_now = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
                        rng.shuffle(missing_now)
                        for u_add, v_add in missing_now[:40]:
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
    
    def simulated_annealing(adj, seed, time_limit):
        """Simulated annealing to maximize edges in K2,3-free graph."""
        rng = random.Random(seed)
        adj = [b for b in adj]  # Copy
        
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = [b for b in adj]
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        cooling_rate = 0.9995
        
        T = T_start
        start_time_sa = time.time()
        
        while time.time() - start_time_sa < time_limit and T > T_end:
            if time.time() > deadline:
                break
            
            # Pick a random edge to toggle
            idx = rng.randint(0, TOTAL_EDGES - 1)
            u, v = all_edges[idx]
            
            if adj[u] & (1 << v):
                # Remove edge
                remove_edge(adj, u, v)
                new_edges = current_edges - 1
                
                # Always accept removal
                accept = True
            else:
                # Try to add edge
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    new_edges = current_edges + 1
                    accept = True
                else:
                    # Can't add due to constraint - reject
                    T *= cooling_rate
                    continue
            
            if accept:
                current_edges = new_edges
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [b for b in adj]
            
            T *= cooling_rate
        
        return best_adj, best_edges
    
    # Multiple restarts with different strategies
    best_adj = [0] * N
    best_count = 0
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initial strategies
        if restart % 3 == 0:
            # Pure random greedy
            adj = greedy_build(restart)
        elif restart % 3 == 1:
            # Greedy with bias toward high-degree vertices
            rng = random.Random(restart)
            adj = [0] * N
            # Start with a random matching
            vertices = list(range(N))
            rng.shuffle(vertices)
            for i in range(0, N - 1, 2):
                u, v = vertices[i], vertices[i + 1]
                add_edge(adj, u, v)
            # Then greedily add more
            edge_indices = list(range(TOTAL_EDGES))
            rng.shuffle(edge_indices)
            for idx in edge_indices:
                u, v = all_edges[idx]
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
        else:
            # Start from a known good structure and perturb
            adj = [0] * N
            # Create 4 disjoint K5s
            for k in range(4):
                base = k * 5
                for i in range(5):
                    for j in range(i + 1, 5):
                        add_edge(adj, base + i, base + j)
            # Add some cross edges
            rng = random.Random(restart)
            for _ in range(30):
                u = rng.randint(0, N - 1)
                v = rng.randint(0, N - 1)
                if u != v and not (adj[u] & (1 << v)):
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
        
        cur_count = count_edges(adj)
        
        # Apply local search
        adj, cur_count = local_search(adj, restart, time_limit=2.0)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = [b for b in adj]
        
        # If we have time, try simulated annealing on the best solution
        if time.time() < deadline - 30:
            adj_sa, sa_count = simulated_annealing(best_adj, restart, time_limit=5.0)
            if sa_count > best_count:
                best_count = sa_count
                best_adj = [b for b in adj_sa]
    
    # Final polish: try to add any remaining edges
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not (best_adj[u] & (1 << v)):
            if can_add_edge(best_adj, u, v):
                add_edge(best_adj, u, v)
                best_count += 1
    
    return bitmask_to_edge_list(best_adj)
