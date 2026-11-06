"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    n = 40
    all_mask = (1 << n) - 1

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        Optimized by pre-computing the union of neighbors for the inner loop.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                Na = adj[a]
                Nd = adj[d]
                
                # b must be in N(a) \ {u, v, d}
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                if not b_mask:
                    continue
                    
                # c must be in N(d) \ {u, v, a}
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not c_mask:
                    continue
                
                # Check if there exists b in b_mask such that adj[b] intersects c_mask
                # Compute union of neighbors of all b in b_mask
                union_b_neighbors = 0
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    union_b_neighbors |= adj[b]
                
                if union_b_neighbors & c_mask:
                    return True
        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def k5_seed():
        """
        Create a seed of 8 disjoint K5 blocks.
        Vertices 0-4, 5-9, ..., 35-39 form K5s.
        """
        edges = set()
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.add((base + a, base + b))
        return edges

    def add_random_inter_block_edges(edges_set, seed_val, prob=0.3):
        """
        Add random edges between different K5 blocks.
        """
        random.seed(seed_val)
        blocks = [list(range(5*k, 5*k+5)) for k in range(8)]
        
        # Add edges between all pairs of blocks
        for i in range(8):
            for j in range(i+1, 8):
                for u in blocks[i]:
                    for v in blocks[j]:
                        if random.random() < prob:
                            edges_set.add((u, v))
        return edges_set

    def prune_c6s(edges_set):
        """
        Remove edges that create C6s until the graph is C6-free.
        """
        adj = build_adj(edges_set)
        changed = True
        while changed:
            changed = False
            to_remove = []
            for u, v in list(edges_set):
                # Temporarily remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                if would_create_c6(adj, u, v):
                    to_remove.append((u, v))
                else:
                    # Keep it
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            for u, v in to_remove:
                edges_set.remove((u, v))
                changed = True
        return edges_set

    def greedy_fill(edges_set, seed_val):
        """
        Add any non-edge that doesn't create a C6.
        """
        random.seed(seed_val)
        adj = build_adj(edges_set)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 5:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_set.add((u, v))
        
        return edges_set

    def local_search_sa(edges_set, seed_val, T_start=1.0, T_end=0.1, max_iter=200):
        """
        Simulated Annealing Local Search.
        """
        random.seed(seed_val)
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        current_score = len(current_edges)
        
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        T = T_start
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            move_type = random.random()
            
            if move_type < 0.3 and non_edges:
                # Add an edge
                idx = random.randint(0, len(non_edges) - 1)
                u, v = non_edges[idx]
                
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    non_edges.pop(idx)
                    current_score += 1
            
            elif move_type < 0.6 and current_edges:
                # Remove an edge
                edge_list = list(current_edges)
                idx = random.randint(0, len(edge_list) - 1)
                e_rem = edge_list[idx]
                u, v = e_rem
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges.remove(e_rem)
                
                if random.random() < np.exp(-1.0 / T):
                    current_score -= 1
                    non_edges.append(e_rem)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add(e_rem)
            
            else:
                # Swap: Remove 1, Add 1
                if current_edges and non_edges:
                    edge_list = list(current_edges)
                    idx_rem = random.randint(0, len(edge_list) - 1)
                    e_rem = edge_list[idx_rem]
                    u_rem, v_rem = e_rem
                    
                    idx_add = random.randint(0, len(non_edges) - 1)
                    e_add = non_edges[idx_add]
                    u_add, v_add = e_add
                    
                    adj[u_rem] &= ~(1 << v_rem)
                    adj[v_rem] &= ~(1 << u_rem)
                    current_edges.remove(e_rem)
                    
                    if not would_create_c6(adj, u_add, v_add):
                        adj[u_add] |= (1 << v_add)
                        adj[v_add] |= (1 << u_add)
                        current_edges.add(e_add)
                        
                        non_edges.pop(idx_add)
                        non_edges.append(e_rem)
                    else:
                        # Revert
                        adj[u_rem] |= (1 << v_rem)
                        adj[v_rem] |= (1 << u_rem)
                        current_edges.add(e_rem)
            
            T = max(T_end, T * 0.98)
            
            # Periodic greedy fill
            if it % 20 == 0:
                added_any = False
                for i in range(len(non_edges)):
                    u, v = non_edges[i]
                    if not would_create_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges.add((u, v))
                        non_edges.pop(i)
                        added_any = True
                        break
                if not added_any and it > 50:
                    break
            
        return current_edges

    # Main search loop
    best_edges = set()
    best_count = 0
    
    # Try multiple seeds for inter-block edges
    for seed_val in [42, 123, 777, 999, 555, 111, 222, 333, 444, 555]:
        if time.time() > deadline - 20:
            break
        
        # 1. Create K5 seed
        edges = k5_seed()
        
        # 2. Add random inter-block edges
        edges = add_random_inter_block_edges(edges, seed_val, prob=0.4)
        
        # 3. Prune C6s
        edges = prune_c6s(edges)
        
        # 4. Greedy fill
        edges = greedy_fill(edges, seed_val)
        
        # 5. Local Search
        edges = local_search_sa(edges, seed_val)
        
        if len(edges) > best_count:
            best_edges = edges
            best_count = len(edges)
    
    # Final polish: try to add any remaining non-edges
    adj_final = build_adj(best_edges)
    final_non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_final[i] & (1 << j)):
                final_non_edges.append((i, j))
    
    random.shuffle(final_non_edges)
    for u, v in final_non_edges:
        if time.time() > deadline - 2:
            break
        if not would_create_c6(adj_final, u, v):
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
            best_edges.add((u, v))
    
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
