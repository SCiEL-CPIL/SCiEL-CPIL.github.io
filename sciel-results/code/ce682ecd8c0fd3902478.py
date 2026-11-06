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
                
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Vectorized check: for each b in b_mask, check if adj[b] & c_mask
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def dense_random_init(seed_val, density=0.65):
        """
        Create a dense random initialization and prune C6s.
        """
        random.seed(seed_val)
        edges = set()
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    edges.add((i, j))
        
        adj = build_adj(edges)
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            for u, v in list(edges):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                if would_create_c6(adj, u, v):
                    to_remove.append((u, v))
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            for u, v in to_remove:
                edges.remove((u, v))
                changed = True
        
        return edges

    def run_greedy(start_edges, seed_val):
        random.seed(seed_val)
        
        adj = build_adj(start_edges)
        edges = set(start_edges)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 2:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search(edges_set, seed_val, T_start=1.0, T_end=0.1, max_iter=300):
        """
        Simulated Annealing Local Search.
        State: edge set.
        Move: Add an edge, remove an edge, or swap (remove 1, add 1).
        Objective: Maximize edge count.
        """
        random.seed(seed_val)
        
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        current_score = len(current_edges)
        
        # Precompute non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        # We will perform a fixed number of iterations
        # Temperature schedule
        T = T_start
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.4 and non_edges:
                # Add an edge
                idx = random.randint(0, len(non_edges) - 1)
                u, v = non_edges[idx]
                
                # Check if valid
                if not would_create_c6(adj, u, v):
                    # Apply move
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    new_score = len(current_edges)
                    
                    # Always accept improvement
                    non_edges.pop(idx)
                    current_score = new_score
            
            elif move_type < 0.7 and current_edges:
                # Remove an edge
                idx = random.randint(0, len(current_edges) - 1)
                e_rem = list(current_edges)[idx]
                u, v = e_rem
                
                # Apply move
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges.remove(e_rem)
                new_score = len(current_edges)
                
                # Accept with probability exp(-1/T)
                if random.random() < np.exp(-1.0 / T):
                    current_score = new_score
                    non_edges.append(e_rem)
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add(e_rem)
            
            else:
                # Swap: Remove 1, Add 1
                if current_edges and non_edges:
                    idx_rem = random.randint(0, len(current_edges) - 1)
                    e_rem = list(current_edges)[idx_rem]
                    u_rem, v_rem = e_rem
                    
                    idx_add = random.randint(0, len(non_edges) - 1)
                    e_add = non_edges[idx_add]
                    u_add, v_add = e_add
                    
                    # Remove e_rem
                    adj[u_rem] &= ~(1 << v_rem)
                    adj[v_rem] &= ~(1 << u_rem)
                    current_edges.remove(e_rem)
                    
                    # Check if e_add is valid
                    if not would_create_c6(adj, u_add, v_add):
                        # Add e_add
                        adj[u_add] |= (1 << v_add)
                        adj[v_add] |= (1 << u_add)
                        current_edges.add(e_add)
                        
                        # Accept neutral move
                        non_edges.pop(idx_add)
                        non_edges.append(e_rem)
                    else:
                        # Revert removal
                        adj[u_rem] |= (1 << v_rem)
                        adj[v_rem] |= (1 << u_rem)
                        current_edges.add(e_rem)
            
            # Cool down
            T = max(T_end, T * 0.98)
            
            # Periodic greedy fill?
            if it % 50 == 0:
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
                if not added_any:
                    break # No more immediate improvements
            
        return current_edges

    def perturb_and_refine(edges_set, seed_val, num_remove):
        """
        Remove some edges randomly, then run greedy + local search to refine.
        """
        random.seed(seed_val)
        edges = set(edges_set)
        
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        adj = build_adj(edges)
        
        # Run greedy
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
                edges.add((u, v))
        
        # Run SA Local Search
        edges = local_search(edges, seed_val)
        
        return edges

    # Main search loop
    
    best_edges = set()
    best_count = 0
    
    # Try multiple dense random initializations
    if time.time() < deadline - 40:
        for seed_val in [42, 123, 777, 999, 555]:
            if time.time() > deadline - 35:
                break
            dense_edges = dense_random_init(seed_val)
            if len(dense_edges) > best_count:
                best_edges = dense_edges
                best_count = len(best_edges)
    
    # Initial greedy
    best_edges = run_greedy(best_edges, 42)
    
    # SA Local Search
    best_edges = local_search(best_edges, 42)
    
    best_count = len(best_edges)
    
    # Multiple restarts with perturbation
    restart_seed = 100
    for i in range(50):
        if time.time() > deadline - 10:
            break
        
        num_remove = random.randint(2, 8)
        perturbed = perturb_and_refine(best_edges, restart_seed + i, num_remove)
        
        if len(perturbed) > best_count:
            best_edges = perturbed
            best_count = len(perturbed)
        
        restart_seed += 1
    
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
