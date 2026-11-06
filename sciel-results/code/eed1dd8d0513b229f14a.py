"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    n = 40

    def would_create_c6(adj, u, v):
        """
        Optimized C6 check using bitmask union.
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
                if not b_mask:
                    continue
                    
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not c_mask:
                    continue
                
                # Check if any b in b_mask has a neighbor in c_mask
                union_b = 0
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    union_b |= adj[b]
                    b_temp ^= lsb_b
                
                if union_b & c_mask:
                    return True
        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_random_non_edge(adj):
        for _ in range(20):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if not (adj[i] & (1 << j)):
                return (i, j)
        return None

    def dense_random_init(seed_val):
        random.seed(seed_val)
        edges = set()
        
        # Try a few densities
        best_edges = set()
        best_count = 0
        
        for density in [0.60, 0.65, 0.70]:
            if time.time() > deadline - 30:
                break
                
            random.seed(seed_val + int(density * 100))
            candidate_edges = set()
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < density:
                        candidate_edges.add((i, j))
            
            adj = build_adj(candidate_edges)
            
            # Prune C6s
            changed = True
            while changed:
                changed = False
                to_remove = []
                for u, v in list(candidate_edges):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    if would_create_c6(adj, u, v):
                        to_remove.append((u, v))
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                
                for u, v in to_remove:
                    candidate_edges.remove((u, v))
                    changed = True
            
            if len(candidate_edges) > best_count:
                best_edges = candidate_edges
                best_count = len(candidate_edges)
        
        return best_edges

    def run_greedy(edges_set, adj, seed_val):
        random.seed(seed_val)
        edges = set(edges_set)
        adj = adj[:]
        
        # Get all non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 5:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges, adj

    def local_search(edges_set, adj, seed_val, time_limit):
        random.seed(seed_val)
        edges = set(edges_set)
        adj = adj[:]
        
        # Pass 1: Greedy Add
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > time_limit:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        # Pass 2: Swaps (1-for-1 and 1-for-2)
        removable = list(edges)
        random.shuffle(removable)
        
        consecutive_failures = 0
        max_failures = 50
        
        while time.time() < time_limit:
            if consecutive_failures > max_failures:
                break
            if not removable:
                break
            
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            old_adj = adj[:]
            old_edges = set(edges)
            
            # Remove
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            edges.remove(e_rem)
            
            # Try to add 2 non-edges
            added = 0
            # Generate fresh non-edges list for this iteration? Too slow.
            # Just try random non-edges
            attempts = 0
            while added < 2 and attempts < 50 and time.time() < time_limit:
                e_add = get_random_non_edge(adj)
                if e_add is None:
                    break
                u_a, v_a = e_add
                if e_add in edges:
                    attempts += 1
                    continue
                if not would_create_c6(adj, u_a, v_a):
                    adj[u_a] |= (1 << v_a)
                    adj[v_a] |= (1 << u_a)
                    edges.add(e_add)
                    added += 1
                attempts += 1
            
            if added >= 2:
                consecutive_failures = 0
                removable = list(edges)
                random.shuffle(removable)
            elif added == 1:
                if random.random() < 0.3:
                    consecutive_failures = 0
                    removable = list(edges)
                    random.shuffle(removable)
                else:
                    adj = old_adj
                    edges = old_edges
                    consecutive_failures += 1
            else:
                adj = old_adj
                edges = old_edges
                consecutive_failures += 1

        return edges, adj

    def simulated_annealing(edges_set, adj, seed_val, time_limit):
        random.seed(seed_val)
        edges = set(edges_set)
        adj = adj[:]
        count = len(edges)
        
        best_edges = set(edges)
        best_count = count
        
        T = 1.0
        cooling = 0.9995
        
        while time.time() < time_limit:
            move_type = random.random()
            
            if move_type < 0.4: # Add
                e = get_random_non_edge(adj)
                if e:
                    u, v = e
                    if not would_create_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add(e)
                        count += 1
                        if count > best_count:
                            best_count = count
                            best_edges = set(edges)
            
            elif move_type < 0.7: # Remove
                if edges:
                    e = random.choice(list(edges))
                    u, v = e
                    # Accept with probability exp(-1/T)
                    if random.random() < math.exp(-1.0 / T):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edges.remove(e)
                        count -= 1
            
            else: # Swap
                if edges:
                    e_rem = random.choice(list(edges))
                    e_add = get_random_non_edge(adj)
                    if e_add and e_add not in edges:
                        u_r, v_r = e_rem
                        u_a, v_a = e_add
                        
                        old_adj = adj[:]
                        old_edges = set(edges)
                        
                        adj[u_r] &= ~(1 << v_r)
                        adj[v_r] &= ~(1 << u_r)
                        edges.remove(e_rem)
                        
                        if not would_create_c6(adj, u_a, v_a):
                            adj[u_a] |= (1 << v_a)
                            adj[v_a] |= (1 << u_a)
                            edges.add(e_add)
                            # Count unchanged
                        else:
                            adj = old_adj
                            edges = old_edges
                        
            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_edges, best_count

    # Main Loop
    best_edges = set()
    best_count = 0
    
    # Initial dense random
    for seed in [42, 123, 777]:
        if time.time() > deadline - 40:
            break
        init_edges = dense_random_init(seed)
        if len(init_edges) > best_count:
            best_edges = init_edges
            best_count = len(best_edges)
    
    if not best_edges:
        # Fallback
        best_edges = dense_random_init(42)
        best_count = len(best_edges)
    
    adj = build_adj(best_edges)
    
    # Polish with Greedy
    best_edges, adj = run_greedy(best_edges, adj, 42)
    best_count = len(best_edges)
    
    # Main Search Loop: SA + Local Search
    restart_seed = 100
    while time.time() < deadline - 10:
        # Run SA for a short duration
        sa_time_limit = min(time.time() + 10, deadline - 10)
        sa_edges, sa_adj = simulated_annealing(best_edges, adj, restart_seed, sa_time_limit)
        
        if len(sa_edges) > best_count:
            best_edges = sa_edges
            best_count = len(best_edges)
            adj = sa_adj
        
        # Polish with Local Search
        ls_time_limit = min(time.time() + 10, deadline - 10)
        ls_edges, ls_adj = local_search(best_edges, adj, restart_seed + 1, ls_time_limit)
        
        if len(ls_edges) > best_count:
            best_edges = ls_edges
            best_count = len(best_edges)
            adj = ls_adj
            
        restart_seed += 1
        
        # Occasional full restart with new random init?
        if restart_seed % 5 == 0 and time.time() < deadline - 30:
            new_init = dense_random_init(restart_seed)
            if len(new_init) > best_count - 5: # If it's close, try it
                 # Maybe just use it as a seed for SA
                 pass
                 
    # Final Polish
    final_adj = build_adj(best_edges)
    best_edges, final_adj = run_greedy(best_edges, final_adj, 999)
    
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
