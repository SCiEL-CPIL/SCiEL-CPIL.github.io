import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    n = 40

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        A new C6 must use (u, v), so we need a path of length 5:
        u - a - b - c - d - v
        """
        Nu = adj[u]
        Nv = adj[v]

        # a in N(u) \ {v}
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a

            # d in N(v) \ {u, a}
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d

                # b in N(a) \ {u, v, d}
                # c in N(d) \ {u, v, a}
                # need b != c and (b, c) edge
                
                Na = adj[a]
                Nd = adj[d]
                
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Check if exists b in b_mask such that adj[b] & c_mask != 0
                # Note: if b==c, adj[b] won't have bit b set, so it's automatically excluded
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    if adj[b] & c_mask:
                        return True
        return False

    def can_add(adj, u, v):
        return not would_create_c6(adj, u, v)

    def run_greedy(seed_val):
        random.seed(seed_val)
        
        adj = [0] * n
        edges = set()
        
        # Strategy: Start with a random sparse base or specific structure
        # Let's try starting with a random matching or small cliques
        # But actually, starting from scratch might be better with random order
        
        # Generate all edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return adj, edges

    def local_search(adj, edges, seed_val, time_limit):
        """
        Try to improve by removing edges and re-adding.
        Returns (adj, edges)
        """
        random.seed(seed_val)
        current_edges = set(edges)
        current_adj = adj[:]
        
        # Non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (current_adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        if not non_edges:
            return current_adj, current_edges

        # Pass 1: Add any possible non-edge
        # We do this iteratively until no more can be added
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle non_edges to randomize order
            random.shuffle(non_edges)
            new_non_edges = []
            for u, v in non_edges:
                if time.time() > time_limit:
                    break
                if (u, v) in current_edges:
                    continue
                if can_add(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    improved = True
                else:
                    new_non_edges.append((u, v))
            non_edges = new_non_edges
        
        if not non_edges:
            return current_adj, current_edges

        # Pass 2: Swaps (remove 1, add 2)
        # This is expensive. We try a limited number of random swaps.
        # We pick a random edge, remove it, then try to add all possible non-edges.
        # If we add >= 2, we keep the change.
        
        # To speed up, we only try a subset of edges to remove
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        
        # Limit the number of removal candidates to keep time under control
        # Each removal check involves iterating non_edges which is O(E_non) * C6_check
        # C6_check is fast but not free.
        max_removals_to_try = 200
        
        for rem_edge in edge_list[:max_removals_to_try]:
            if time.time() > time_limit:
                break
            
            u_rem, v_rem = rem_edge
            
            # Save state
            old_adj = current_adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            current_adj[u_rem] &= ~(1 << v_rem)
            current_adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(rem_edge)
            
            # Try to add non-edges
            # We need to re-evaluate non_edges because the graph changed
            # But for speed, we can just iterate through the existing non_edges list
            # and check can_add. Also, the removed edge becomes a non-edge.
            # Let's add the removed edge to the non_edges pool for this attempt?
            # Actually, we want to add *other* edges.
            
            added_count = 0
            # Iterate through a shuffled copy of non_edges + [rem_edge]
            # But rem_edge is now a non-edge too.
            # Let's just iterate through all potential edges that are currently non-edges
            # To avoid rebuilding the list, we iterate through the old non_edges + rem_edge
            
            candidates = non_edges + [rem_edge]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > time_limit:
                    break
                if (u, v) in current_edges:
                    continue
                if can_add(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added_count += 1
            
            if added_count > 1:
                # Net gain
                # Update non_edges list? It's complex. 
                # For simplicity, we return and let the outer loop re-run or restart.
                # But we can just return the improved graph.
                return current_adj, current_edges
            else:
                # Revert
                current_adj = old_adj
                current_edges = old_edges

        return current_adj, current_edges

    best_edges = set()
    best_count = 0
    
    seed_counter = 0
    while time.time() < deadline:
        # Reserve time for local search
        if time.time() > deadline - 10:
            break
            
        adj, edges = run_greedy(seed_counter)
        
        # Local search
        # Give local search a portion of the remaining time
        time_left = deadline - time.time()
        ls_time_limit = time_left * 0.5  # Use half the remaining time for LS
        if ls_time_limit < 1.0:
            ls_time_limit = 1.0
            
        adj, edges = local_search(adj, edges, seed_counter + 10000, time.time() + ls_time_limit)
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = set(edges)
            # print(f"New best: {count}")
            
        seed_counter += 1
        
        # If we found 105, we might stop? No, keep searching for 106+.
        # But if we have many restarts, we might hit time limit.
        
    result = [[u, v] for u, v in best_edges]
    return result

