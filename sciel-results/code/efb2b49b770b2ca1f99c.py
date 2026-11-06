import time
import random

def construct():
    n = 20
    start_time = time.time()
    # Use a conservative deadline to ensure we return before the 120s kill
    deadline = start_time + 100.0 

    def is_add_valid(adj, u, v):
        """
        Fast check: only examine triples where the new edge (u,v) could create a K3,3.
        When adding (u,v), a new K3,3 can only form if there exist a,b such that:
        - v is a neighbor of both a and b, and u,a,b already have >=2 common neighbors
          (so adding v as a 3rd common neighbor creates K3,3)
        - u is a neighbor of both a and b, and v,a,b already have >=2 common neighbors
        
        Key insight: we only need to check pairs (a,b) that are BOTH neighbors of the
        "other" endpoint. If a or b is not a neighbor of v, then v can't be a common
        neighbor of {u,a,b}, so no new K3,3 involving v as common neighbor can form.
        """
        # Triples {u, a, b}: need v to be common neighbor of u,a,b
        # v is already neighbor of u (we just added it). Need v neighbor of a and b.
        # So a,b must be neighbors of v (excluding u and v).
        # Also need: adj[u] & adj[a] & adj[b] already has >=2 bits (so adding v makes 3)
        v_nb = adj[v] & ~(1 << u) & ~(1 << v)
        vn_list = [a for a in range(n) if v_nb & (1 << a)]
        
        for i in range(len(vn_list)):
            a = vn_list[i]
            for j in range(i + 1, len(vn_list)):
                b = vn_list[j]
                # Count common neighbors of u,a,b BEFORE adding v
                # adj[u] already includes v, so we need to exclude v from count
                common_ub = (adj[u] & adj[a] & adj[b]) & ~(1 << v)
                if common_ub.bit_count() >= 2:
                    return False
        
        # Triples {v, a, b}: need u to be common neighbor of v,a,b
        u_nb = adj[u] & ~(1 << v) & ~(1 << u)
        un_list = [a for a in range(n) if u_nb & (1 << a)]
        
        for i in range(len(un_list)):
            a = un_list[i]
            for j in range(i + 1, len(un_list)):
                b = un_list[j]
                common_va = (adj[v] & adj[a] & adj[b]) & ~(1 << u)
                if common_va.bit_count() >= 2:
                    return False
        
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """
        Greedily adds edges in random order.
        Returns the number of edges in the final adj.
        """
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_add_valid(adj, u, v):
                pass
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return count_edges(adj)

    best_adj = [0] * n
    best_count = 0
    current_adj = [0] * n
    
    # Initial greedy fill from empty graph
    current_count = greedy_fill(current_adj)
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]

    # Local search / Restart strategy with 2-edge swap moves
    iterations = 0
    no_improve_count = 0
    while time.time() < deadline:
        iterations += 1
        current_adj = best_adj[:]
        
        # Get current edges
        edges_to_remove = []
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u] & (1 << v):
                    edges_to_remove.append((u, v))
        
        if not edges_to_remove:
            continue
        
        # Choose move type: kick (remove+regreedy) or 2-edge swap
        move_type = random.random()
        
        if move_type < 0.5:
            # Kick: remove random edges, then re-greedy
            if no_improve_count < 10:
                num_removals = random.randint(1, 3)
            elif no_improve_count < 30:
                num_removals = random.randint(2, 6)
            else:
                num_removals = random.randint(3, 10)
            
            shuffled = edges_to_remove[:]
            random.shuffle(shuffled)
            for u, v in shuffled[:num_removals]:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            
            current_count = greedy_fill(current_adj)
            
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # 2-edge swap: remove one edge, try to add another
            shuffled = edges_to_remove[:]
            random.shuffle(shuffled)
            if len(shuffled) == 0:
                no_improve_count += 1
                continue
                
            # Remove one random edge
            ru, rv = shuffled[0]
            current_adj[ru] &= ~(1 << rv)
            current_adj[rv] &= ~(1 << ru)
            
            # Try to add a random missing edge
            all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n) 
                         if not (current_adj[u] & (1 << v))]
            if all_pairs:
                au, av = random.choice(all_pairs)
                current_adj[au] |= (1 << av)
                current_adj[av] |= (1 << au)
                if not is_add_valid(current_adj, au, av):
                    current_adj[au] &= ~(1 << av)
                    current_adj[av] &= ~(1 << au)
                    # Re-add the removed edge
                    current_adj[ru] |= (1 << rv)
                    current_adj[rv] |= (1 << ru)
            
            current_count = count_edges(current_adj)
            
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Rollback swap
                current_adj[ru] |= (1 << rv)
                current_adj[rv] |= (1 << ru)
                if (current_adj[au] & (1 << av)):
                    current_adj[au] &= ~(1 << av)
                    current_adj[av] &= ~(1 << au)
            
        # If stuck for too long, do a full restart
        if no_improve_count >= 50:
            current_adj = [0] * n
            current_count = greedy_fill(current_adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
            no_improve_count = 0

    # Final polish: try adding any missing edge that doesn't create K3,3
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if best_adj[u] & (1 << v):
                    continue
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                if is_add_valid(best_adj, u, v):
                    improved = True
                else:
                    best_adj[u] &= ~(1 << v)
                    best_adj[v] &= ~(1 << u)
            if time.time() > deadline:
                break

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
