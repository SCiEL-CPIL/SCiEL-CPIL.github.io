import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_pairs = len(all_pairs)

    # Precompute for each pair of vertices, the list of other vertices
    # This helps with fast K3,3 checking
    # For a triple (i,j,k), K3,3 exists iff |N(i) ∩ N(j) ∩ N(k)| >= 3
    # where N(x) is the neighborhood of x (excluding x itself)

    # Use bitmasks for adjacency
    # adj[i] is a bitmask of neighbors of vertex i

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        """Check if no three vertices have three or more common neighbors."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) would create a K3,3."""
        if adj[u] & (1 << v):
            return True
        
        nu = adj[u]
        nv = adj[v]
        
        # After adding (u,v), u's neighbors become nu | (1<<v)
        # v's neighbors become nv | (1<<u)
        # 
        # A K3,3 would be formed if there exist vertices a,b,c such that
        # |N'(a) ∩ N'(b) ∩ N'(c)| >= 3 where N' is the new neighborhood.
        #
        # The only way adding (u,v) creates a new K3,3 is if the triple
        # includes u or v, and the new common neighbor includes the other.
        #
        # Case 1: Triple includes u but not v.
        #   For vertices x,y (not u,v), if x,y are both neighbors of v,
        #   then N'(u) ∩ N'(x) ∩ N'(y) includes v (new) plus old common neighbors.
        #   Old common neighbors of u,x,y: nu & adj[x] & adj[y]
        #   If this has size 2, adding v makes it 3 -> K3,3
        #
        # Case 2: Triple includes v but not u.
        #   Symmetric: for x,y neighbors of u, if nv & adj[x] & adj[y] has size 2
        #
        # Case 3: Triple includes both u and v.
        #   N'(u) ∩ N'(v) ∩ N'(w) for some w.
        #   N'(u) ∩ N'(v) = (nu | (1<<v)) & (nv | (1<<u))
        #   = (nu & nv) | (nu & (1<<u)) | ((1<<v) & nv) | ((1<<v) & (1<<u))
        #   Since u is not in nu and v is not in nv:
        #   = (nu & nv) | (1<<v) | (1<<u)
        #   Wait, nu & (1<<u) = 0 (u not neighbor of itself), (1<<v) & nv: v not in nv
        #   Actually: (nu | (1<<v)) & (nv | (1<<u)) 
        #   = (nu & nv) | (nu & (1<<u)) | ((1<<v) & nv) | ((1<<v) & (1<<u))
        #   = (nu & nv) | 0 | 0 | 0
        #   = nu & nv
        #   So N'(u) ∩ N'(v) = nu & nv (same as before, plus u and v aren't in each other's old neighborhoods)
        #   Wait, that's not right. Let me reconsider.
        #   N'(u) = nu ∪ {v}, N'(v) = nv ∪ {u}
        #   N'(u) ∩ N'(v) = (nu ∪ {v}) ∩ (nv ∪ {u})
        #   = (nu ∩ nv) ∪ (nu ∩ {u}) ∪ ({v} ∩ nv) ∪ ({v} ∩ {u})
        #   = (nu ∩ nv) ∪ ∅ ∪ ∅ ∪ ∅
        #   = nu ∩ nv
        #   So the intersection of u and v's new neighborhoods is the same as before.
        #   Therefore, for any w, N'(u) ∩ N'(v) ∩ N'(w) = (nu ∩ nv) ∩ N'(w)
        #   N'(w) = adj[w] if w != u,v, or adj[w] ∪ {other} if w is u or v.
        #   Since w is a third vertex, N'(w) = adj[w] (w's neighborhood doesn't change unless w=u or w=v, but w is different).
        #   Actually wait - if w is neither u nor v, adj[w] doesn't change.
        #   So N'(u) ∩ N'(v) ∩ N'(w) = (nu & nv) & adj[w]
        #   This is the same as before adding the edge. So no new K3,3 from triples containing both u and v.
        #
        # So we only need to check cases 1 and 2.

        # Case 1: Triple (u, x, y) where x,y are neighbors of v (so v becomes common neighbor)
        # We need: (nu & adj[x] & adj[y]).bit_count() == 2
        # (because adding v as a new common neighbor makes it 3)
        
        # Iterate over pairs of neighbors of v (excluding u)
        nv_excl_u = nv & ~(1 << u)
        neighbors_of_v = []
        m = nv_excl_u
        while m:
            low = m & (-m)
            idx = low.bit_length() - 1
            neighbors_of_v.append(idx)
            m ^= low
        
        for idx_x in range(len(neighbors_of_v)):
            x = neighbors_of_v[idx_x]
            if x == u:
                continue
            ax = adj[x]
            for idx_y in range(idx_x + 1, len(neighbors_of_v)):
                y = neighbors_of_v[idx_y]
                if y == u:
                    continue
                ay = adj[y]
                common = nu & ax & ay
                if common.bit_count() == 2:
                    return False
        
        # Case 2: Triple (v, x, y) where x,y are neighbors of u (so u becomes common neighbor)
        nu_excl_v = nu & ~(1 << v)
        neighbors_of_u = []
        m = nu_excl_v
        while m:
            low = m & (-m)
            idx = low.bit_length() - 1
            neighbors_of_u.append(idx)
            m ^= low
        
        for idx_x in range(len(neighbors_of_u)):
            x = neighbors_of_u[idx_x]
            if x == v:
                continue
            ax = adj[x]
            for idx_y in range(idx_x + 1, len(neighbors_of_u)):
                y = neighbors_of_u[idx_y]
                if y == v:
                    continue
                ay = adj[y]
                common = nv & ax & ay
                if common.bit_count() == 2:
                    return False
        
        return True

    def can_remove_edge(adj, u, v):
        """Always valid to remove an edge."""
        return True

    def local_search(adj, max_iters=200):
        """Local search: try to add edges, then try swaps."""
        adj = adj[:]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Phase 1: Try to add any missing edge
            added = False
            # Shuffle for diversity
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if adj[u] & (1 << v):
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if added:
                continue
            
            # Phase 2: Try swaps - remove one edge, add another
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            
            if not edges_in or not edges_out:
                break
            
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            
            swapped = False
            # Try more candidates
            for e_rem in edges_in[:50]:
                if swapped:
                    break
                u1, v1 = e_rem
                for e_add in edges_out[:50]:
                    u2, v2 = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    # Remove edge (u1,v1)
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    if can_add_edge(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        swapped = True
                        break
                    else:
                        # Restore
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                if not swapped:
                    continue
            
            if not swapped:
                break
        
        return adj

    def simulated_annealing(adj, max_steps=8000):
        """Simulated annealing to escape local optima."""
        adj = adj[:]
        current_count = count_edges(adj)
        best_count = current_count
        best_adj = adj[:]
        
        T = 3.0
        T_min = 0.01
        cooling = 0.9995
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            if T < T_min:
                break
                
            move_type = random.random()
            
            if move_type < 0.35:
                # Add a random missing edge
                missing = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
                if not missing:
                    break
                u, v = random.choice(missing)
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_valid(adj):
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # Roll back
                    adj[u] = old_u
                    adj[v] = old_v
                    
            elif move_type < 0.65:
                # Remove a random existing edge
                existing = [(i, j) for i, j in all_pairs if (adj[i] & (1 << j))]
                if not existing:
                    break
                u, v = random.choice(existing)
                old_u, old_v = adj[u], adj[v]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Accept with probability based on temperature (always accept removals for exploration)
                current_count -= 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                    
            else:
                # Swap: remove one, add another
                existing = [(i, j) for i, j in all_pairs if (adj[i] & (1 << j))]
                missing = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
                if not existing or not missing:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2):
                    continue
                    
                old_u1, old_v1 = adj[u1], adj[v1]
                old_u2, old_v2 = adj[u2], adj[v2]
                
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                
                if is_valid(adj):
                    # Edge count unchanged
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    adj[u1] = old_u1
                    adj[v1] = old_v1
                    adj[u2] = old_u2
                    adj[v2] = old_v2
            
            T *= cooling
        
        return best_adj

    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Multiple random restarts
    num_trials = 40
    for trial in range(num_trials):
        if time.time() > deadline - 20:
            break
        seed = 12345 + trial * 7919
        random.seed(seed)
        
        if trial % 3 == 0:
            # Greedy construction
            adj = [0] * n
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        elif trial % 3 == 1:
            # Dense random start with repair
            adj = [0] * n
            p = 0.65 + 0.05 * (trial % 5)
            for i, j in all_pairs:
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            # Repair: remove edges to fix violations
            for _ in range(100):
                if is_valid(adj):
                    break
                # Find a violation
                found = False
                for i in range(n):
                    if found:
                        break
                    for j in range(i+1, n):
                        if found:
                            break
                        cij = adj[i] & adj[j]
                        if cij.bit_count() < 3:
                            continue
                        for k in range(j+1, n):
                            cijk = cij & adj[k]
                            if cijk.bit_count() >= 3:
                                # Remove a random edge incident to one of i,j,k
                                # to break the K3,3
                                candidates = []
                                for vert in [i, j, k]:
                                    for nbr in range(n):
                                        if adj[vert] & (1 << nbr):
                                            candidates.append((vert, nbr))
                                if candidates:
                                    u, v = random.choice(candidates)
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                found = True
                                break
        else:
            # Greedy with different seed
            adj = [0] * n
            random.seed(seed + 5000)
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Local search
        adj = local_search(adj, max_iters=150)
        update_best(adj)
        
        # Simulated annealing
        adj = simulated_annealing(adj, max_steps=6000)
        update_best(adj)
        
        # Final local search polish
        adj = local_search(adj, max_iters=200)
        update_best(adj)

    if not best_edges:
        # Fallback: return a simple valid graph
        return [[i, i+1] for i in range(19)]

    return best_edges
