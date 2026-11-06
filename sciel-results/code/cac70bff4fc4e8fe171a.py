"""Combined approach: Paley(17) initialization + refined local search for K4-free graph optimization."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n = 18
    
    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def has_k4(adj):
        """Check if graph contains a K4 using bitmask operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        return True
        return False
    
    def find_k4(adj):
        """Find a K4 in the graph, return tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        d_bit = d_candidates & (-d_candidates)
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None
    
    def max_triangle_free_size(adj, time_limit=5.0):
        """Find largest induced triangle-free subgraph using branch and bound with pruning."""
        best = [0]
        t0 = time.time()
        timeout = [False]
        
        neigh = adj[:]
        
        # Precompute conflict lists for faster triangle detection
        # conflict_lists[v] contains (u, mask) such that if u is in current set, 
        # any vertex in mask cannot be added with v.
        # Actually, simpler: for each v, precompute neighbors of neighbors that form triangles with v?
        # If we add v, we cannot add any pair (u,w) such that u~w, u~v, w~v.
        # This is equivalent to: if we add v, we must remove all edges from the remaining candidate set that are incident to neighbors of v?
        # No, induced subgraph. 
        # Constraint: The induced subgraph on S must be triangle-free.
        # If v in S, then for any u, w in S \ {v}, u and w cannot be adjacent if both are adjacent to v?
        # No, triangle (v,u,w) exists if v~u, v~w, u~w.
        # So if v is included, we cannot include any edge (u,w) from the remaining graph where u~v and w~v.
        
        # Branch and Bound:
        # State: (idx, current_mask, current_size)
        # Order vertices by degree (descending) to prune faster? Or ascending?
        # High degree vertices create more conflicts, so processing them early might prune branches where we include them and force exclusions.
        
        deg = [bin(adj[v]).count('1') for v in range(n)]
        # Sort by degree descending
        order = sorted(range(n), key=lambda v: -deg[v])
        
        # Precompute for each vertex v in order, the list of vertices that form a triangle with v and any other vertex?
        # Let's just check on the fly but efficiently.
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - idx) <= best[0]:
                return
            
            v = order[idx]
            
            # Option 1: Exclude v
            backtrack(idx + 1, current_mask, current_size)
            
            # Option 2: Include v
            if not timeout[0]:
                can_include = True
                # Check if adding v creates a triangle with any pair in current_mask
                # Triangle (v, u, w) exists if u, w in current_mask, u~v, w~v, u~w.
                # We can check this by iterating u in current_mask neighbors of v.
                # If u is in current_mask and u~v, then we must ensure no neighbor of u is also in current_mask and neighbor of v?
                # Simpler: 
                # For each u in current_mask:
                #   if u~v:
                #      for each w in current_mask:
                #         if w~v and w~u: triangle found.
                
                # Optimization: Precompute "triangle partners" for v?
                # Let N_v = neighbors of v.
                # If we include v, we cannot include any edge from the subgraph induced by N_v intersect current_mask.
                # So, let C = current_mask & N_v.
                # If there is any edge (u,w) with u,w in C, then we cannot include v.
                
                N_v_mask = adj[v]
                C = current_mask & N_v_mask
                
                # Check if subgraph induced by C has any edge
                # Iterate u in C
                tmp_c = C
                valid = True
                while tmp_c:
                    bit = tmp_c & (-tmp_c)
                    u = bit.bit_length() - 1
                    tmp_c ^= bit
                    # Check if u has any neighbor in C
                    if adj[u] & C:
                        valid = False
                        break
                
                if valid:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
        
        backtrack(0, 0, 0)
        return best[0]
    
    def make_k4_free(adj):
        """Remove edges to make graph K4-free. Returns True if successful."""
        for _ in range(500):
            k4 = find_k4(adj)
            if k4 is None:
                return True
            a, b, c, d = k4
            # Remove a random edge from the K4
            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
            x, y = pair
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
        return not has_k4(adj)
    
    def construct_paley_17():
        """Construct Paley(17) graph on vertices 0..16, and vertex 17 isolated."""
        adj = [0] * n
        # Quadratic residues modulo 17:
        # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
        residues = {1, 4, 9, 16, 8, 2, 15, 13}
        
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Vertex 17 is isolated
        return adj
    
    def edge_swap(adj, u, v, w, x):
        """Perform edge swap: remove (u,v) and (w,x), add (u,w) and (v,x)."""
        if u == v or w == x:
            return False
        if u == w or u == x or v == w or v == x:
            return False
        if not ((adj[u] >> v) & 1):
            return False
        if not ((adj[w] >> x) & 1):
            return False
        if (adj[u] >> w) & 1:
            return False
        if (adj[v] >> x) & 1:
            return False
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[w] &= ~(1 << x)
        adj[x] &= ~(1 << w)
        adj[u] |= (1 << w)
        adj[w] |= (1 << u)
        adj[v] |= (1 << x)
        adj[x] |= (1 << v)
        return True
    
    random.seed(42)
    candidates = []
    
    # Strategy 1: Paley(17) + isolated vertex
    adj_paley = construct_paley_17()
    if make_k4_free(adj_paley):
        candidates.append(adj_paley[:])
    
    # Strategy 2: Random perturbations of Paley(17)
    for trial in range(50):
        if time.time() > deadline - 50:
            break
        adj = adj_paley[:]
        # Flip 1-3 random edges
        num_flips = random.randint(1, 3)
        for _ in range(num_flips):
            i = random.randint(0, 16) # Avoid vertex 17 initially to keep structure
            j = random.randint(0, 16)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        if make_k4_free(adj):
            candidates.append(adj[:])
    
    # Strategy 3: Other regular graphs if Paley fails or for diversity
    for degree in [6, 7, 8]:
        for trial in range(10):
            if time.time() > deadline - 50:
                break
            adj = [0] * n
            # Random regular graph generation (simplified)
            # Just use random graph and make K4 free
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < degree / (n - 1):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if make_k4_free(adj):
                candidates.append(adj[:])

    # Evaluate all candidates
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 30:
            break
        # Use a tighter time limit for initial screening
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=2.0)
    
    # Local search with simulated annealing
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score
    
    temperature = 1.0
    cooling = 0.995
    max_iter = 8000
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges
    
    for iteration in range(max_iter):
        if time.time() > deadline - 15:
            break
        if best_score_sa <= 6:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Single edge flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_adj = adj[:]
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            
            if has_k4(adj):
                adj = old_adj
                continue
            
            score = max_triangle_free_size(adj, time_limit=1.5)
            delta = score - current_score
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                exp_val = math.exp(-delta / max(temperature, 0.01))
                if random.random() < exp_val:
                    accept = True
            
            if accept:
                current_score = score
                if score < best_score_sa:
                    best_score_sa = score
                    best_adj_sa = adj[:]
            else:
                adj = old_adj
        
        elif move_type < 0.9:
            # Edge swap
            edges = get_edges(adj)
            if len(edges) < 2:
                continue
            
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            u, v = e1
            w, x = e2
            
            old_adj = adj[:]
            if edge_swap(adj, u, v, w, x):
                if not has_k4(adj):
                    score = max_triangle_free_size(adj, time_limit=1.5)
                    delta = score - current_score
                    
                    accept = False
                    if delta <= 0:
                        accept = True
                    else:
                        exp_val = math.exp(-delta / max(temperature, 0.01))
                        if random.random() < exp_val:
                            accept = True
                    
                    if accept:
                        current_score = score
                        if score < best_score_sa:
                            best_score_sa = score
                            best_adj_sa = adj[:]
                    else:
                        adj = old_adj
                else:
                    adj = old_adj
            else:
                adj = old_adj
        else:
            # Multi-edge flip
            num_flips = random.randint(2, 3)
            old_adj = adj[:]
            for _ in range(num_flips):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            
            if has_k4(adj):
                adj = old_adj
                continue
            
            score = max_triangle_free_size(adj, time_limit=1.5)
            delta = score - current_score
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                exp_val = math.exp(-delta / max(temperature, 0.01))
                if random.random() < exp_val:
                    accept = True
            
            if accept:
                current_score = score
                if score < best_score_sa:
                    best_score_sa = score
                    best_adj_sa = adj[:]
            else:
                adj = old_adj
        
        temperature *= cooling
    
    # Final evaluation with more time
    final_score = max_triangle_free_size(best_adj_sa, time_limit=20.0)
    
    return edges_from_adj(best_adj_sa)
