import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Generate Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    # Adjacency matrix for 37 vertices
    # We will work with 42 vertices.
    # Initialize 42x42 matrix with 0s.
    n_target = 42
    adj = [[0] * n_target for _ in range(n_target)]
    
    # Fill first 37x37 with Paley 37
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = 1
                adj[j][i] = 1
                
    # Initialize remaining vertices (37-41) with random edges
    random.seed(42)
    for i in range(q, n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 1
            # Edges to existing vertices
            for k in range(i):
                if random.random() < 0.5:
                    adj[i][k] = 1
                    adj[k][i] = 1

    # Helper to count K5 and independent 5
    # For efficiency in SA, we need incremental update.
    # But first, let's establish a baseline cost.
    
    def count_k5_and_ind5(adj):
        n = len(adj)
        k5_count = 0
        ind5_count = 0
        # Iterate over all 5-combinations
        # This is slow: C(42,5) = 850668
        # Each check is 10 edge lookups.
        # 8.5M lookups. ~0.5s in Python.
        # We can do this once or twice.
        for comb in itertools.combinations(range(n), 5):
            # Check K5
            is_clique = True
            is_indep = True
            for u_idx in range(5):
                u = comb[u_idx]
                for v_idx in range(u_idx + 1, 5):
                    v = comb[v_idx]
                    if adj[u][v] == 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique:
                k5_count += 1
            if is_indep:
                ind5_count += 1
        return k5_count + ind5_count

    # Initial cost
    current_cost = count_k5_and_ind5(adj)
    best_adj = [row[:] for row in adj]
    best_cost = current_cost
    
    # Simulated Annealing
    # We need a fast cost update for edge flips.
    # Delta K5 when flipping (u,v):
    # If we add edge (u,v):
    #   New K5s are formed by (u,v) + 3 common neighbors that form a triangle.
    #   Delta = # triangles in N(u) & N(v)
    # If we remove edge (u,v):
    #   Removed K5s are those containing (u,v).
    #   Delta = - (# triangles in N(u) & N(v))
    
    # Similarly for Independent sets:
    # If we add edge (u,v):
    #   We break independent sets containing (u,v).
    #   An independent set of size 5 containing (u,v) requires 3 common NON-neighbors that form an independent set of size 3 (i.e., no edges).
    #   Delta = - (# independent triangles in (V \ N(u)) & (V \ N(v)) ? No.
    #   Common non-neighbors: w such that adj[u][w]=0 and adj[v][w]=0.
    #   We need 3 such w's that form an independent set in the induced subgraph?
    #   Actually, if u and v are non-adjacent, they are candidates for an independent set.
    #   Adding edge (u,v) destroys independent sets containing {u,v}.
    #   Such an independent set is {u, v, a, b, c} where a,b,c are common non-neighbors of u and v, AND {a,b,c} is an independent set in the graph.
    #   So Delta Ind = - (# independent triangles in the subgraph induced by common non-neighbors).
    
    # This incremental update is complex to implement correctly and efficiently in Python within the time limit.
    # Given the risk of bugs in incremental update leading to incorrect cost and thus invalid acceptance criteria,
    # and the fact that we only have 300s, maybe a simpler approach is better:
    # Just run SA with full recount every step?
    # 8.5M ops per step. 300s / 0.5s per step = 600 steps.
    # 600 steps of SA is likely not enough to escape local minima in a 42-vertex graph.
    
    # Alternative:
    # Use the fact that we just need ONE valid graph.
    # We can try multiple restarts with different random seeds for the initial 5 vertices.
    # Or, we can use a "Greedy Repair" strategy.
    
    # Let's try a different tactic:
    # We know Paley 37 is valid.
    # We can try to find a valid 42-vertex graph by searching the space of graphs that contain Paley 37 as an induced subgraph?
    # No, the 42-vertex graph might not contain Paley 37.
    
    # Let's go with the incremental update SA, but implement it carefully.
    # Precompute neighbor lists?
    
    # Actually, there is a simpler check.
    # If we can't guarantee 42, we return 37.
    # The probability of finding 42 with a naive SA in 300s might be low.
    
    # What if I just hardcode the 42 vertex graph?
    # I don't have the edge list.
    
    # Let's try to generate a 42-vertex graph using a "Construction" method.
    # The Exoo graph is a (42, 4, 4) Ramsey graph.
    # It is often constructed from a (42, 14, 4) design?
    
    # Let's stick to the SA with full recount but optimize the recount using NumPy?
    # NumPy can vectorize the clique check?
    # Not easily for specific combinations.
    
    # Let's try a hybrid:
    # 1. Try to find a valid graph on 42 vertices using SA with incremental updates.
    # 2. If successful, return it.
    # 3. If not, return Paley 37.
    
    # To make incremental updates work, I need to count triangles in induced subgraphs quickly.
    # For n=42, counting triangles in a subgraph of size ~20 is fast.
    
    # Let's implement the incremental update.
    
    def count_triangles_in_subgraph(adj, vertices):
        # vertices is a list of node indices
        # Count triangles in the induced subgraph
        count = 0
        n_v = len(vertices)
        # Map vertices to 0..n_v-1 for local indexing? Or just use global indices.
        # Using global indices is fine.
        for i in range(n_v):
            u = vertices[i]
            for j in range(i + 1, n_v):
                v = vertices[j]
                if adj[u][v]:
                    # Check if there is a w in vertices that connects to both u and v
                    # w must be in vertices
                    # To avoid double counting, w > v? No, just check all w in vertices.
                    # Triangle (u,v,w).
                    # We can just iterate w in vertices.
                    # O(n^3). n_v <= 42. 42^3 = 74k. Fast.
                    for k in range(j + 1, n_v):
                        w = vertices[k]
                        if adj[u][w] and adj[v][w]:
                            count += 1
        return count

    def count_ind_triangles_in_subgraph(adj, vertices):
        # Count independent triangles (no edges) in induced subgraph
        count = 0
        n_v = len(vertices)
        for i in range(n_v):
            u = vertices[i]
            for j in range(i + 1, n_v):
                v = vertices[j]
                if not adj[u][v]:
                    for k in range(j + 1, n_v):
                        w = vertices[k]
                        if not adj[u][w] and not adj[v][w]:
                            count += 1
        return count

    # Precompute common neighbors and non-neighbors for all pairs?
    # This is O(n^2) lists.
    # common_neigh[u][v] = list of w such that adj[u][w] and adj[v][w]
    # common_non_neigh[u][v] = list of w such that not adj[u][w] and not adj[v][w]
    
    # Updating these when an edge is flipped is O(n).
    
    # Let's implement this.
    
    # 1. Build common_neigh and common_non_neigh
    common_neigh = [[[] for _ in range(n_target)] for _ in range(n_target)]
    common_non_neigh = [[[] for _ in range(n_target)] for _ in range(n_target)]
    
    def update_common_structures(u, v, edge_added):
        # When edge (u,v) is flipped, the common neighbors of (u,v) change?
        # No, common neighbors of (u,v) are nodes w connected to BOTH u and v.
        # Flipping (u,v) does NOT change the adjacency of w to u or w to v.
        # So common_neigh[u][v] and common_non_neigh[u][v] do NOT change when (u,v) is flipped.
        # 
        # HOWEVER, flipping (u,v) DOES change the common neighbors of pairs involving u and other nodes?
        # No.
        # Flipping (u,v) changes:
        # 1. The cost contribution of the edge (u,v) itself (triangles involving u,v).
        # 2. The common neighbor lists for pairs (u, w) and (v, w) for all w?
        #    Let's check.
        #    common_neigh[u][w] contains v if adj[u][v] and adj[v][w].
        #    If we flip (u,v), adj[u][v] changes.
        #    So v is added to or removed from common_neigh[u][w] for all w where adj[v][w] is 1.
        #    Similarly for common_neigh[v][w].
        #    So we MUST update common_neigh[u][w] and common_neigh[v][w] for all w.
        #    This is O(n) updates per edge flip.
        #    Each update is adding/removing one element from a list.
        #    So total update time O(n).
        
        # Let's re-evaluate.
        # If we maintain common_neigh[u][v] as a list, updating it for all pairs involving u or v is O(n^2) if we do it naively.
        # But we only need to update the lists for pairs (u, w) and (v, w).
        # There are 2*(n-2) such pairs.
        # For each pair, we check if the other node is a neighbor of the flipped node.
        # This is O(n).
        
        # Let's implement the update.
        
        # We need to know the state of the edge (u,v).
        # If edge_added is True, we just added it.
        
        # For all w != u, v:
        #   If adj[v][w] == 1:
        #       If edge_added: add u to common_neigh[v][w]? No.
        #       common_neigh[v][w] contains x if adj[v][x] and adj[w][x].
        #       We are flipping (u,v).
        #       This affects common_neigh[u][w] and common_neigh[v][w]?
        #       common_neigh[u][w] contains x if adj[u][x] and adj[w][x].
        #       The flip of (u,v) changes adj[u][v].
        #       So v is a neighbor of u.
        #       Does v belong to common_neigh[u][w]?
        #       v belongs if adj[u][v] and adj[w][v].
        #       So if adj[w][v] is 1, then v is added to common_neigh[u][w] when edge (u,v) is added.
        #       Similarly, u is added to common_neigh[v][w] if adj[w][u] is 1.
        
        # So:
        for w in range(n_target):
            if w == u or w == v:
                continue
            if adj[v][w]:
                if edge_added:
                    # v becomes a common neighbor of u and w
                    common_neigh[u][w].append(v)
                    common_neigh[w][u].append(v)
                else:
                    # v is removed
                    if v in common_neigh[u][w]:
                        common_neigh[u][w].remove(v)
                    if v in common_neigh[w][u]:
                        common_neigh[w][u].remove(v)
            if adj[u][w]:
                if edge_added:
                    common_neigh[v][w].append(u)
                    common_neigh[w][v].append(u)
                else:
                    if u in common_neigh[v][w]:
                        common_neigh[v][w].remove(u)
                    if u in common_neigh[w][v]:
                        common_neigh[w][v].remove(u)
                        
            # Similarly for non-neighbors
            if not adj[v][w]:
                if edge_added:
                    common_non_neigh[u][w].append(v)
                    common_non_neigh[w][u].append(v)
                else:
                    if v in common_non_neigh[u][w]:
                        common_non_neigh[u][w].remove(v)
                    if v in common_non_neigh[w][u]:
                        common_non_neigh[w][u].remove(v)
            if not adj[u][w]:
                if edge_added:
                    common_non_neigh[v][w].append(u)
                    common_non_neigh[w][v].append(u)
                else:
                    if u in common_non_neigh[v][w]:
                        common_non_neigh[v][w].remove(u)
                    if u in common_non_neigh[w][v]:
                        common_non_neigh[w][v].remove(u)

    # Initialize common structures
    for u in range(n_target):
        for v in range(u + 1, n_target):
            for w in range(n_target):
                if w == u or w == v: continue
                if adj[u][w] and adj[v][w]:
                    common_neigh[u][v].append(w)
                    common_neigh[v][u].append(w)
                elif not adj[u][w] and not adj[v][w]:
                    common_non_neigh[u][v].append(w)
                    common_non_neigh[v][u].append(w)

    # Current cost
    current_cost = 0
    # Calculate initial cost using the common structures?
    # Sum of triangles in common_neigh[u][v] for all edges (u,v)?
    # No, that counts each triangle 3 times?
    # A K5 is 5 edges.
    # Let's just use the slow count for initialization.
    current_cost = count_k5_and_ind5(adj)
    
    best_adj = [row[:] for row in adj]
    best_cost = current_cost
    
    # SA Loop
    T = 1.0
    cooling = 0.99
    steps = 0
    
    while time.time() < deadline:
        # Pick random edge
        u = random.randint(0, n_target - 1)
        v = random.randint(0, n_target - 1)
        if u == v: continue
        if u > v: u, v = v, u
        
        edge_exists = adj[u][v] == 1
        
        # Calculate Delta Cost
        # Delta K5
        # If adding edge: + triangles in common_neigh[u][v]
        # If removing edge: - triangles in common_neigh[u][v]
        tri_count = count_triangles_in_subgraph(adj, common_neigh[u][v])
        
        # Delta Ind5
        # If adding edge: - ind_triangles in common_non_neigh[u][v]
        # If removing edge: + ind_triangles in common_non_neigh[u][v]
        ind_tri_count = count_ind_triangles_in_subgraph(adj, common_non_neigh[u][v])
        
        if edge_exists:
            # Removing edge
            delta_k5 = -tri_count
            delta_ind = ind_tri_count
        else:
            # Adding edge
            delta_k5 = tri_count
            delta_ind = -ind_tri_count
            
        delta_cost = delta_k5 + delta_ind
        
        # Accept?
        if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
            # Apply move
            adj[u][v] = 1 - adj[u][v]
            adj[v][u] = 1 - adj[u][v]
            
            # Update structures
            update_common_structures(u, v, not edge_exists)
            
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = [row[:] for row in adj]
                if best_cost == 0:
                    # Found valid graph!
                    edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target) if adj[i][j]]
                    return n_target, edges
                    
        T *= cooling
        steps += 1
        
    # If not found, return Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
