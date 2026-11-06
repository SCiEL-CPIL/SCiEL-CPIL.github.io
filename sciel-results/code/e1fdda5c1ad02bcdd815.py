"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def _clebsch_edges():
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                d = bin(i ^ j).count('1')
                if d == 1 or d == 3:
                    edges.append([i, j])
        return edges

    def _build_adj_bits(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def _find_k4(adj, n):
        """Return a K4 as tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                common_ab = adj[a] & adj[b]
                common_ab &= ~(1 << a)
                common_ab &= ~(1 << b)
                c_mask = common_ab
                while c_mask:
                    c = (c_mask & -c_mask).bit_length() - 1
                    c_mask &= c_mask - 1
                    if c <= b:
                        continue
                    common_abc = common_ab & adj[c]
                    common_abc &= ~(1 << a)
                    common_abc &= ~(1 << b)
                    common_abc &= ~(1 << c)
                    d_mask = common_abc
                    while d_mask:
                        d = (d_mask & -d_mask).bit_length() - 1
                        d_mask &= d_mask - 1
                        if d > c:
                            return (a, b, c, d)
        return None

    def _find_ind6(adj, n):
        """Return an independent set of 6 as tuple or None. Uses backtracking."""
        # Complement adjacency: non-edge in original = edge in complement
        # We need clique of size 6 in complement
        comp_adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    comp_adj[i] |= (1 << j)
                    comp_adj[j] |= (1 << i)
        
        # Backtracking to find clique of size 6 in complement
        def _has_clique_k(adj_comp, k):
            """Check if adj_comp has a clique of size k using greedy backtracking."""
            if k <= 1:
                return True
            # Use recursive approach with candidate set
            candidates = list(range(len(adj_comp)))
            
            def _search(cands, depth):
                if depth == k:
                    return True
                if not cands:
                    return False
                # Pruning: if remaining candidates can't fill depth
                if len(cands) < depth:
                    return False
                # Try each candidate
                for idx in range(len(cands)):
                    v = cands[idx]
                    # New candidates must be neighbors of v and come after idx in cands
                    new_cands = []
                    for j in range(idx + 1, len(cands)):
                        u = cands[j]
                        if (adj_comp[v] >> u) & 1:
                            new_cands.append(u)
                    if _search(new_cands, depth + 1):
                        return True
                return False
            
            return _search(candidates, 1)
        
        # For efficiency, use randomized sampling for large n
        if n > 30:
            for _ in range(200):
                sample = random.sample(range(n), 6)
                valid = True
                for i in range(6):
                    for j in range(i+1, 6):
                        if (adj[sample[i]] >> sample[j]) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return tuple(sample)
            return None
        else:
            # Exhaustive check for smaller n
            for combo in itertools.combinations(range(n), 6):
                valid = True
                for i in range(6):
                    for j in range(i+1, 6):
                        if (adj[combo[i]] >> combo[j]) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return combo
            return None

    def _is_valid(adj, n):
        """Check if graph is K4-free and has no independent set of 6."""
        if _find_k4(adj, n) is not None:
            return False
        if _find_ind6(adj, n) is not None:
            return False
        return True

    def _slack_for_vertex(adj, n, new_vertex):
        """
        Compute slack for adding new_vertex with given adjacencies.
        Slack = min(k4_slack, ind_slack) where:
        - k4_slack: how many more neighbors we can add before creating K4
        - ind_slack: how many more non-neighbors we can add before creating Ind6
        """
        # For K4 constraint: new_vertex's neighbors must not form a triangle together
        # with any existing edge. More precisely, among new_vertex's neighbors,
        # there should be no pair (u,v) that are both adjacent to new_vertex
        # and also share a common neighbor w that is adjacent to both u and v
        # (which would create K4: new_vertex, u, v, w).
        
        # Simpler: for each pair of neighbors (u,v) of new_vertex,
        # check if they have a common neighbor w. If so, adding both u and v
        # as neighbors would create K4.
        
        # Actually, let's compute:
        # k4_slack = maximum number of additional edges from new_vertex to existing vertices
        # that we can add without creating a K4.
        
        # ind_slack = maximum number of non-edges from new_vertex to existing vertices
        # that we can add without creating an independent set of 6.
        
        # This is complex. Let's use a simpler heuristic:
        # Count how many "dangerous" configurations exist.
        
        neighbors = []
        non_neighbors = []
        for v in range(new_vertex):
            if (adj[new_vertex] >> v) & 1:
                neighbors.append(v)
            else:
                non_neighbors.append(v)
        
        # K4 risk: among neighbors, count pairs that have common neighbors
        k4_risk = 0
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                u, v = neighbors[i], neighbors[j]
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                common &= ~(1 << new_vertex)
                if common:
                    k4_risk += 1
        
        # Ind6 risk: among non_neighbors, count how many can form independent sets
        # If non_neighbors already contain an independent set of size 5,
        # then adding new_vertex as non-neighbor to all would create Ind6.
        ind_risk = 0
        if len(non_neighbors) >= 5:
            # Check if non_neighbors contain an independent 5-set
            for combo in itertools.combinations(non_neighbors, 5):
                valid = True
                for i in range(5):
                    for j in range(i+1, 5):
                        if (adj[combo[i]] >> combo[j]) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    ind_risk += 1
                    break  # One such set is enough to flag risk
        
        # Slack heuristic: prefer configurations with lower risk
        # Return a score where higher is better (less risky)
        return -k4_risk - 2 * ind_risk  # Ind6 is more critical

    def _greedy_construct(target_n, max_steps=5000):
        """
        Greedily construct a K4-free, no-Ind6 graph on target_n vertices.
        Build vertex by vertex, choosing edges to maximize slack.
        """
        adj = [0] * target_n
        
        # Start with first few vertices as a small valid seed
        # Use a 5-vertex graph: complete graph K5 is K4-free? No, K5 contains K4.
        # Use C5 (cycle of 5): K4-free, independence number 2. Valid.
        # Or use a 5-vertex graph with no K4 and alpha < 6.
        # Simplest: empty graph on 5 vertices has alpha=5 < 6, and no K4. Valid.
        # But that's too sparse. Let's use a better seed.
        
        # Seed: 5 vertices, use a 5-cycle plus one chord to make it denser
        # Actually, let's just start with vertices 0..4 and add edges greedily.
        
        for new_v in range(target_n):
            if time.time() > deadline - 5.0:
                return None
            
            if new_v == 0:
                continue
            
            # Try different adjacency patterns and pick the best
            best_score = -float('inf')
            best_pattern = None
            
            # Generate candidate patterns: random subsets of existing vertices
            num_candidates = min(20, 1 << new_v)
            for _ in range(num_candidates):
                # Random subset of existing vertices to connect to
                candidate_neighbors = set()
                for v in range(new_v):
                    if random.random() < 0.5:
                        candidate_neighbors.add(v)
                
                # Apply this pattern
                for v in range(new_v):
                    if v in candidate_neighbors:
                        adj[new_v] |= (1 << v)
                        adj[v] |= (1 << new_v)
                    else:
                        adj[new_v] &= ~(1 << v)
                        adj[v] &= ~(1 << new_v)
                
                # Check if this creates a K4
                k4 = _find_k4(adj, new_v + 1)
                if k4 is not None:
                    # Undo and skip
                    for v in range(new_v):
                        if v in candidate_neighbors:
                            adj[new_v] &= ~(1 << v)
                            adj[v] &= ~(1 << new_v)
                    continue
                
                # Check if this creates an Ind6
                ind6 = _find_ind6(adj, new_v + 1)
                if ind6 is not None:
                    # Undo and skip
                    for v in range(new_v):
                        if v in candidate_neighbors:
                            adj[new_v] &= ~(1 << v)
                            adj[v] &= ~(1 << new_v)
                    continue
                
                # Compute slack
                score = _slack_for_vertex(adj, new_v + 1, new_v)
                if score > best_score:
                    best_score = score
                    best_pattern = set(candidate_neighbors)
                
                # Undo for next candidate
                for v in range(new_v):
                    if v in candidate_neighbors:
                        adj[new_v] &= ~(1 << v)
                        adj[v] &= ~(1 << new_v)
            
            # Apply best pattern
            if best_pattern is not None:
                for v in range(new_v):
                    if v in best_pattern:
                        adj[new_v] |= (1 << v)
                        adj[v] |= (1 << new_v)
                    else:
                        adj[new_v] &= ~(1 << v)
                        adj[v] &= ~(1 << new_v)
            else:
                # No valid pattern found; try connecting to no one
                for v in range(new_v):
                    adj[new_v] &= ~(1 << v)
                    adj[v] &= ~(1 << new_v)
                
                # Check if this is valid
                k4 = _find_k4(adj, new_v + 1)
                ind6 = _find_ind6(adj, new_v + 1)
                if k4 is not None or ind6 is not None:
                    return None
        
        # Final validation
        if not _is_valid(adj, target_n):
            return None
        
        return adj

    def _local_search(adj, n, max_steps=2000):
        """
        Local search: flip edges to eliminate violations.
        Returns improved adj or None if failed.
        """
        adj = adj[:]  # Copy
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                return None
            
            # Check for K4
            k4 = _find_k4(adj, n)
            if k4 is not None:
                a, b, c, d = k4
                # Flip a random edge in this K4
                verts = [a, b, c, d]
                random.shuffle(verts)
                u, v = verts[0], verts[1]
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue
            
            # Check for Ind6
            ind6 = _find_ind6(adj, n)
            if ind6 is not None:
                verts = list(ind6)
                random.shuffle(verts)
                u, v = verts[0], verts[1]
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue
            
            # No violations
            return adj
        
        # Still has violations after max_steps
        if _is_valid(adj, n):
            return adj
        return None

    # Start with Clebsch graph as base
    best_n = 16
    best_edges = _clebsch_edges()
    adj_base = _build_adj_bits(best_n, best_edges)
    
    if not _is_valid(adj_base, best_n):
        # Fallback: empty graph on 5 vertices
        best_n = 5
        best_edges = []
    
    # Try to extend to larger n using greedy construction + local search
    for target_n in range(17, 36):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline - 10.0:
                break
            
            # Greedy construction
            adj = _greedy_construct(target_n)
            if adj is None:
                continue
            
            # Local search to clean up any remaining issues
            adj2 = _local_search(adj, target_n, max_steps=1000)
            if adj2 is not None and _is_valid(adj2, target_n):
                # Extract edges
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (adj2[i] >> j) & 1:
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                found = True
                break
        
        if not found:
            # Try one more time with a different approach: start from current best and add vertices
            if time.time() > deadline - 10.0:
                break
            
            # Extend from current best graph
            adj_ext = _build_adj_bits(best_n, best_edges)
            # Add new vertices one by one
            success = True
            for new_v in range(best_n, target_n):
                if time.time() > deadline - 8.0:
                    success = False
                    break
                
                # Try to add new_v with various adjacency patterns
                best_score = -float('inf')
                best_pattern = None
                
                for _ in range(15):
                    candidate_neighbors = set()
                    for v in range(new_v):
                        if random.random() < 0.5:
                            candidate_neighbors.add(v)
                    
                    # Apply pattern
                    adj_ext.extend([0])  # Add new vertex
                    for v in range(new_v):
                        if v in candidate_neighbors:
                            adj_ext[new_v] |= (1 << v)
                            adj_ext[v] |= (1 << new_v)
                        else:
                            adj_ext[new_v] &= ~(1 << v)
                            adj_ext[v] &= ~(1 << new_v)
                    
                    # Check validity
                    k4 = _find_k4(adj_ext, new_v + 1)
                    ind6 = _find_ind6(adj_ext, new_v + 1)
                    if k4 is not None or ind6 is not None:
                        # Undo
                        for v in range(new_v):
                            if v in candidate_neighbors:
                                adj_ext[new_v] &= ~(1 << v)
                                adj_ext[v] &= ~(1 << new_v)
                        continue
                    
                    score = _slack_for_vertex(adj_ext, new_v + 1, new_v)
                    if score > best_score:
                        best_score = score
                        best_pattern = set(candidate_neighbors)
                    
                    # Undo
                    for v in range(new_v):
                        if v in candidate_neighbors:
                            adj_ext[new_v] &= ~(1 << v)
                            adj_ext[v] &= ~(1 << new_v)
                
                if best_pattern is not None:
                    for v in range(new_v):
                        if v in best_pattern:
                            adj_ext[new_v] |= (1 << v)
                            adj_ext[v] |= (1 << new_v)
                        else:
                            adj_ext[new_v] &= ~(1 << v)
                            adj_ext[v] &= ~(1 << new_v)
                else:
                    # Connect to no one
                    for v in range(new_v):
                        adj_ext[new_v] &= ~(1 << v)
                        adj_ext[v] &= ~(1 << new_v)
                    
                    k4 = _find_k4(adj_ext, new_v + 1)
                    ind6 = _find_ind6(adj_ext, new_v + 1)
                    if k4 is not None or ind6 is not None:
                        success = False
                        break
            
            if success and _is_valid(adj_ext, target_n):
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (adj_ext[i] >> j) & 1:
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                found = True
        
        if not found:
            break
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
