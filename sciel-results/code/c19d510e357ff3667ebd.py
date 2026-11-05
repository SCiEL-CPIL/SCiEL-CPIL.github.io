"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 300  # 300s wall-clock limit; use 290s safety margin
    safety_margin = 10
    
    def time_left():
        return deadline - time.time() - safety_margin
    
    def check_timeout():
        return time.time() > deadline - safety_margin
    
    # Build Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    n = q
    # adj[i] is a set of neighbors of vertex i
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    def has_clique_k(adj, k, vertex_set=None):
        """Check if graph has a clique of size k. If vertex_set is given, only check within that set."""
        if vertex_set is None:
            vertex_set = set(range(len(adj)))
        vs = sorted(vertex_set)
        if len(vs) < k:
            return False
        if k == 1:
            return len(vs) >= 1
        if k == 2:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] in adj[vs[i]]:
                        return True
            return False
        if k == 3:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] in adj[vs[i]]:
                        common = adj[vs[i]] & adj[vs[j]] & vertex_set
                        for m in vs[j+1:]:
                            if m in common:
                                return True
            return False
        if k == 4:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] in adj[vs[i]]:
                        common_ij = adj[vs[i]] & adj[vs[j]] & vertex_set
                        for m in vs[j+1:]:
                            if m in common_ij:
                                common_ijk = common_ij & adj[m]
                                for p in vs[vs.index(m)+1:]:
                                    if p in common_ijk:
                                        return True
            return False
        if k == 5:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] in adj[vs[i]]:
                        common_ij = adj[vs[i]] & adj[vs[j]] & vertex_set
                        for m in vs[j+1:]:
                            if m in common_ij:
                                common_ijk = common_ij & adj[m]
                                for p in vs[vs.index(m)+1:]:
                                    if p in common_ijk:
                                        common_ijkl = common_ijk & adj[p]
                                        for r in vs[vs.index(p)+1:]:
                                            if r in common_ijkl:
                                                return True
            return False
        return False
    
    def has_independent_k(adj, k, vertex_set=None):
        """Check if graph has an independent set of size k. If vertex_set is given, only check within that set."""
        if vertex_set is None:
            vertex_set = set(range(len(adj)))
        # Independent set in original = clique in complement
        # Build complement adjacency within vertex_set
        vs = sorted(vertex_set)
        if len(vs) < k:
            return False
        if k == 1:
            return len(vs) >= 1
        if k == 2:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        return True
            return False
        if k == 3:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        common_non = [x for x in vs[j+1:] if x not in adj[vs[i]] and x not in adj[vs[j]]]
                        if common_non:
                            return True
            return False
        if k == 4:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        common_non_ij = [x for x in vs[j+1:] if x not in adj[vs[i]] and x not in adj[vs[j]]]
                        for m_idx, m in enumerate(common_non_ij):
                            common_non_ijk = [x for x in common_non_ij[m_idx+1:] if x not in adj[m]]
                            for p in common_non_ijk:
                                return True
            return False
        if k == 5:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        common_non_ij = [x for x in vs[j+1:] if x not in adj[vs[i]] and x not in adj[vs[j]]]
                        for m_idx, m in enumerate(common_non_ij):
                            common_non_ijk = [x for x in common_non_ij[m_idx+1:] if x not in adj[m]]
                            for p_idx, p in enumerate(common_non_ijk):
                                common_non_ijkl = [x for x in common_non_ijk[p_idx+1:] if x not in adj[p]]
                                for r in common_non_ijkl:
                                    return True
            return False
        return False
    
    def is_valid_addition_fast(adj, n, new_neighbor_set):
        """Fast check: verify no K4 in neighbors and no independent 4-set in non-neighbors."""
        # 1. Check no K4 in new_neighbor_set (would form K5 with new vertex)
        if new_neighbor_set:
            nbrs = sorted(new_neighbor_set)
            if len(nbrs) >= 4:
                # Check all 4-subsets for clique
                for a in range(len(nbrs)):
                    na = nbrs[a]
                    for b in range(a+1, len(nbrs)):
                        nb = nbrs[b]
                        if nb not in adj[na]:
                            continue
                        # Find common neighbors of na, nb within nbrs
                        common_ab = adj[na] & adj[nb]
                        for c in range(b+1, len(nbrs)):
                            nc = nbrs[c]
                            if nc not in common_ab:
                                continue
                        # For each valid (a,b,c), check if any d forms K4
                        for d in range(c+1, len(nbrs)):
                            nd = nbrs[d]
                            if nd in common_ab and nd in adj[nc]:
                                return False
        # 2. Check no independent 4-set in non_neighbors (would form independent 5-set)
        non_neighbors = set(range(n)) - new_neighbor_set
        if non_neighbors:
            non_nbrs = sorted(non_neighbors)
            if len(non_nbrs) >= 4:
                # Check all 4-subsets for independent set (no edges between them)
                for a in range(len(non_nbrs)):
                    na = non_nbrs[a]
                    for b in range(a+1, len(non_nbrs)):
                        nb = non_nbrs[b]
                        if nb in adj[na]:
                            continue
                        for c in range(b+1, len(non_nbrs)):
                            nc = non_nbrs[c]
                            if nc in adj[na] or nc in adj[nb]:
                                continue
                            for d in range(c+1, len(non_nbrs)):
                                nd = non_nbrs[d]
                                if nd in adj[na] or nd in adj[nb] or nd in adj[nc]:
                                    continue
                                return False
        return True
    
    def try_add_vertex(adj, n, max_attempts=200000):
        """Try to find a valid connection pattern for a new vertex n."""
        all_vertices = list(range(n))
        
        # First try: connect to all (check if valid)
        if is_valid_addition_fast(adj, n, set(all_vertices)):
            return set(all_vertices)
        
        # Try: connect to none
        if is_valid_addition_fast(adj, n, set()):
            return set()
        
        # Strategy 1: Try all single-vertex additions (add one edge at a time from empty)
        # Start from empty and add vertices one by one
        S = set()
        for v in all_vertices:
            S_trial = S | {v}
            if is_valid_addition_fast(adj, n, S_trial):
                S = S_trial
        if is_valid_addition_fast(adj, n, S):
            return S
        
        # Strategy 2: Try removing vertices from full set
        S = set(all_vertices)
        for v in all_vertices:
            S_trial = S - {v}
            if is_valid_addition_fast(adj, n, S_trial):
                S = S_trial
        if is_valid_addition_fast(adj, n, S):
            return S
        
        # Strategy 3: Randomized local search
        attempts = 0
        while attempts < max_attempts:
            if check_timeout():
                break
            
            # Random start with varying density
            density = random.uniform(0.3, 0.7)
            S = set(v for v in all_vertices if random.random() < density)
            
            # Local search: try adding/removing single vertices
            improved = True
            local_steps = 0
            while improved and local_steps < 100:
                improved = False
                local_steps += 1
                if check_timeout():
                    break
                
                # Try adding a random vertex
                candidates_to_add = [v for v in all_vertices if v not in S]
                if candidates_to_add:
                    v = random.choice(candidates_to_add)
                    S_trial = S | {v}
                    if is_valid_addition_fast(adj, n, S_trial):
                        S = S_trial
                        improved = True
                        continue
                
                # Try removing a random vertex
                if S:
                    v = random.choice(list(S))
                    S_trial = S - {v}
                    if is_valid_addition_fast(adj, n, S_trial):
                        S = S_trial
                        improved = True
            
            if is_valid_addition_fast(adj, n, S):
                return S
            attempts += 1
        
        # Strategy 4: Systematic bias sweep
        for density in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            if check_timeout():
                break
            S = set(v for v in all_vertices if random.random() < density)
            if is_valid_addition_fast(adj, n, S):
                return S
            # Local improvement
            for _ in range(50):
                if check_timeout():
                    break
                candidates_to_add = [v for v in all_vertices if v not in S]
                if candidates_to_add:
                    v = random.choice(candidates_to_add)
                    S_trial = S | {v}
                    if is_valid_addition_fast(adj, n, S_trial):
                        S = S_trial
                if S:
                    v = random.choice(list(S))
                    S_trial = S - {v}
                    if is_valid_addition_fast(adj, n, S_trial):
                        S = S_trial
            if is_valid_addition_fast(adj, n, S):
                return S
        
        return None
    
    # Try to extend the graph with backtracking
    max_target = 45  # We know upper bound is 46, so try up to 45
    current_n = n
    
    def backtrack_extend(adj, current_n, max_target):
        """Recursively try to extend graph with backtracking."""
        if check_timeout():
            return current_n
        if current_n >= max_target:
            return current_n
        
        # Try multiple random restarts for this vertex
        best_result = current_n
        for restart in range(10):
            if check_timeout():
                break
            added = try_add_vertex(adj, current_n)
            if added is not None:
                # Save state
                saved_adj = [set(s) for s in adj]
                new_idx = current_n
                adj.append(set(added))
                for v in added:
                    adj[v].add(new_idx)
                
                # Recursively try to extend further
                result = backtrack_extend(adj, current_n + 1, max_target)
                if result > best_result:
                    best_result = result
                
                # Restore state if we want to try different connections
                # But keep the best result
                if result < max_target:
                    # Restore and try different connection
                    adj = saved_adj
                    # Shuffle for different random choices
                    random.seed(random.randint(0, 1000000))
        
        return best_result
    
    # Use backtracking to find best extension
    current_n = backtrack_extend(adj, current_n, max_target)
    
    # Final validation: ensure no K5 and no independent 5-set
    if has_clique_k(adj, 5) or has_independent_k(adj, 5):
        # Fallback to original Paley graph if extension created invalid graph
        adj = [set() for _ in range(q)]
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
        current_n = q
    
    # Build edge list
    edges = []
    for i in range(current_n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return current_n, edges
# EVOLVE-BLOCK-END
