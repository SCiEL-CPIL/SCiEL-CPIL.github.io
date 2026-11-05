import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_vertices = list(range(n))
    all_edges = list(itertools.combinations(range(n), 2))
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    edge_index.update({(v, u): i for i, (u, v) in enumerate(all_edges)})
    
    # Q3 structure
    # Vertices 0..7
    # Edges:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    
    Q3_ADJ = [
        [1, 2, 4],
        [0, 3, 5],
        [0, 3, 6],
        [1, 2, 7],
        [0, 5, 6],
        [1, 4, 7],
        [2, 4, 7],
        [3, 5, 6]
    ]
    
    Q3_EDGES = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Precompute all valid embeddings of Q3 into 16 vertices
    # An embedding is a tuple (v0, v1, ..., v7) of distinct vertices
    # such that for every edge (u, v) in Q3_EDGES, (v_u, v_v) is an edge in K16 (always true)
    # But we need to check if a specific graph G contains this embedding.
    # Instead of precomputing embeddings, we precompute the "patterns" of edges.
    # Actually, let's precompute all sets of 12 edges in K16 that form a Q3.
    # There are 12870 subsets of 8 vertices.
    # For each subset, check if it induces a Q3 (or contains a Q3).
    # Since Q3 is 3-regular on 8 vertices, if the induced subgraph is exactly Q3, it's a witness.
    # If the induced subgraph has extra edges, it still contains Q3.
    # We want to find *any* Q3 subgraph.
    
    # Strategy:
    # 1. Generate all permutations of 8 vertices from 16? No, too many.
    # 2. Generate all combinations of 8 vertices.
    # 3. For each combination, check if it contains a Q3.
    #    To check if a set of 8 vertices contains a Q3, we can try all isomorphisms.
    #    There are 8! = 40320 isomorphisms.
    #    For each isomorphism, check if all 12 edges are present.
    #    This is feasible for 12870 subsets * 40320 * 12 ~ 6e9 ops. Too slow.
    
    # Better: Precompute all Q3 subgraphs in K16.
    # A Q3 subgraph is defined by a mapping of {0..7} to {0..15} such that edges map to edges.
    # We can iterate over all mappings? 16P8 is huge.
    
    # Alternative:
    # Use the "backtrack" approach but optimize it heavily.
    # Or, precompute the list of all edge-sets that form a Q3.
    # Let's try to generate them.
    # Fix vertex 0 of Q3 to vertex 0 of G? No, Q3 can be anywhere.
    
    # Let's stick to the dynamic check but optimize the check itself.
    # The current `find_blocking_q3` is decent but can be improved.
    # Key insight: When adding edge (u, v), we only need to check Q3s that *use* (u, v).
    # A Q3 uses (u, v) if u and v are mapped to adjacent vertices in Q3.
    # There are 12 edges in Q3.
    # For each edge (i, j) in Q3, we can try to map u->i, v->j and u->j, v->i.
    # Then we need to map the remaining 6 Q3 vertices to remaining G vertices such that all other edges exist.
    
    # Let's implement a fast check:
    def has_q3_with_edge(u, v, adj_mask):
        # adj_mask is a list of integers, where adj_mask[i] is a bitmask of neighbors of i.
        # Check if adding (u, v) creates a Q3.
        # This is equivalent to checking if there is a Q3 in the current graph + (u, v) that uses (u, v).
        
        # For each edge (i, j) in Q3:
        for i, j in Q3_EDGES:
            # Case 1: u maps to i, v maps to j
            if _check_q3_extension(i, u, j, v, adj_mask):
                return True
            # Case 2: u maps to j, v maps to i
            if _check_q3_extension(j, u, i, v, adj_mask):
                return True
        
        return False

    def _check_q3_extension(q3_u, g_u, q3_v, g_v, adj_mask):
        # We have mapped q3_u -> g_u and q3_v -> g_v.
        # We need to map the remaining 6 Q3 vertices to G vertices.
        # The remaining Q3 vertices are those not in {q3_u, q3_v}.
        # The remaining G vertices are those not in {g_u, g_v}.
        # Constraints:
        # 1. For any edge (a, b) in Q3 where a is mapped and b is not, g_b must be a neighbor of g_a in G.
        # 2. For any edge (a, b) in Q3 where both are unmapped, g_a and g_b must be adjacent in G.
        
        # Let's use backtracking.
        # State:
        # - mapping: dict q3_vertex -> g_vertex
        # - available_g: set of available g_vertices
        
        # Initial state:
        mapping = {q3_u: g_u, q3_v: g_v}
        used_g = {g_u, g_v}
        available_g = set(range(n)) - used_g
        
        # Order of assignment:
        # Pick the Q3 vertex with the most constraints (most neighbors already mapped).
        
        def backtrack():
            if len(mapping) == 8:
                return True
            
            # Find unassigned Q3 vertex with most mapped neighbors
            unassigned = [x for x in range(8) if x not in mapping]
            
            # Sort by number of mapped neighbors
            def constraint_score(q3_x):
                score = 0
                for nb in Q3_ADJ[q3_x]:
                    if nb in mapping:
                        score += 1
                return score
            
            unassigned.sort(key=constraint_score, reverse=True)
            next_q3 = unassigned[0]
            
            # Filter candidates for next_q3
            candidates = []
            for g_cand in available_g:
                valid = True
                for nb in Q3_ADJ[next_q3]:
                    if nb in mapping:
                        g_nb = mapping[nb]
                        if not (adj_mask[g_nb] >> g_cand) & 1:
                            valid = False
                            break
                if valid:
                    candidates.append(g_cand)
            
            for g_cand in candidates:
                mapping[next_q3] = g_cand
                available_g.remove(g_cand)
                
                if backtrack():
                    return True
                
                del mapping[next_q3]
                available_g.add(g_cand)
            
            return False
        
        return backtrack()

    def count_edges(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2

    def greedy_hitting_set(families):
        if not families:
            return set()
        
        all_elements = set()
        for f in families:
            all_elements.update(f)
        
        if not all_elements:
            return set()
            
        uncovered = list(families)
        solution = set()
        
        while uncovered:
            best_elem = None
            max_count = 0
            candidate_elements = set()
            for f in uncovered:
                candidate_elements.update(f)
                
            for e in candidate_elements:
                count = 0
                for f in uncovered:
                    if e in f:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_elem = e
            
            if best_elem is None or max_count == 0:
                break
                
            solution.add(best_elem)
            
            new_uncovered = []
            for f in uncovered:
                if best_elem not in f:
                    new_uncovered.append(f)
            uncovered = new_uncovered
            
        return solution

    def find_blocking_q3(u, v, adj_mask):
        """
        Returns a list of edges forming a Q3 that blocks (u,v), or None.
        """
        for i, j in Q3_EDGES:
            # Case 1: u->i, v->j
            if _find_q3_extension(i, u, j, v, adj_mask):
                return _reconstruct_q3_edges(i, u, j, v, adj_mask)
            # Case 2: u->j, v->i
            if _find_q3_extension(j, u, i, v, adj_mask):
                return _reconstruct_q3_edges(j, u, i, v, adj_mask)
        return None

    def _find_q3_extension(q3_u, g_u, q3_v, g_v, adj_mask):
        # Similar to _check_q3_extension but returns the mapping
        mapping = {q3_u: g_u, q3_v: g_v}
        used_g = {g_u, g_v}
        available_g = set(range(n)) - used_g
        
        def backtrack():
            if len(mapping) == 8:
                return True
            
            unassigned = [x for x in range(8) if x not in mapping]
            def constraint_score(q3_x):
                score = 0
                for nb in Q3_ADJ[q3_x]:
                    if nb in mapping:
                        score += 1
                return score
            
            unassigned.sort(key=constraint_score, reverse=True)
            next_q3 = unassigned[0]
            
            candidates = []
            for g_cand in available_g:
                valid = True
                for nb in Q3_ADJ[next_q3]:
                    if nb in mapping:
                        g_nb = mapping[nb]
                        if not (adj_mask[g_nb] >> g_cand) & 1:
                            valid = False
                            break
                if valid:
                    candidates.append(g_cand)
            
            for g_cand in candidates:
                mapping[next_q3] = g_cand
                available_g.remove(g_cand)
                
                if backtrack():
                    return True
                
                del mapping[next_q3]
                available_g.add(g_cand)
            
            return False
        
        return backtrack()

    def _reconstruct_q3_edges(q3_u, g_u, q3_v, g_v, adj_mask):
        # Re-run the search to get the mapping
        mapping = {q3_u: g_u, q3_v: g_v}
        used_g = {g_u, g_v}
        available_g = set(range(n)) - used_g
        
        def backtrack():
            if len(mapping) == 8:
                return True
            
            unassigned = [x for x in range(8) if x not in mapping]
            def constraint_score(q3_x):
                score = 0
                for nb in Q3_ADJ[q3_x]:
                    if nb in mapping:
                        score += 1
                return score
            
            unassigned.sort(key=constraint_score, reverse=True)
            next_q3 = unassigned[0]
            
            candidates = []
            for g_cand in available_g:
                valid = True
                for nb in Q3_ADJ[next_q3]:
                    if nb in mapping:
                        g_nb = mapping[nb]
                        if not (adj_mask[g_nb] >> g_cand) & 1:
                            valid = False
                            break
                if valid:
                    candidates.append(g_cand)
            
            for g_cand in candidates:
                mapping[next_q3] = g_cand
                available_g.remove(g_cand)
                
                if backtrack():
                    return True
                
                del mapping[next_q3]
                available_g.add(g_cand)
            
            return False
        
        if backtrack():
            edges = []
            for u, v in Q3_EDGES:
                gu = mapping[u]
                gv = mapping[v]
                edges.append((min(gu, gv), max(gu, gv)))
            return edges
        return None

    def random_greedy_init(seed):
        random.seed(seed)
        adj_mask = [0] * n
        edges = list(all_edges)
        random.shuffle(edges)
        
        for u, v in edges:
            if not has_q3_with_edge(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        return adj_mask

    best_adj = None
    best_count = 0
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        adj = random_greedy_init(r)
        current_count = count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        stuck_iterations = 0
        max_stuck = 50
        
        while time.time() < deadline and stuck_iterations < max_stuck:
            missing = []
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
            
            if not missing:
                break
            
            random.shuffle(missing)
            
            added = False
            for u, v in missing[:20]:
                if not has_q3_with_edge(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
                    added = True
                    stuck_iterations = 0
                    break
            
            if not added:
                stuck_iterations += 1
                
                # Smart Kick
                blockers = []
                for u, v in missing[:10]:
                    q3_edges = find_blocking_q3(u, v, adj)
                    if q3_edges:
                        blockers.append(set(q3_edges))
                
                if blockers:
                    edges_to_remove = greedy_hitting_set(blockers)
                    
                    if edges_to_remove:
                        for u, v in edges_to_remove:
                            if (adj[u] >> v) & 1:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                current_count -= 1
                        stuck_iterations = 0
            
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
        
        # Local Search Polish
        if time.time() < deadline - 5:
            for _ in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for u, v in all_edges:
                    if not ((adj[u] >> v) & 1):
                        if not has_q3_with_edge(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            current_count += 1
                            improved = True
                            if current_count > best_count:
                                best_count = current_count
                                best_adj = adj[:]
                if not improved:
                    break

    result = []
    for u, v in all_edges:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
