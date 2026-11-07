"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    _deadline = time.time() + 140.0  # Safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        # Generate edges using the required instance() function
        edges = instance(n, k, p, seed)
        
        # Separate edges
        k_edges = []
        cross = []
        for u, v in edges:
            if u < k and v < k:
                k_edges.append((u, v))
            else:
                cross.append((u, v))
        
        # Build adjacency for K-vertices
        rem_adj = [0] * k
        for u, v in k_edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        # Helper to find a maximal clique in a candidate set using rem_adj
        def find_maximal_clique(candidates_mask):
            clique = []
            current_mask = candidates_mask
            
            while current_mask:
                # Pick vertex with max degree in current induced subgraph
                best_v = -1
                best_deg = -1
                m = current_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    deg = (current_mask & rem_adj[v]).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    m ^= lsb
                
                if best_v == -1:
                    break
                    
                clique.append(best_v)
                current_mask &= rem_adj[best_v]
                
            return clique

        result = []
        
        # Process cross edges: for each v in I, form cliques {v} U S where S is clique in K
        neighbors_of_v = {v: [] for v in range(k, n)}
        for u, v in cross:
            neighbors_of_v[v].append(u)
        
        for v in range(k, n):
            if not neighbors_of_v[v]:
                continue
            
            candidates_mask = 0
            for u in neighbors_of_v[v]:
                candidates_mask |= (1 << u)
            
            while candidates_mask:
                if time.time() > _deadline:
                    # Fallback: add remaining as pairs
                    for u in neighbors_of_v[v]:
                        if (1 << u) & candidates_mask:
                            result.append([v, u])
                    break
                
                S = find_maximal_clique(candidates_mask)
                if not S:
                    break
                    
                clique_vertices = [v] + S
                result.append(clique_vertices)
                
                # Consume internal edges of S
                for i in range(len(S)):
                    u = S[i]
                    for j in range(i + 1, len(S)):
                        w = S[j]
                        rem_adj[u] &= ~(1 << w)
                        rem_adj[w] &= ~(1 << u)
                
                for u in S:
                    candidates_mask &= ~(1 << u)
        
        # Cover remaining K edges
        while True:
            if time.time() > _deadline:
                # Fallback: add all remaining edges as pairs
                for u in range(k):
                    for v in range(u + 1, k):
                        if (rem_adj[u] >> v) & 1:
                            result.append([u, v])
                            rem_adj[u] &= ~(1 << v)
                            rem_adj[v] &= ~(1 << u)
                break
            
            active_mask = 0
            for u in range(k):
                if rem_adj[u]:
                    active_mask |= (1 << u)
            
            if active_mask == 0:
                break
            
            # Pick an edge (u, v) from active graph
            u = -1
            m = active_mask
            while m:
                lsb = m & -m
                candidate = lsb.bit_length() - 1
                if rem_adj[candidate]:
                    u = candidate
                    break
                m ^= lsb
            
            if u == -1:
                break
                
            v_mask = rem_adj[u] & active_mask
            if not v_mask:
                break
            
            lsb_v = v_mask & -v_mask
            v = lsb_v.bit_length() - 1
            
            common_neighbors = rem_adj[u] & rem_adj[v]
            S = find_maximal_clique(common_neighbors)
            
            clique_vertices = [u, v] + S
            result.append(clique_vertices)
            
            # Remove edges in this clique
            rem_adj[u] &= ~(1 << v)
            rem_adj[v] &= ~(1 << u)
            
            for s in S:
                rem_adj[u] &= ~(1 << s)
                rem_adj[s] &= ~(1 << u)
                rem_adj[v] &= ~(1 << s)
                rem_adj[s] &= ~(1 << v)
                for s2 in S:
                    if s < s2:
                        rem_adj[s] &= ~(1 << s2)
                        rem_adj[s2] &= ~(1 << s)
        
        # Post-processing: merge overlapping cliques to reduce count
        # Two cliques can be merged if their union is also a clique
        # and they share at least one edge (to ensure we don't create duplicates)
        
        # Build edge-to-clique mapping
        edge_to_cliques = {}
        for idx, clique in enumerate(result):
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    edge = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    if edge not in edge_to_cliques:
                        edge_to_cliques[edge] = []
                    edge_to_cliques[edge].append(idx)
        
        # Check if two cliques can be merged
        def can_merge(clique1, clique2):
            # Union must be a clique
            union_set = set(clique1) | set(clique2)
            if len(union_set) < 2:
                return False
            
            # Check all pairs in union are edges in original graph
            # For K-K edges: check rem_adj (but we've consumed some, so use original)
            # Actually, we need to check against the original graph structure
            # Since we don't have original adjacency easily, let's check:
            # All vertices in clique1 are mutually adjacent (they were a clique)
            # All vertices in clique2 are mutually adjacent (they were a clique)
            # We need: every vertex in clique1 is adjacent to every vertex in clique2
            
            # For vertices both in K (index < k): check original K-edges
            # For cross edges: one in K, one in I - they're adjacent iff in cross_edges
            # But we don't have that stored. Let's rebuild a quick check.
            
            # Simpler: just try and verify at the end
            return True
        
        # Try merging: repeatedly look for pairs of cliques that can be merged
        # Use a greedy approach: try to merge clique i with clique j if possible
        merged = True
        while merged and time.time() < _deadline:
            merged = False
            i = 0
            while i < len(result) and time.time() < _deadline:
                j = i + 1
                while j < len(result) and time.time() < _deadline:
                    if can_merge(result[i], result[j]):
                        # Check if union is a valid clique
                        union = list(set(result[i]) | set(result[j]))
                        # Verify all pairs in union are edges
                        valid = True
                        for a in range(len(union)):
                            for b in range(a + 1, len(union)):
                                u, v = union[a], union[b]
                                if u < k and v < k:
                                    # Check K-K edge: was it in original k_edges?
                                    # We consumed them, so this is tricky.
                                    # Let's skip complex verification and just merge
                                    pass
                                else:
                                    # At least one is in I
                                    # Cross edge exists iff in original cross list
                                    pass
                                # For safety, let's just not merge unless we can verify
                                # Actually, since both cliques were valid, and we're taking union,
                                # we need cross-edges between them. This is complex to verify.
                                # Let's skip merging for now to avoid invalid results.
                                valid = False
                                break
                            if not valid:
                                break
                        
                        if valid:
                            result[i] = union
                            result.pop(j)
                            merged = True
                            break
                    j += 1
                i += 1
        
        # Ensure all cliques have at least 2 vertices
        result = [c for c in result if len(c) >= 2]
        
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _deadline - 5:
            break
        results.append(solve_one(*spec))
    
    return results
# EVOLVE-BLOCK-END
