import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        
        # List of all edges as (u, v) with u < v
        all_edges = []
        for u in range(n):
            m = adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    all_edges.append((u, v))
                m ^= lsb
        
        if not all_edges:
            return []

        # Map edge to index for bitmask representation
        edge_to_idx = {}
        for i, (u, v) in enumerate(all_edges):
            edge_to_idx[(u, v)] = i
            
        num_edges = len(all_edges)

        # Generate candidate cliques
        # 1. Greedy expansion from each vertex
        candidates = []
        degrees = [adj[i].bit_count() for i in range(n)]
        vertices_sorted = sorted(range(n), key=lambda x: -degrees[x])
        
        for start_v in vertices_sorted:
            clique = [start_v]
            cand_mask = adj[start_v]
            
            while cand_mask:
                # Pick vertex in cand_mask with max degree in cand_mask
                best_v = -1
                best_deg = -1
                m = cand_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    deg_in_cand = (adj[v] & cand_mask).bit_count()
                    if deg_in_cand > best_deg:
                        best_deg = deg_in_cand
                        best_v = v
                    m ^= lsb
                
                if best_v == -1:
                    break
                    
                clique.append(best_v)
                cand_mask &= adj[best_v]
                
            if len(clique) >= 2:
                candidates.append(clique)
        
        # 2. Add all pairs as candidates
        for u, v in all_edges:
            candidates.append([u, v])

        # Precompute bitmasks for candidates
        candidate_masks = []
        for clique in candidates:
            mask = 0
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i+1, len(clique)):
                    v = clique[j]
                    if u > v:
                        u, v = v, u
                    e_idx = edge_to_idx.get((u, v))
                    if e_idx is not None:
                        mask |= (1 << e_idx)
            if mask:
                candidate_masks.append(mask)
            else:
                # Should not happen if clique is valid
                candidate_masks.append(0)

        # Greedy Set Cover
        uncovered = (1 << num_edges) - 1
        selected_cliques = []
        active_indices = list(range(len(candidate_masks)))
        
        # Optimization: If a candidate mask is 0, remove it
        active_indices = [i for i in active_indices if candidate_masks[i] != 0]
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_idx = -1
            best_covered = 0
            
            # Find candidate that covers most uncovered edges
            for i in active_indices:
                mask = candidate_masks[i]
                # If this candidate covers no uncovered edges, skip
                if (mask & uncovered) == 0:
                    continue
                covered = (mask & uncovered).bit_count()
                if covered > best_covered:
                    best_covered = covered
                    best_idx = i
            
            if best_idx == -1 or best_covered == 0:
                # Fallback: pick any edge
                # This should not happen if we have all pairs
                break
            
            selected_cliques.append(candidates[best_idx])
            uncovered &= ~candidate_masks[best_idx]
            
            # Remove candidates that no longer cover any uncovered edges to speed up future iterations
            # This is O(K) per iteration, which is fine
            new_active = []
            for i in active_indices:
                if (candidate_masks[i] & uncovered) != 0:
                    new_active.append(i)
            active_indices = new_active

        return selected_cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
