"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 145  # 145s to leave safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_instance(n, p, seed, time_budget):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency as bitmasks for fast intersection
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        remaining_mask = [0] * n  # bitmask of remaining edges per vertex
        for u, v in edges:
            remaining_mask[u] |= (1 << v)
            remaining_mask[v] |= (1 << u)
        
        # Count remaining edges
        remaining_count = len(edges)
        
        cliques = []
        
        def get_remaining_neighbors(v):
            """Get set of vertices u such that (min(v,u), max(v,u)) is in remaining."""
            mask = remaining_mask[v]
            result = []
            while mask:
                lsb = mask & (-mask)
                u = lsb.bit_length() - 1
                result.append(u)
                mask ^= lsb
            return result
        
        def greedy_clique_from_start(start, max_time):
            """Greedy build a clique starting from 'start' using only remaining edges."""
            t0 = time.time()
            clique = [start]
            # Candidates: neighbors of start via remaining edges
            candidates = set(get_remaining_neighbors(start))
            
            while candidates:
                if time.time() - t0 > max_time:
                    break
                # Pick candidate that maximizes connections to other candidates via remaining edges
                best_cand = None
                best_score = -1
                for cand in candidates:
                    # Count how many other candidates connect to cand via remaining edges
                    score = 0
                    cand_neigh = set(get_remaining_neighbors(cand))
                    for other in candidates:
                        if other != cand and other in cand_neigh:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                if best_cand is None:
                    break
                clique.append(best_cand)
                candidates.discard(best_cand)
                # Update candidates: must connect to best_cand via remaining edges
                best_cand_neigh = set(get_remaining_neighbors(best_cand))
                candidates = candidates & best_cand_neigh
            
            return clique if len(clique) >= 2 else None
        
        def find_best_clique(time_budget_per_trial):
            """Try multiple starting vertices, return the largest clique found."""
            t_start = time.time()
            best_clique = None
            best_size = 1
            
            # Try vertices in order of remaining degree
            degrees = []
            for v in range(n):
                deg = bin(remaining_mask[v]).count('1')
                degrees.append((deg, v))
            degrees.sort(reverse=True)
            
            for deg, v in degrees:
                if time.time() - t_start > time_budget_per_trial:
                    break
                if deg < 2:
                    break
                clique = greedy_clique_from_start(v, 0.05)
                if clique and len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
                    if best_size >= 8:  # Good enough for dense graphs
                        break
            
            return best_clique
        
        while remaining_count > 0:
            if time.time() > deadline - 10:
                break
            
            # Find a good clique
            clique = find_best_clique(2.0)
            
            if clique is None or len(clique) < 2:
                # Fallback: take a single edge
                # Find any remaining edge
                found = False
                for u in range(n):
                    mask = remaining_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            remaining_mask[u] &= ~(1 << v)
                            remaining_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            found = True
                            break
                        mask ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue
            
            # Remove all edges within clique from remaining
            clique_set = set(clique)
            removed = 0
            for i, u in enumerate(clique):
                for v in clique[i+1:]:
                    if remaining_mask[u] & (1 << v):
                        remaining_mask[u] &= ~(1 << v)
                        remaining_mask[v] &= ~(1 << u)
                        removed += 1
            
            if removed == 0:
                # Shouldn't happen, but fallback
                e = [u, v]
                for u in range(n):
                    mask = remaining_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            remaining_mask[u] &= ~(1 << v)
                            remaining_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            break
                        mask ^= lsb
                    if remaining_count < len(edges):
                        break
                continue
            
            cliques.append(clique)
            remaining_count -= removed
        
        # Handle any remaining edges as individual edge-cliques
        for u in range(n):
            mask = remaining_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    remaining_mask[u] &= ~(1 << v)
                    remaining_mask[v] &= ~(1 << u)
                mask ^= lsb
        
        return cliques
    
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed, deadline - time.time() - 5))
    
    return result
# EVOLVE-BLOCK-END
