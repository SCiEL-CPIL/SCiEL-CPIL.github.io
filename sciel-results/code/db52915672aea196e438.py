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
        
        def popcount(x):
            """Fast popcount for Python ints."""
            return bin(x).count('1')
        
        def get_remaining_neighbors_mask(v):
            """Get bitmask of remaining neighbors of v."""
            return remaining_mask[v]
        
        def greedy_clique_from_start(start):
            """Greedy build a clique starting from 'start' using only remaining edges.
            Uses bitmask operations for speed."""
            clique = [start]
            # Candidates: neighbors of start via remaining edges
            candidates_mask = remaining_mask[start]
            
            while candidates_mask:
                # Pick candidate with max connections to other candidates
                # Score each candidate by how many of its remaining neighbors are in candidates
                best_cand = None
                best_score = -1
                cm = candidates_mask
                while cm:
                    lsb = cm & (-cm)
                    cand = lsb.bit_length() - 1
                    # Count candidates that are also remaining-neighbors of cand
                    score = popcount(remaining_mask[cand] & candidates_mask)
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                    cm ^= lsb
                
                if best_cand is None:
                    break
                clique.append(best_cand)
                # Update candidates: must connect to best_cand via remaining edges
                candidates_mask &= remaining_mask[best_cand]
            
            return clique if len(clique) >= 2 else None
        
        def local_improve(clique, max_iter=5):
            """Try to extend the clique by checking if any remaining neighbor can be added."""
            clique_set = set(clique)
            clique_mask = 0
            for v in clique:
                clique_mask |= (1 << v)
            
            for _ in range(max_iter):
                # Find a vertex outside clique that connects to all clique members via remaining edges
                # Candidate must be a remaining-neighbor of all clique members
                common_mask = remaining_mask[clique[0]]
                for v in clique[1:]:
                    common_mask &= remaining_mask[v]
                # Remove clique members themselves
                common_mask &= ~clique_mask
                
                if not common_mask:
                    break
                
                # Pick the one with highest remaining degree among candidates
                best_v = None
                best_deg = -1
                cm = common_mask
                while cm:
                    lsb = cm & (-cm)
                    v = lsb.bit_length() - 1
                    deg = popcount(remaining_mask[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cm ^= lsb
                
                if best_v is None:
                    break
                clique.append(best_v)
                clique_set.add(best_v)
                clique_mask |= (1 << best_v)
            
            return clique
        
        def find_best_clique():
            """Try multiple starting vertices, return the largest clique found."""
            best_clique = None
            best_size = 1
            
            # Try vertices in order of remaining degree
            degrees = []
            for v in range(n):
                deg = popcount(remaining_mask[v])
                if deg >= 2:
                    degrees.append((deg, v))
            degrees.sort(reverse=True)
            
            # Try top vertices
            max_trials = min(20, len(degrees))
            for i in range(max_trials):
                deg, v = degrees[i]
                clique = greedy_clique_from_start(v)
                if clique:
                    # Try to improve locally
                    clique = local_improve(clique, 3)
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique
                        if best_size >= 10:  # Good enough for dense graphs
                            break
            
            return best_clique
        
        while remaining_count > 0:
            if time.time() > deadline - 10:
                break
            
            # Find a good clique
            clique = find_best_clique()
            
            if clique is None or len(clique) < 2:
                # Fallback: take a single edge
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
            removed = 0
            for i, u in enumerate(clique):
                for v in clique[i+1:]:
                    if remaining_mask[u] & (1 << v):
                        remaining_mask[u] &= ~(1 << v)
                        remaining_mask[v] &= ~(1 << u)
                        removed += 1
            
            if removed == 0:
                # Shouldn't happen, but fallback
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
