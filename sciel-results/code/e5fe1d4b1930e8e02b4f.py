"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks for fast intersection
        rem_mask = [0] * n
        for u, v in edges:
            rem_mask[u] |= (1 << v)
            rem_mask[v] |= (1 << u)

        remaining_count = len(edges)
        cliques = []

        # Helper to get list of neighbors from mask
        def mask_to_list(mask):
            res = []
            while mask:
                lsb = mask & (-mask)
                res.append(lsb.bit_length() - 1)
                mask ^= lsb
            return res

        def find_large_clique():
            """
            Finds a large clique using a fast greedy heuristic.
            Returns a list of vertices forming a clique.
            """
            if not any(rem_mask):
                return []
            
            best_clique = []
            best_size = 0
            
            # Try multiple starting vertices (those with high degree)
            vertices = [v for v in range(n) if rem_mask[v]]
            vertices.sort(key=lambda v: bin(rem_mask[v]).count('1'), reverse=True)
            
            # Try top 5 starting vertices
            for start_v in vertices[:5]:
                # Greedy clique construction
                clique = [start_v]
                candidates = [u for u in vertices if u != start_v and (rem_mask[start_v] >> u) & 1]
                
                # Repeatedly add the vertex with highest degree among candidates
                while candidates:
                    # Sort candidates by degree within current candidate set
                    candidates.sort(key=lambda v: sum(1 for u in candidates if u != v and (rem_mask[v] >> u) & 1), reverse=True)
                    
                    # Pick the best candidate
                    v = candidates[0]
                    clique.append(v)
                    
                    # Filter candidates to only those adjacent to v
                    candidates = [u for u in candidates if u != v and (rem_mask[v] >> u) & 1]
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            # Fallback
            if not best_clique and vertices:
                best_clique = [vertices[0]]
                for u in vertices:
                    if u != vertices[0] and (rem_mask[vertices[0]] >> u) & 1:
                        best_clique.append(u)
                        break

            return best_clique

        while remaining_count > 0:
            if time.time() > deadline - 10:
                break

            clique = find_large_clique()

            if not clique or len(clique) < 2:
                # Fallback: take a single edge
                found = False
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
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
                for v in clique[i + 1:]:
                    if (rem_mask[u] >> v) & 1:
                        rem_mask[u] &= ~(1 << v)
                        rem_mask[v] &= ~(1 << u)
                        removed += 1

            if removed == 0:
                # Shouldn't happen, but fallback
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
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
            mask = rem_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    rem_mask[u] &= ~(1 << v)
                    rem_mask[v] &= ~(1 << u)
                mask ^= lsb

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
# EVOLVE-BLOCK-END
