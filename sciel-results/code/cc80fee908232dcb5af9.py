"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed):
        deadline = time.time() + 140.0
        rng = Random(seed)

        # Generate cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))

        # Build adjacency for independent vertices
        indep_neighbors = {v: set() for v in range(k, n)}
        for (u, v) in cross_edges:
            indep_neighbors[v].add(u)

        # All internal edges
        internal_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        
        # Precompute frequency of each internal edge in neighborhoods of independent vertices
        # freq[e] = number of v in I such that e is covered by {v} U N(v)
        freq = {e: 0 for e in internal_edges}
        for v in range(k, n):
            Nv = indep_neighbors[v]
            if len(Nv) < 2:
                continue
            Nv_list = sorted(Nv)
            for i in range(len(Nv_list)):
                for j in range(i + 1, len(Nv_list)):
                    e = (Nv_list[i], Nv_list[j])
                    if e in freq:
                        freq[e] += 1

        best_cliques = None
        best_count = float('inf')
        
        # Multi-start stochastic greedy
        # We try several different orderings and tie-breaking strategies
        max_starts = 20
        # Adjust max_starts based on time
        # Rough estimate: each start takes O(n * k^2) or so.
        # For n=90, k=30, this is fast.
        
        for start_idx in range(max_starts):
            if time.time() > deadline - 5.0:
                break
                
            # Create a shuffled order of independent vertices
            # Weights for shuffling: higher degree vertices earlier
            indep_vertices = [v for v in range(k, n) if indep_neighbors[v]]
            
            # Strategy: Sort by degree descending, then add some random noise to order
            # Or just shuffle and sort by a random key?
            # Let's use a random priority for each vertex
            priorities = {v: random.Random(seed + start_idx).random() for v in indep_vertices}
            
            # Sort by degree descending, tie-break by priority
            indep_vertices.sort(key=lambda v: (-len(indep_neighbors[v]), priorities[v]))
            
            uncovered_internal = set(internal_edges)
            cliques = []
            
            for v in indep_vertices:
                if time.time() > deadline - 2.0:
                    break
                Nv = indep_neighbors[v]
                if not Nv:
                    continue
                
                # We want to select S subset of Nv
                # To maximize value, we prioritize covering internal edges that are:
                # 1. Currently uncovered
                # 2. Have low frequency (rare edges are more "costly" to leave for pairs)
                
                # Greedy construction of S
                S = set()
                remaining_candidates = set(Nv)
                
                # We will iteratively add vertices to S
                # Score of adding u to S:
                # Sum of weights of edges (u, w) for w in S that are uncovered
                # Weight of edge e = 1.0 / (freq[e] + 1)  (higher for rare edges)
                
                while remaining_candidates:
                    if time.time() > deadline - 1.0:
                        break
                    
                    best_u = None
                    best_score = 0.0
                    
                    # Optimization: Only check a subset of candidates if Nv is large?
                    # For k=30, Nv is max 30. 30^2 = 900 ops. Fine.
                    
                    for u in remaining_candidates:
                        score = 0.0
                        for w in S:
                            e = (min(u, w), max(u, w))
                            if e in uncovered_internal:
                                # Reward covering this edge
                                # Rare edges are more valuable
                                f = freq.get(e, 0)
                                score += 1.0 / (f + 1.0)
                        
                        if score > best_score:
                            best_score = score
                            best_u = u
                    
                    if best_u is not None and best_score > 0:
                        S.add(best_u)
                        remaining_candidates.discard(best_u)
                        # Update uncovered_internal? No, we just check membership.
                        # We don't remove from uncovered_internal until we commit to the clique.
                        # But we need to ensure we don't double count in score?
                        # The score calculation is local to S.
                    else:
                        # No more vertices add value
                        break
                
                if len(S) >= 1:
                    # Commit to clique {v} U S
                    clique = sorted([v] + list(S))
                    cliques.append(clique)
                    
                    # Remove covered internal edges from uncovered_internal
                    S_list = sorted(S)
                    for i in range(len(S_list)):
                        for j in range(i + 1, len(S_list)):
                            e = (S_list[i], S_list[j])
                            uncovered_internal.discard(e)
                
                # If S is empty, we don't form a clique for v.
                # Its cross edges will be covered by... wait.
                # If we don't form a clique for v, its cross edges are UNCOVERED.
                # This is INVALID.
                # We MUST cover all cross edges.
                # So if S is empty, we MUST form a clique {v, u} for some u in Nv?
                # But if we form {v, u}, it covers no internal edges.
                # So we should always form at least {v, u} if Nv is not empty.
                
                if len(S) == 0 and Nv:
                    # Pick one neighbor arbitrarily
                    u = next(iter(Nv))
                    cliques.append([v, u])
                    # No internal edges covered.

            # Cover remaining uncovered internal edges with pairs
            for e in list(uncovered_internal):
                if time.time() > deadline - 0.5:
                    break
                u, w = e
                cliques.append([u, w])
            
            # Validate and count
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques

        # Fallback: If no valid solution found (should not happen), return seed-style
        if best_cliques is None:
            best_cliques = []
            if k >= 2:
                best_cliques.append(list(range(k)))
            for (u, v) in cross_edges:
                best_cliques.append([u, v])
        
        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(build_partition(*spec))

    return results
# EVOLVE-BLOCK-END
