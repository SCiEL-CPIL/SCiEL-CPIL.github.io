import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Direct Independent Set Optimization via Vertex Exchanges.
    Instead of optimizing a permutation (order), we optimize the Set itself.
    1. Start with the greedy set from a low-degree heuristic.
    2. Iteratively attempt to swap a vertex OUT of the set (high degree/cost) 
       with a vertex IN (low degree/benefit) that is not a neighbor of the new set.
    3. This directly targets the objective (set size) without the overhead of 
       permutation simulation.
    4. Final scores are derived from the optimized set: Set members get high scores, 
       non-members get low scores, ordered by "desirability" to maximize the chance 
       that the fixed greedy decoder reconstructs a large independent set.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbors and degrees
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    degrees = np.array([len(nbrs) for nbrs in neighbors_list])
    
    # Helper: Greedy MIS on a subset of allowed vertices, or with a fixed order
    def greedy_mis(order, forbidden_mask=None):
        """
        Returns the set of vertices picked by greedy MIS given the order.
        forbidden_mask: boolean array, True if vertex is already 'taken' (e.g., by a fixed core).
        """
        if forbidden_mask is None:
            active = np.ones(n, dtype=bool)
        else:
            active = ~forbidden_mask.copy()
        
        picked = set()
        for v in order:
            if active[v]:
                picked.add(v)
                active[v] = False
                if neighbors_list[v]:
                    active[neighbors_list[v]] = False
        return picked

    # 1. Initial Heuristic: Low degree first
    # We want to start with a good set.
    init_order = np.argsort(degrees, kind='stable')
    current_set = greedy_mis(init_order)
    
    # Precompute set membership for fast checking
    in_set = np.zeros(n, dtype=bool)
    for v in current_set:
        in_set[v] = True
        
    current_size = len(current_set)
    best_set = set(current_set)
    best_size = current_size
    
    # Identify "candidates" to swap in.
    # Generally, low degree vertices are good candidates for inclusion.
    # High degree vertices in the set are "expensive" and might be swapped out
    # if we can find a low degree vertex that doesn't conflict.
    
    # To make the search efficient, we don't try all pairs.
    # We focus on:
    # 1. Removing a vertex v_out from the set.
    # 2. Adding a vertex v_in not in the set, not a neighbor of v_out, and not a neighbor of other set members?
    #    Actually, if we remove v_out, its neighbors become available.
    #    We can add v_in if v_in is not a neighbor of (current_set - {v_out}).
    
    # Let's use a local search loop.
    # Move: Swap v_out (in set) with v_in (not in set).
    # Condition: v_in is not a neighbor of any vertex in (current_set - {v_out}).
    # If valid, new_set = (current_set - {v_out}) U {v_in}.
    # Size change: 0 (swap) or +1 (if we can add without removing? No, standard swap keeps size same).
    # Wait, we want to INCREASE size.
    # So we look for v_in that can be ADDED without removing anyone?
    # If such v_in exists, we add it. Size +1.
    # If not, we look for swaps that maintain size but improve "potential" for future additions?
    # Or just accept size-preserving swaps if they allow future +1 moves?
    # Simpler: Just look for +1 moves first. If stuck, do size-preserving swaps to escape local optima.
    
    # Let's define a "score" for a vertex to be in the set.
    # Low degree is good.
    # Also, being "isolated" from other high-degree nodes is good?
    
    # Let's implement the search.
    
    rng = np.random.default_rng(42)
    T = 1.0
    cooling = 0.9995
    
    # Lists for fast access
    # in_set_list: list of vertices in current_set
    # out_set_list: list of vertices NOT in current_set
    
    def get_lists():
        in_list = [v for v in range(n) if in_set[v]]
        out_list = [v for v in range(n) if not in_set[v]]
        return in_list, out_list
    
    in_list, out_list = get_lists()
    
    # Precompute neighbor sets for fast intersection checks?
    # neighbors_set = [set(neighbors_list[i]) for i in range(n)]
    # This might be heavy. Let's use boolean masks or just list checks for n=200.
    # n=200 is small. List checks are fine.
    
    max_time = 145.0 # Safety margin
    
    while time.time() - start_time < max_time:
        
        # 1. Try to ADD a vertex (Increase size)
        # Look for v in out_list such that v is not a neighbor of ANY vertex in in_list.
        # This is O(|out| * |in| * avg_deg) roughly, but we can optimize.
        # Actually, if v is not a neighbor of any in_set, it's independent from the whole set.
        
        found_add = False
        # Sample a few candidates from out_list to avoid O(n^2) scan every step
        # If we find one, add it.
        if out_list:
            # Sort out_list by degree? We want low degree first.
            # Let's just pick randomly from low-degree candidates.
            # Or scan all out_list if n is small? 200 is small.
            for v_in in out_list:
                # Check if v_in is neighbor of any in in_list
                is_conflict = False
                nbrs_v = neighbors_list[v_in]
                for u in in_list:
                    if adj[u, v_in]: # Use matrix for O(1) check
                        is_conflict = True
                        break
                if not is_conflict:
                    # Add v_in
                    in_set[v_in] = True
                    current_set.add(v_in)
                    in_list.append(v_in)
                    out_list.remove(v_in)
                    current_size += 1
                    best_set = set(current_set)
                    best_size = current_size
                    found_add = True
                    break # Restart loop to re-evaluate state
                    
        if found_add:
            continue
            
        # 2. Try to SWAP (Maintain size, escape local optima)
        # Pick v_out from in_list, v_in from out_list.
        # Condition: v_in is not a neighbor of (in_list - {v_out}).
        
        # To be efficient, we don't check all pairs.
        # We pick a random v_out.
        # We look for a v_in that is "compatible" with in_list - {v_out}.
        
        if in_list and out_list:
            v_out = in_list[rng.integers(0, len(in_list))]
            
            # Find a compatible v_in
            # v_in must NOT be a neighbor of any u in in_list where u != v_out.
            # Also v_in should be "better" than v_out in some sense?
            # e.g., degree[v_in] < degree[v_out]?
            
            candidates = []
            for v_in in out_list:
                if v_in == v_out: continue
                # Check compatibility
                compatible = True
                for u in in_list:
                    if u == v_out: continue
                    if adj[u, v_in]:
                        compatible = False
                        break
                if compatible:
                    # Heuristic: Prefer lower degree
                    candidates.append(v_in)
            
            if candidates:
                # Pick best candidate (lowest degree)
                v_in = min(candidates, key=lambda x: degrees[x])
                
                # Perform swap
                # Remove v_out
                in_set[v_out] = False
                current_set.remove(v_out)
                in_list.remove(v_out)
                out_list.append(v_out)
                
                # Add v_in
                in_set[v_in] = True
                current_set.add(v_in)
                in_list.append(v_in)
                out_list.remove(v_in)
                
                # Size is unchanged.
                # We accept this swap to explore.
                # To avoid oscillation, we might want to accept only if it leads to a better "potential"?
                # For now, accept always in this branch to escape local minima.
                # But this might degrade the set.
                # Let's add a check: only swap if degree[v_in] < degree[v_out] + small_noise?
                # Or just accept if it allows future additions.
                # Let's accept if degree[v_in] < degree[v_out].
                if degrees[v_in] >= degrees[v_out]:
                    # Revert
                    in_set[v_in] = False
                    current_set.remove(v_in)
                    in_list.remove(v_in)
                    out_list.append(v_in)
                    
                    in_set[v_out] = True
                    current_set.add(v_out)
                    in_list.append(v_out)
                    out_list.remove(v_out)
                else:
                    # Keep swap
                    pass
            
        T *= cooling
        
    # Final Scores
    # The greedy decoder picks vertices in descending score order.
    # We want it to pick our `best_set`.
    # So all vertices in `best_set` should have higher scores than vertices NOT in `best_set`?
    # Not necessarily. If a vertex not in the set has a high score, it might be picked first,
    # blocking a vertex in the set.
    # So:
    # 1. All vertices in best_set get high scores.
    # 2. All vertices NOT in best_set get lower scores.
    # 3. Within best_set, order doesn't matter much for inclusion, but let's order by degree (low first) to be safe?
    #    Actually, if they are all in the set, they are independent. Order among them doesn't affect the fact they are picked,
    #    UNLESS a non-set vertex with a higher score blocks them.
    #    So the critical constraint is: min(score_in_set) > max(score_not_in_set).
    
    scores = np.zeros(n)
    
    # Assign scores
    # Base score for in-set: 1000
    # Base score for out-set: 0
    
    # To be robust, let's rank them.
    # In-set vertices: Rank 1 to |S| (highest scores)
    # Out-set vertices: Rank |S|+1 to n (lower scores)
    
    in_vertices = list(best_set)
    out_vertices = [v for v in range(n) if v not in best_set]
    
    # Sort in_vertices by degree (ascending) - lower degree first?
    # Actually, for the greedy decoder, if we assign high scores to in_set,
    # the decoder will pick them.
    # If we assign low scores to out_set, they will be picked later.
    # If an out_set vertex is picked, it might block in_set vertices?
    # No, because in_set vertices have HIGHER scores, so they are picked BEFORE out_set vertices.
    # So any out_set vertex is considered AFTER all in_set vertices are picked (or blocked).
    # Since in_set is an independent set, none of them block each other.
    # So all in_set vertices WILL be picked.
    # Then, out_set vertices are considered.
    # Some out_set vertices might be picked if they are not blocked by in_set.
    # This is fine. We just want to maximize the count.
    # The count is at least |best_set|.
    # It might be larger if some out_set vertices are also independent from best_set.
    # To maximize this, we should order out_set vertices such that those "compatible" with best_set are picked first?
    # But we don't know which ones are compatible easily without checking.
    # However, the greedy decoder will pick them if they are not blocked.
    # So, to maximize the final set size, we should give higher scores to out_set vertices that are LESS likely to be blocked?
    # Or just give them low scores so they don't interfere?
    # Actually, if we give them low scores, they are picked last.
    # If we give them high scores (but still lower than in_set), they might be picked earlier among out_set.
    # Does order among out_set matter?
    # Yes. If v1 and v2 are both out_set, and v1 blocks v2, and v1 has higher score, v1 is picked, v2 is blocked.
    # If v2 has higher score, v2 is picked, v1 is blocked.
    # We want to pick the one that allows more future picks?
    # This is the original problem.
    # But since in_set is fixed and picked first, the remaining problem is on the subgraph induced by out_set,
    # with some vertices blocked by in_set.
    # Let's just use a heuristic for out_set: Low degree first.
    
    # Sort in_set by degree (doesn't strictly matter for inclusion, but for stability)
    in_vertices.sort(key=lambda x: degrees[x])
    # Sort out_set by degree
    out_vertices.sort(key=lambda x: degrees[x])
    
    # Assign scores
    # In-set: n down to n - |in| + 1
    # Out-set: n - |in| down to 1
    
    for i, v in enumerate(in_vertices):
        scores[v] = n - i
        
    for i, v in enumerate(out_vertices):
        scores[v] = n - len(in_vertices) - i
        
    return scores
