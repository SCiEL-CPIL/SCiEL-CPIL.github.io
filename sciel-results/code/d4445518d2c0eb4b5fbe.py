"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict
import sys
import math

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def get_maximal_cliques(edges_set, n_vertices, k):
        """
        Generate candidate maximal cliques in a split graph.
        1. Cliques entirely within K (0..k-1).
        2. Cliques involving I vertices (k..n-1).
        For an I vertex v, its neighbors in K form a clique (since K is a clique).
        So any clique containing v is a subset of {v} U N_K(v).
        The maximal cliques involving v are {v} U N_K(v) itself? 
        Not necessarily, because N_K(v) might not be a clique? 
        Wait, K IS a clique. So N_K(v) is a subset of K, hence a clique.
        So {v} U N_K(v) is a clique.
        Is it maximal? 
        A clique C is maximal if no vertex can be added.
        If we have {v} U N_K(v), can we add another I vertex w? 
        Only if w is connected to v? No, I is an independent set. So w is not connected to v.
        So we cannot add another I vertex.
        Can we add a K vertex u? u is in K. If u is connected to v, u is in N_K(v). 
        So {v} U N_K(v) is a maximal clique containing v.
        
        What about cliques entirely in K?
        The whole K is a clique. Is it maximal? 
        If any I vertex is connected to all of K, then K U {v} is a larger clique.
        Otherwise, K is maximal? 
        Actually, if K is not contained in any {v} U N_K(v) for any v, then K is maximal.
        But K is always a clique. 
        The set of all maximal cliques in a split graph is:
        - For each v in I: C_v = {v} U N_K(v)
        - K itself, IF K is not a subset of any C_v.
        Actually, K is a subset of C_v iff N_K(v) = K.
        If there exists v with N_K(v) = K, then C_v = K U {v} is a clique, so K is not maximal.
        If for all v, N_K(v) != K, then K is a maximal clique.
        
        So the list of maximal cliques is:
        [ {v} U N_K(v) for v in I ] 
        plus [ K ] if max(|N_K(v)| for v in I) < k.
        
        However, we don't necessarily need to cover edges using ONLY maximal cliques.
        But any clique is contained in a maximal clique.
        If we use a non-maximal clique C, we can replace it with a maximal clique C' containing it.
        C' covers all edges C covers, plus maybe more.
        Using C' instead of C might allow us to remove other cliques.
        So there exists an optimal solution using only maximal cliques.
        
        Let's generate these maximal cliques.
        """
        neighbors_in_K = defaultdict(set)
        for u, v in edges_set:
            if u < k and v >= k:
                neighbors_in_K[v].add(u)
            elif u >= k and v < k:
                neighbors_in_K[u].add(v)
        
        max_cliques = []
        # 1. Cliques involving I vertices
        for v in range(k, n_vertices):
            if neighbors_in_K[v]:
                clique = list(neighbors_in_K[v]) + [v]
                max_cliques.append(clique)
        
        # 2. The clique K itself
        # Check if K is contained in any C_v
        is_contained = False
        for v in range(k, n_vertices):
            if len(neighbors_in_K[v]) == k:
                is_contained = True
                break
        
        if not is_contained:
            max_cliques.append(list(range(k)))
            
        return max_cliques

    def greedy_set_cover_with_local_search(edges_set, n_vertices, k, seed):
        """
        1. Generate maximal cliques.
        2. Greedy Set Cover: repeatedly pick clique covering most uncovered edges.
        3. Local Search: try to swap cliques to improve.
        """
        if not edges_set:
            return []
            
        max_cliques = get_maximal_cliques(edges_set, n_vertices, k)
        
        # Map edge to set of clique indices containing it
        edge_to_cliques = defaultdict(list)
        clique_edges = []
        for i, clique in enumerate(max_cliques):
            c_edges = set()
            for u in clique:
                for v in clique:
                    if u < v:
                        e = (u, v)
                        if e in edges_set:
                            c_edges.add(e)
            clique_edges.append(c_edges)
            for e in c_edges:
                edge_to_cliques[e].append(i)
        
        # Pre-compute edge counts for each clique
        # Note: Some cliques might be empty? No, maximal cliques have at least 2 vertices if they cover edges.
        # But if a clique has no edges in edges_set, it's useless.
        # Let's filter out cliques that don't cover any edges.
        valid_cliques = []
        valid_clique_edges = []
        valid_idx_map = {} # old idx -> new idx
        for i, c_edges in enumerate(clique_edges):
            if c_edges:
                new_idx = len(valid_cliques)
                valid_idx_map[i] = new_idx
                valid_cliques.append(max_cliques[i])
                valid_clique_edges.append(c_edges)
        
        if not valid_cliques:
            return [] # Should not happen if edges_set is not empty
            
        # Re-map edge_to_cliques to new indices
        new_edge_to_cliques = defaultdict(list)
        for e, old_idxs in edge_to_cliques.items():
            for old_idx in old_idxs:
                if old_idx in valid_idx_map:
                    new_edge_to_cliques[e].append(valid_idx_map[old_idx])
                    
        uncovered = set(edges_set)
        selected = []
        selected_clique_indices = set()
        
        # Greedy Phase
        # To speed up, we can maintain a count of how many uncovered edges each clique covers.
        # cover_count[c] = number of edges in clique c that are currently uncovered.
        cover_count = [0] * len(valid_cliques)
        for e in uncovered:
            for c in new_edge_to_cliques[e]:
                cover_count[c] += 1
                
        while uncovered:
            # Find clique with max cover_count
            best_c = -1
            best_count = -1
            for c in range(len(valid_cliques)):
                if cover_count[c] > best_count:
                    best_count = cover_count[c]
                    best_c = c
            
            if best_c == -1:
                break # Should not happen
                
            selected.append(best_c)
            selected_clique_indices.add(best_c)
            
            # Update uncovered and cover_counts
            c_edges = valid_clique_edges[best_c]
            for e in c_edges:
                if e in uncovered:
                    uncovered.remove(e)
                    # Decrement cover_count for all cliques containing e
                    for c in new_edge_to_cliques[e]:
                        if c != best_c: # best_c is already handled
                            cover_count[c] -= 1
            cover_count[best_c] = 0 # It's selected, so we don't care about its count anymore (or set to -inf)
            
        # Local Search Phase
        # Objective: Minimize len(selected)
        # Move: Replace a selected clique with a non-selected one, if it covers all edges of the removed one 
        # and possibly allows removing others? 
        # Simpler move: Try to remove a selected clique if its edges are covered by other selected cliques.
        # This is "redundancy removal".
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Build a set of edges covered by each selected clique
            # And a count of how many selected cliques cover each edge
            edge_cover_count = defaultdict(int)
            for c in selected:
                for e in valid_clique_edges[c]:
                    edge_cover_count[e] += 1
            
            # Try to remove each selected clique
            for i, c in enumerate(selected):
                # Check if all edges in c are covered by at least one other selected clique
                removable = True
                for e in valid_clique_edges[c]:
                    if edge_cover_count[e] <= 1:
                        removable = False
                        break
                
                if removable:
                    # Remove c
                    selected.pop(i)
                    # Update edge_cover_count
                    for e in valid_clique_edges[c]:
                        edge_cover_count[e] -= 1
                    improved = True
                    break # Restart loop since list changed
                    
        # Convert selected indices to clique vertex lists
        result = []
        for c in selected:
            result.append(valid_cliques[c])
            
        return result

    def verify_partition(cliques, edges_set, n):
        """
        Verify that cliques form a valid partition.
        1. Each clique is a clique in the graph.
        2. Every edge is in exactly one clique.
        3. No edge is in multiple cliques.
        """
        covered_edges = set()
        for clique in cliques:
            # Check clique validity
            for u in clique:
                for v in clique:
                    if u < v:
                        e = (u, v)
                        if e not in edges_set:
                            return False # Not a clique
                        if e in covered_edges:
                            return False # Overlap
                        covered_edges.add(e)
        
        if covered_edges != edges_set:
            return False # Missing edges
            
        return True

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edges_set = set(edges)
        
        # Try multiple seeds for the greedy/local search to find a good solution
        best_cliques = None
        best_len = float('inf')
        
        # The greedy part is deterministic given the order, but we can randomize the tie-breaking
        # or just run a few times with different random seeds if we introduce randomness.
        # Currently get_maximal_cliques is deterministic.
        # Greedy selection is deterministic if we just pick max.
        # Let's add some randomness to the greedy step to explore different solutions.
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Shuffle max cliques? No, that changes the structure.
            # Instead, in greedy step, if there's a tie, pick randomly.
            # But my current greedy implementation doesn't have random tie-breaking.
            # Let's modify greedy to accept a seed and use it for tie-breaking.
            
            # For now, just run the standard one.
            # To get diversity, we can perturb the graph slightly? No, must be exact.
            # We can try different starting points for local search?
            
            cliques = greedy_set_cover_with_local_search(edges_set, n, k, seed + attempt)
            
            if verify_partition(cliques, edges_set, n):
                if len(cliques) < best_len:
                    best_len = len(cliques)
                    best_cliques = cliques
            else:
                # Fallback: ensure validity
                # This should not happen if the logic is correct, but just in case
                # Construct a valid partition: each edge as a clique
                fallback = [[e[0], e[1]] for e in edges_set]
                if best_cliques is None or len(fallback) < best_len:
                    best_cliques = fallback
                    best_len = len(fallback)
                    
        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
