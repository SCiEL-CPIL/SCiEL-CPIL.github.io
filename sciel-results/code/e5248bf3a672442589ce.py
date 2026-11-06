"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_partition(edges, n, time_limit=140):
        if not edges:
            return []
        
        # Build adjacency sets for fast lookup
        # We need to update this as edges are removed?
        # It's better to maintain a set of remaining edges and check connectivity dynamically or update adj.
        # Given n <= 64, updating adj lists is cheap.
        
        start_time = time.time()
        best_cliques = None
        best_count = float('inf')
        
        # Number of attempts
        # We have ~140s. Each attempt is fast.
        # Let's do 200 attempts.
        max_attempts = 200
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Shuffle edges to randomize the order in which they are picked
            # This is crucial for the randomized greedy
            shuffled_edges = edges[:]
            random.shuffle(shuffled_edges)
            
            remaining_edges = set(shuffled_edges)
            cliques = []
            
            # To speed up, we can keep a list of active vertices
            # But checking if an edge is in remaining_edges is O(1) with set.
            
            while remaining_edges:
                # Pick a random edge from the remaining edges
                # To make it more effective, pick an edge that is "central"
                # For simplicity, pick the first one in the shuffled list that is still remaining
                # Or just pick a random one from the set
                # Picking a random one from a set is O(1) if we use list(remaining_edges)[randint] but that's O(N) to convert.
                # Better: maintain a list of remaining edges and remove from it.
                
                # Let's use a list for remaining_edges to allow random selection
                # and a set for O(1) removal?
                # Actually, let's just iterate through the shuffled list.
                # This ensures we don't get stuck in a loop.
                
                # We need to pick an edge that exists in remaining_edges.
                # Since shuffled_edges is a permutation of the original, and we remove from remaining_edges,
                # we can just find the first edge in shuffled_edges that is in remaining_edges.
                # But that's O(N) per step.
                
                # Better approach:
                # Keep a list `active_edges` which is a subset of shuffled_edges.
                # When we remove a clique, we filter `active_edges`.
                
                pass # Placeholder, logic below

        # Re-implementing the loop properly with list filtering
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Use a different random seed for each attempt to vary the shuffle
            # The global random state is shared, which is fine.
            shuffled_edges = edges[:]
            random.shuffle(shuffled_edges)
            
            active_edges = shuffled_edges[:]
            cliques = []
            
            while active_edges:
                # Pick a random edge from active_edges
                # To bias towards "hard" edges, we could pick randomly.
                # Let's pick the first one for determinism within the shuffle, or random.
                # Random is better for exploration.
                idx = random.randint(0, len(active_edges) - 1)
                e = active_edges[idx]
                u, v = e
                
                # Build a clique starting with {u, v}
                clique = {u, v}
                # Candidates are vertices connected to all in clique
                # Initially, connected to u and v
                candidates = set()
                for w in range(n):
                    if w != u and w != v:
                        if (min(u,w), max(u,w)) in {(min(u,v), max(u,v))} or True: # Check edge u-w
                            # We need to check if edge (u,w) and (v,w) exist in the ORIGINAL graph?
                            # No, they must exist in the REMAINING graph?
                            # The clique must be a clique in the original graph.
                            # The edges of the clique must be in the remaining_edges set.
                            # So, to add w, edges (w, x) for all x in clique must be in remaining_edges.
                            
                            # Let's check if (u,w) and (v,w) are in remaining_edges
                            # We need a fast way to check edge existence in remaining_edges.
                            # remaining_edges is a set of tuples.
                            
                            pass

        # This approach is getting messy with state management.
        # Let's use a cleaner implementation.
        
        def greedy_partition(edges, n):
            # edges is a list of (u, v)
            # We maintain a set of remaining edges
            remaining = set(edges)
            # We also maintain adjacency lists for the ORIGINAL graph to check connectivity quickly?
            # No, we only care about edges in 'remaining'.
            # But to check if (u,w) is in remaining, we need a set.
            
            # To make removal efficient, we can just iterate.
            # For n=64, max edges ~ 2000.
            # A set of tuples is fine.
            
            cliques = []
            
            # We need to pick edges.
            # Let's shuffle the list of edges to pick from.
            # But we must ensure we only pick edges that are still remaining.
            
            # Strategy:
            # 1. Shuffle all edges.
            # 2. Iterate through the shuffled list.
            # 3. If the edge is in 'remaining', use it as a seed for a clique.
            # 4. Expand the clique.
            # 5. Remove all edges of the clique from 'remaining'.
            # 6. Continue.
            
            # This is O(E * K) where K is avg clique size.
            
            shuffled = edges[:]
            random.shuffle(shuffled)
            
            # We need to check if an edge is in remaining.
            # 'remaining' is a set.
            
            # Helper to check if edge (a,b) is in remaining
            def has_edge(a, b):
                if a > b: a, b = b, a
                return (a, b) in remaining

            # Helper to remove edge
            def remove_edge(a, b):
                if a > b: a, b = b, a
                remaining.discard((a, b))

            while remaining:
                # Find the first edge in shuffled that is still in remaining
                seed_edge = None
                for e in shuffled:
                    if e in remaining:
                        seed_edge = e
                        break
                
                if seed_edge is None:
                    break # Should not happen if remaining is not empty and shuffled covers all
                
                u, v = seed_edge
                clique = {u, v}
                
                # Expand clique
                # Candidates: vertices w such that (w, x) in remaining for all x in clique
                # Start with candidates = neighbors of u and v in the remaining graph
                # But we don't have an adjacency list for the remaining graph.
                # We can just iterate all vertices w in 0..n-1
                # Check if w is connected to all in clique via remaining edges.
                
                # To speed up, we can maintain a list of active candidates.
                # Initially, w must be connected to u and v.
                candidates = []
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if has_edge(u, w) and has_edge(v, w):
                        candidates.append(w)
                        
                while candidates:
                    # Pick a candidate w that maximizes something
                    # e.g., number of connections to other candidates?
                    # Or just pick the first one?
                    # Let's pick the one with the highest degree in the remaining graph (approx).
                    # Or just random to add diversity?
                    # For greedy, picking the one that connects to the most other candidates is good.
                    
                    # Score w by number of edges to other candidates
                    best_w = -1
                    best_score = -1
                    for w in candidates:
                        score = 0
                        for v2 in candidates:
                            if w == v2: continue
                            if has_edge(w, v2):
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_w = w
                            
                    if best_w == -1:
                        break
                    
                    clique.add(best_w)
                    
                    # Update candidates: remove best_w, and remove any w that is not connected to best_w
                    new_candidates = []
                    for w in candidates:
                        if w == best_w:
                            continue
                        if has_edge(w, best_w):
                            new_candidates.append(w)
                    candidates = new_candidates
                    
                # Add clique to result
                cliques.append(list(clique))
                
                # Remove all edges of the clique from remaining
                clique_list = list(clique)
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        remove_edge(clique_list[i], clique_list[j])
                        
            return cliques

        # Run multiple attempts
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            partition = greedy_partition(edges, n)
            if len(partition) < best_count:
                best_count = len(partition)
                best_cliques = partition
                
        return best_cliques

    instances = [(32, 0.6, 601), (64, 0.55, 602)]
    results = []
    for n, p, seed in instances:
        edges = random_graph(n, p, seed)
        partition = solve_partition(edges, n)
        results.append(partition)
    return results

# EVOLVE-BLOCK-END
