"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute adjacency lists for fast access in validity checks
    # We will maintain a dynamic graph state
    
    def get_degrees_and_adj(edges):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj

    def count_violations(edges):
        """Returns the number of violating paths s-u-v-t where deg(s) == deg(t)."""
        if not edges:
            return 0
        deg, adj = get_degrees_and_adj(edges)
        violations = 0
        for u in range(n):
            if not adj[u]:
                continue
            # For each neighbor v of u, check paths s-u-v-t
            # s is in adj[u] \ {v}
            # t is in adj[v] \ {u}
            # We need deg(s) != deg(t)
            
            # Optimization: Group neighbors of u by degree
            deg_counts_u = {}
            for s in adj[u]:
                if s not in adj[v] or True: # s is neighbor of u
                    pass # We need to be careful with the loop structure
            
            # Let's iterate over edges (u, v) directly
            for v in adj[u]:
                # S = {deg(s) | s in adj[u], s != v}
                # T = {deg(t) | t in adj[v], t != u}
                
                # Count degrees in S
                count_s = {}
                for s in adj[u]:
                    if s != v:
                        d = deg[s]
                        count_s[d] = count_s.get(d, 0) + 1
                
                if not count_s:
                    continue
                
                # Check intersections with T
                for t in adj[v]:
                    if t != u:
                        dt = deg[t]
                        if dt in count_s:
                            violations += count_s[dt]
        return violations

    def is_valid(edges):
        return count_violations(edges) == 0

    # Strategy: Constructive Heuristic with Random Restarts and Backtracking
    # 1. Pick a random target degree sequence that is likely valid.
    #    Validity depends on degrees. If we ensure that for any edge (u,v),
    #    the set of degrees of N(u)\{v} and N(v)\{u} are disjoint, we are safe.
    #    
    #    A sufficient condition for validity:
    #    If the graph is a "blow-up" of a small core, or if degrees are very distinct.
    #    
    #    Let's try to build a graph with a specific structure:
    #    Partition vertices into groups G1, G2, ... Gk.
    #    Connect groups in a way that degrees are controlled.
    #    
    #    Alternative: Greedy construction with "Lookahead".
    #    1. Start with empty graph.
    #    2. At each step, pick an edge that maximizes "potential" (e.g., allows many more edges) 
    #       while maintaining validity.
    #    3. If stuck, backtrack.
    
    # To make this fast, we need a fast "can we add edge e?" check.
    # Adding edge (u,v) changes deg(u) and deg(v).
    # This changes the sets S and T for all paths involving u or v.
    # This is expensive to check dynamically.
    
    # Let's use a different structural insight:
    # If we ensure that all vertices in a connected component have distinct degrees, 
    # is it valid?
    # Path s-u-v-t. deg(s) != deg(t).
    # If all degrees in the component are distinct, then deg(s) != deg(t) for s != t.
    # So, any graph where all vertices have distinct degrees is VALID.
    # 
    # Can we have a graph on 15 vertices with all distinct degrees?
    # Max degree is 14. Min is 0.
    # Degrees must be a permutation of a subset of {0..14}.
    # If we use degrees 1..15? No, max is 14.
    # We can have at most 15 distinct degrees.
    # A graph with degrees 0, 1, ..., 14 is impossible (sum of degrees must be even, 0+...+14 = 105, odd).
    # So we can't have all 15 distinct.
    # We can have 14 distinct degrees. e.g., 0, 1, ..., 13 and one duplicate.
    # Or 1, 2, ..., 14 and one duplicate.
    # 
    # If we can construct a graph with 14 distinct degrees and 1 duplicate, 
    # and ensure the duplicate vertices are not endpoints of a 3-path with equal degree neighbors...
    # Actually, if degrees are distinct, it's automatically valid.
    # The only risk is when deg(s) == deg(t). This can ONLY happen if s and t have the same degree.
    # If we ensure that for any two vertices x, y with deg(x) == deg(y), 
    # they are NOT connected by a path of length 3 (s-u-v-t where s=x, t=y or vice versa), 
    # then the graph is valid.
    
    # So the strategy is:
    # 1. Generate a degree sequence with many distinct values.
    # 2. Realize the graph.
    # 3. Ensure that vertices with equal degrees are "far apart" (distance > 3).
    
    # Let's try to maximize edges while keeping degrees distinct as much as possible.
    # A "semi-complete" graph has high edge count.
    
    # Let's try a specific construction:
    # Split vertices into two sets A and B.
    # Make A a clique, B a clique, and connect A and B sparsely?
    # No, cliques have many vertices with same degree (within the clique).
    
    # Let's try:
    # 1. Randomly assign target degrees d_i.
    # 2. Use Havel-Hakimi to build a graph.
    # 3. Check validity.
    # 4. If invalid, perturb the degree sequence or the realization.
    
    def havel_hakimi(deg_seq):
        """Returns a list of edges for the given degree sequence, or None if not graphical."""
        # Sort vertices by degree
        # This is a simplified Havel-Hakimi that returns a specific realization
        # We need to map back to original vertex IDs
        
        # Create list of (degree, original_id)
        items = sorted([(d, i) for i, d in enumerate(deg_seq)], reverse=True)
        
        edges = []
        # We will greedily connect the highest degree vertex to the next highest
        # This is a heuristic realization, not guaranteed to find a solution if one exists,
        # but works well for dense graphs.
        
        # Actually, standard HH reduces the sequence. We need to track which vertices are connected.
        # Let's use a simpler greedy:
        # Sort vertices by degree descending.
        # For each vertex u, connect to the 'deg[u]' available vertices with highest remaining degree.
        
        # This might fail. Let's use a robust random greedy realization.
        
        # Re-initialize
        remaining_deg = list(deg_seq)
        adj = [set() for _ in range(n)]
        
        # Sort indices by degree descending
        order = sorted(range(n), key=lambda x: remaining_deg[x], reverse=True)
        
        # We need to be careful: when we connect u to v, we reduce deg of both.
        # The "highest remaining degree" strategy changes dynamically.
        
        # Let's just try to build it.
        for u in order:
            if remaining_deg[u] <= 0:
                continue
            # Find 'remaining_deg[u]' candidates v
            # Candidates are v > u (to avoid double counting) or v not yet connected to u
            # And v must have remaining_deg[v] > 0
            # And v != u
            
            # Sort candidates by remaining degree
            candidates = [v for v in range(n) if v != u and v not in adj[u] and remaining_deg[v] > 0]
            candidates.sort(key=lambda v: remaining_deg[v], reverse=True)
            
            if len(candidates) < remaining_deg[u]:
                return None # Not graphical with this greedy
            
            # Connect to top candidates
            for v in candidates[:remaining_deg[u]]:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                remaining_deg[u] -= 1
                remaining_deg[v] -= 1
                if remaining_deg[u] == 0:
                    break
        
        # Check if all degrees are satisfied
        for i in range(n):
            if remaining_deg[i] != 0:
                return None
        
        return edges

    best_edges = []
    best_score = 0
    
    num_restarts = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate a random degree sequence
        # Bias towards high degrees to maximize edges
        # Sum of degrees must be even.
        
        # Strategy: Start with a high base degree and add noise
        base_deg = random.randint(8, 12)
        deg_seq = [base_deg] * n
        
        # Add random variation to create distinct degrees
        for _ in range(random.randint(5, 15)):
            i = random.randint(0, n - 1)
            delta = random.randint(-2, 2)
            new_deg = deg_seq[i] + delta
            if 0 <= new_deg <= n - 1:
                deg_seq[i] = new_deg
        
        # Adjust to make sum even
        total_deg = sum(deg_seq)
        if total_deg % 2 != 0:
            # Flip one vertex
            i = random.randint(0, n - 1)
            if deg_seq[i] > 0:
                deg_seq[i] -= 1
            else:
                deg_seq[i] += 1
        
        # Try to realize
        edges = havel_hakimi(deg_seq)
        
        if edges:
            # Check validity
            if is_valid(edges):
                if len(edges) > best_score:
                    best_edges = edges
                    best_score = len(edges)
            
            # Local Search: Try to add edges
            # Convert to set for fast lookup
            edge_set = set(edges)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try to add a random missing edge
                missing = [e for e in all_edges if e not in edge_set]
                if not missing:
                    break
                
                # Shuffle missing to randomize
                random.shuffle(missing)
                
                for e in missing[:20]: # Try a few
                    new_edges = edges + [e]
                    if is_valid(new_edges):
                        edges = new_edges
                        edge_set.add(e)
                        best_score = max(best_score, len(edges))
                        if len(edges) > len(best_edges):
                            best_edges = edges[:]
                        improved = True
                        break # Restart the inner loop with new state? Or continue?
                        # Let's break to re-evaluate all missing edges with new state
                        # Actually, just break the for loop, the while loop will handle it
        
    # Final Greedy Polish
    if best_edges:
        best_set = set(best_edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in all_edges:
                if e in best_set:
                    continue
                new_edges = best_edges + [e]
                if is_valid(new_edges):
                    best_edges = new_edges
                    best_set.add(e)
                    improved = True
                    break # Restart to check all edges again
    
    # Ensure types are correct
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
