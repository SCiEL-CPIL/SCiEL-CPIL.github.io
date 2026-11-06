"""Construct a dense K2,3-free graph on 20 vertices using a deterministic greedy degree-balancing approach."""

import time
import random

def construct():
    """Return a list of edges [u, v] of a simple graph on vertices 0..19 that is K2,3-free.
    Maximize the number of edges."""
    
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    n = 20
    total_possible_edges = n * (n - 1) // 2  # 190
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # Try multiple randomized orders and take the best
    # Also try a deterministic balanced degree approach
    
    def greedy_build(order):
        """Build a K2,3-free graph by adding edges in the given order if valid."""
        # For each vertex, track its neighbors as a set
        neighbors = [set() for _ in range(n)]
        edges = []
        
        for (u, v) in order:
            if time.time() > DEADLINE:
                break
            
            # Check if adding (u, v) creates a K2,3
            # A K2,3 exists if there are 3 vertices w such that w is neighbor of both u and v
            # After adding (u, v), u and v become neighbors of each other
            # The common neighbors of u and v would be: 
            #   (neighbors[u] & neighbors[v]) -- but wait, u is now in neighbors[v] and v is in neighbors[u]
            #   Actually we need to check BEFORE adding: if u and v share >= 3 common neighbors,
            #   then adding edge (u,v) creates K2,3 with those 3 common neighbors.
            
            common = neighbors[u] & neighbors[v]
            # If common has >= 3 elements, adding (u,v) would create a K2,3
            if len(common) >= 3:
                continue
            
            # Safe to add
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append((u, v))
        
        return edges
    
    # Strategy 1: Deterministic degree-balanced greedy
    # At each step, pick the edge that adds the most "value" while keeping degrees balanced
    # We iterate over all edges multiple times, each time adding the best valid edge
    
    def degree_balanced_greedy():
        neighbors = [set() for _ in range(n)]
        degrees = [0] * n
        edges = []
        # Track which edges have been added
        added = set()
        
        # Pre-compute all edges
        all_e = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # We'll do passes, each time trying to add the best edge
        # "Best" = edge that maximizes degree balance (minimizes max degree difference)
        # and is valid
        
        for iteration in range(total_possible_edges):
            if time.time() > DEADLINE:
                break
            
            best_edge = None
            best_score = -1
            best_degrees = None
            
            for (u, v) in all_e:
                if (u, v) in added:
                    continue
                
                # Check validity
                common = neighbors[u] & neighbors[v]
                if len(common) >= 3:
                    continue
                
                # Score: prefer edges between lower-degree vertices
                # Use a measure like: minimize the maximum degree after adding
                new_max = max(degrees[u] + 1, degrees[v] + 1)
                # Also consider overall balance
                score = -new_max  # Lower is better (we negate for max)
                
                # Tie-break: prefer edges that increase both degrees
                if score > best_score or (score == best_score and best_edge is None):
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            neighbors[u].add(v)
            neighbors[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            edges.append((u, v))
            added.add((u, v))
        
        return edges
    
    # Strategy 2: Randomized greedy with multiple restarts
    def randomized_greedy(num_restarts=50):
        best = []
        best_len = 0
        
        for r in range(num_restarts):
            if time.time() > DEADLINE:
                break
            
            # Random shuffle of edges
            order = all_edges[:]
            random.shuffle(order)
            
            edges = greedy_build(order)
            if len(edges) > best_len:
                best = edges
                best_len = len(edges)
        
        return best, best_len
    
    # First try degree-balanced greedy
    db_edges = degree_balanced_greedy()
    if len(db_edges) > best_count:
        best_edges = db_edges
        best_count = len(db_edges)
    
    if time.time() > DEADLINE:
        return [[u, v] for u, v in best_edges]
    
    # Then try randomized restarts with remaining time
    remaining_time = DEADLINE - time.time()
    if remaining_time > 2:
        # Estimate how many restarts we can do
        # Each restart takes roughly (190 * avg_check_time)
        # Be conservative: try many quick restarts
        num_restarts = int(min(remaining_time * 20, 500))  # Cap at 500
        rand_edges, rand_count = randomized_greedy(num_restarts)
        if rand_count > best_count:
            best_edges = rand_edges
            best_count = rand_count
    
    # Try a few more targeted strategies
    if time.time() < DEADLINE - 5:
        # Try sorted by various criteria
        # Sort by sum of degrees in some heuristic order
        orders = [
            sorted(all_edges, key=lambda e: e[0] * 100 + e[1]),  # Row-major
            sorted(all_edges, key=lambda e: (e[0] + e[1]) % n),  # By sum mod n
            sorted(all_edges, key=lambda e: (e[0] * e[1]) % n),  # By product mod n
        ]
        
        for order in orders:
            if time.time() > DEADLINE:
                break
            edges = greedy_build(order)
            if len(edges) > best_count:
                best_edges = edges
                best_count = len(edges)
    
    # Return as list of lists
    return [[u, v] for u, v in best_edges]


# Keep the EVOLVE-BLOCK markers for compatibility
# EVOLVE-BLOCK-START
# (construct function is defined above)
# EVOLVE-BLOCK-END
