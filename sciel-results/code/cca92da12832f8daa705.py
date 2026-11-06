# EVOLVE-BLOCK-START
import time
from random import Random

def _build_graph(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _greedy_clique_coloring(adj, deadline):
    """
    Treats edge partition as coloring edges of G such that each color class is a clique in G.
    Uses a greedy heuristic: 
    1. Maintain a list of uncolored edges.
    2. For each uncolored edge, determine which existing color classes (cliques) it can join.
    3. Prioritize edges with the fewest options (most constrained).
    4. Assign the edge to the "best" available clique (largest clique or most compatible).
    5. If no existing clique fits, start a new clique with this edge.
    """
    n = len(adj)
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    if not edges:
        return []
        
    # To ensure determinism and consistent behavior, sort edges initially
    # though the heuristic will re-prioritize dynamically
    edges.sort()
    
    # Map edge to index for quick lookup? Or just use set of tuples.
    # Since n is small (32, 64), total edges <= ~2000.
    # We will track uncolored edges as a set of (u, v) tuples.
    uncolored = set(edges)
    
    # List of cliques (each is a set of vertices)
    cliques = []
    
    # Helper to check if an edge (u, v) can be added to a clique C
    # C must be a clique, so u and v must be connected to ALL vertices in C \ {u, v}
    # And u-v must be an edge (which it is).
    def can_add_to_clique(u, v, C):
        # C is a set of vertices
        # u and v must be in C? No, we are adding them.
        # The new clique will be C U {u, v}.
        # For this to be a clique, u must be connected to all in C, v connected to all in C.
        # Note: u and v are already connected.
        for w in C:
            if w == u or w == v:
                continue
            if w not in adj[u] or w not in adj[v]:
                return False
        return True

    # Optimization: Precompute adjacency as bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
            
    def can_add_to_clique_fast(u, v, C_mask):
        # C_mask is bitmask of vertices in C
        # We need (adj_mask[u] & C_mask) == C_mask without u, v
        # and (adj_mask[v] & C_mask) == C_mask without u, v
        # Let C_ex = C_mask & ~(1<<u) & ~(1<<v)
        C_ex = C_mask & ~(1 << u) & ~(1 << v)
        if (adj_mask[u] & C_ex) != C_ex:
            return False
        if (adj_mask[v] & C_ex) != C_ex:
            return False
        return True

    # Store cliques as (set_of_vertices, bitmask)
    # cliques_list: list of [set, mask]
    
    while uncolored and time.time() < deadline:
        # 1. Select the next edge to color
        # Heuristic: Choose edge with minimum number of compatible cliques
        # If tie, choose edge with max degree sum? Or just arbitrary?
        # To save time, we don't check all uncolored edges against all cliques every step.
        # We can batch this or just pick a random edge and check.
        # Given the "saturation" hint, we should try to be smart.
        # However, checking all uncolored (up to 2000) against all cliques (up to 2000) is O(E^2 * K).
        # For E=2000, that's 4M * K checks. Too slow if done naively in Python loop.
        
        # Alternative: Just pick an edge, try to fit it. If it fits many, pick the best fit.
        # If it fits none, start new.
        # To improve, we can prioritize edges that are "hard" to place.
        # Let's try a simpler approach first: Pick an edge, find best clique.
        # If we want to be "saturation" based, we need to know options.
        
        # Let's use a semi-greedy approach:
        # Pick a random edge from uncolored? No, that's bad.
        # Pick edge with highest degree?
        
        # Let's try: Pick an edge. Calculate compatible cliques.
        # If > 0, assign to the one that results in the "densest" extension?
        # Actually, the goal is to MINIMIZE number of cliques.
        # So we want to pack edges into existing cliques as much as possible.
        
        # Strategy:
        # 1. Pick an edge (u,v) from uncolored.
        # 2. Find all cliques C where (u,v) can be added.
        # 3. If any exist, choose the one that maximizes the size of the resulting clique? 
        #    Or the one that leaves the most flexibility for others?
        #    Usually, adding to a larger clique is good because it consumes more "potential"
        #    but reduces the count of cliques by 0 (vs creating a new one).
        #    Wait, creating a new clique increases count by 1. Adding to existing increases by 0.
        #    So ANY existing clique is better than a new one.
        #    Among existing cliques, does it matter which one?
        #    Yes, because it changes the set of vertices in that clique, affecting future edges.
        #    A "safer" choice might be to add to a clique that is "compatible" with the most other uncolored edges?
        #    That's expensive to compute.
        
        # Let's stick to: If it fits, pick the first one found? Or the largest?
        # Let's try picking the edge that has the FEWEST options first (saturation).
        # To do this efficiently:
        # We can sample a few edges, or just iterate.
        # With N=64, E~1100.
        # Let's iterate through uncolored edges, count options, pick min.
        # This is O(E * C) per step. E~1000, C~100. 100k checks per step. 1000 steps -> 100M checks.
        # Too slow for Python in 120s? 100M simple ops is ~5-10s in C, ~50-100s in Python. Risky.
        
        # Optimized Saturation:
        # Maintain a list of uncolored edges.
        # For each step, pick a random subset of uncolored edges (e.g., 10) and check their options.
        # Pick the one with fewest options.
        
        if len(uncolored) > 10:
            import random
            sample = random.sample(list(uncolored), 10)
        else:
            sample = list(uncolored)
            
        best_edge = None
        best_options = []
        min_options_count = float('inf')
        
        for e in sample:
            u, v = e
            options = []
            for i, (C_set, C_mask) in enumerate(cliques):
                if can_add_to_clique_fast(u, v, C_mask):
                    options.append(i)
            if len(options) < min_options_count:
                min_options_count = len(options)
                best_edge = e
                best_options = options
                if min_options_count == 0:
                    break # Can't do better than 0 options (must create new)
                    
        if best_edge is None:
            # Fallback if sample was empty (shouldn't happen)
            best_edge = next(iter(uncolored))
            best_options = []
            for i, (C_set, C_mask) in enumerate(cliques):
                if can_add_to_clique_fast(best_edge[0], best_edge[1], C_mask):
                    best_options.append(i)

        u, v = best_edge
        uncolored.discard(best_edge)
        
        if best_options:
            # Choose the best option from best_options
            # Heuristic: Choose the clique that is currently largest?
            # Or the one where u and v have the highest degree within the clique?
            # Let's just pick the first one, or the one with max size.
            # Picking max size might be better for packing.
            target_idx = max(best_options, key=lambda i: len(cliques[i][0]))
            
            C_set, C_mask = cliques[target_idx]
            C_set.add(u)
            C_set.add(v)
            C_mask |= (1 << u) | (1 << v)
            cliques[target_idx] = [C_set, C_mask]
        else:
            # Create new clique
            new_set = {u, v}
            new_mask = (1 << u) | (1 << v)
            cliques.append([new_set, new_mask])
            
    # Convert to list of lists
    return [list(C_set) for C_set, _ in cliques]

def construct():
    start = time.time()
    deadline = start + 110 # Safety margin
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: trivial partition (edges)
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _greedy_clique_coloring(adj, deadline)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
