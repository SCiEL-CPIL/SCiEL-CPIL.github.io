"""Clique partition of two fixed dense random graphs via Simulated Annealing on Complement Graph Coloring."""
# EVOLVE-BLOCK-START
import time
import random as _random
import math

_T0 = time.time()
_DEADLINE = _T0 + 115.0  # Wall-clock guard


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _solve_instance_coloring(edges, n, time_budget):
    """
    Solve clique partition by coloring the complement graph.
    A clique in G is an independent set in G_bar.
    Partitioning vertices of G into cliques is equivalent to vertex coloring of G_bar.
    """
    if not edges:
        return []

    # Build adjacency matrix for G (True if edge exists)
    # For complement graph, we need to know if there is NO edge.
    # adj_complement[u][v] = True if u, v are NOT connected in G.
    
    # Use bitmasks for efficiency
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Complement adjacency: ~adj[u] & ((1<<n)-1) excluding u itself
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for u in range(n):
        comp_adj[u] = (full_mask ^ adj[u]) & ~(1 << u)

    def get_colors(adj_mask, n):
        """
        Greedy DSATUR-like coloring for the complement graph.
        Returns a list of colors for each vertex.
        """
        colors = [-1] * n
        # Track available colors for each vertex? 
        # Simple greedy: sort by degree in comp graph, assign smallest available color.
        
        # Calculate degrees in comp graph
        degs = [0] * n
        for u in range(n):
            degs[u] = bin(comp_adj[u]).count('1')
            
        order = sorted(range(n), key=lambda x: -degs[x])
        
        # For each vertex in order, assign smallest color not used by neighbors
        for u in order:
            # Get colors of neighbors in comp graph
            neighbor_colors = set()
            mask = comp_adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if colors[v] != -1:
                    neighbor_colors.add(colors[v])
                mask ^= lsb
                
            # Find smallest color not in neighbor_colors
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[u] = c
            
        # Compact colors to 0..k-1
        used_colors = sorted(set(colors))
        color_map = {old: new for new, old in enumerate(used_colors)}
        return [color_map[c] for c in colors]

    def count_cliques(colors):
        return max(colors) + 1 if colors else 0

    # Initial solution: Greedy
    best_colors = get_colors(comp_adj, n)
    best_count = count_cliques(best_colors)
    
    # Simulated Annealing
    # State: colors array
    # Cost: number of colors used
    # Move: Change color of a random vertex to a random valid color
    
    current_colors = best_colors[:]
    current_count = best_count
    
    t_start = time.time()
    T = 1.0
    cooling = 0.999
    
    # Estimate number of steps based on time
    # We want to run until time_budget is exhausted
    
    while time.time() - t_start < time_budget:
        if T < 0.01:
            break
            
        # Pick a random vertex
        u = _random.randint(0, n - 1)
        
        # Determine possible colors for u
        # Neighbors in comp graph cannot have the same color
        neighbor_colors = set()
        mask = comp_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if current_colors[v] != -1:
                neighbor_colors.add(current_colors[v])
            mask ^= lsb
            
        # We can assign any color not in neighbor_colors
        # To minimize colors, we should prefer existing colors or the smallest new color
        # But SA explores the space.
        
        # Current color
        c_old = current_colors[u]
        
        # Generate a new color
        # With some probability, try to remove a color (assign a color that makes the old color unused)
        # With some probability, try a random valid color
        
        # Let's just pick a random valid color from 0 to current_count
        # If we pick a color > current_count, we increase the count (bad, but allowed for SA)
        
        # To be efficient, let's restrict to colors 0..current_count
        # If we want to reduce count, we must assign a color such that c_old is no longer used.
        
        # Simple move: Pick a random color c_new in [0, current_count]
        # If c_new is not in neighbor_colors, it's a valid move.
        
        # If we want to allow increasing colors (to escape local minima), we can pick from [0, current_count + 1]
        
        # Let's try picking a random valid color
        # Valid colors are all c not in neighbor_colors.
        # We prioritize colors < current_count to keep size small, but allow current_count to potentially merge?
        # No, merging happens if we stop using a color.
        
        # Let's pick a random candidate from a small set of "interesting" colors
        # 1. c_old (no change, skip)
        # 2. Colors used by non-neighbors (might be same as c_old)
        # 3. Smallest color not in neighbor_colors (greedy best)
        
        # For SA, let's just pick a random integer from 0 to current_count
        # and check if it's valid.
        
        # To speed up, we can precompute a list of valid colors?
        # neighbor_colors is a set.
        
        # Let's pick a random color from 0 to current_count
        # If it's in neighbor_colors, it's invalid.
        # We can retry a few times or just reject.
        
        # Better: Pick a random color from the set of valid colors?
        # Valid colors = {0, ..., current_count} \ neighbor_colors
        
        # Let's just pick a random color from 0 to current_count
        c_new = _random.randint(0, current_count)
        
        if c_new in neighbor_colors:
            continue
            
        # Apply move
        current_colors[u] = c_new
        
        # Calculate new count
        # We need to check if c_old is still used
        # And if c_new was already used
        
        # Efficient count update:
        # If c_old == c_new: no change
        # Else:
        #   Check if c_old is used by any other vertex
        #   Check if c_new was used by any other vertex
        
        # For simplicity and correctness, recalculate count?
        # n=64, so counting is O(n).
        
        new_count = count_cliques(current_colors)
        
        delta = new_count - current_count
        
        # Acceptance
        if delta < 0 or _random.random() < math.exp(-delta / T):
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_colors = current_colors[:]
        else:
            # Revert
            current_colors[u] = c_old
            
        T *= cooling
        
    # Convert colors to cliques
    cliques = {}
    for u, c in enumerate(best_colors):
        if c not in cliques:
            cliques[c] = []
        cliques[c].append(u)
        
    return [cliques[c] for c in sorted(cliques.keys())]


def _solve_instance(edges, n):
    """
    Solve a single instance.
    """
    if not edges:
        return []
        
    # Allocate time budget
    # We have ~110s total. Split between instances?
    # n=32 is small, n=64 is large.
    # Give n=32 a smaller budget, n=64 a larger one.
    
    # Check time remaining
    time_remaining = _DEADLINE - time.time() - 5.0 # 5s margin
    if time_remaining < 1.0:
        return [[u, v] for u, v in edges] # Fallback: 2-cliques
        
    # Heuristic split: 20% for small, 80% for large
    # But we call this for both. We need to know which one is which?
    # Or just use a fixed budget per call?
    # The caller `construct` should manage the budget.
    
    # Let's assume this function gets a time_budget argument in a refactor?
    # No, I must keep signature.
    # I will modify construct to pass time budgets.
    
    # For now, let's just run for a fixed short time?
    # No, that's bad.
    
    # I will change the signature of the internal helper to accept time.
    # And construct will split the time.
    
    # Wait, the current code has `_solve_instance(edges, n)`.
    # I will change the internal logic to check a global deadline or pass time.
    
    # Let's modify construct to split time.
    
    # In this draft, I'll just use a heuristic time split inside _solve_instance?
    # No, better to let construct handle it.
    
    # I'll add a parameter to _solve_instance?
    # The directive says: "Keep the function names and signatures of the seed program."
    # The seed program has `_solve_instance(edges, n)`.
    # I can add a default argument? `def _solve_instance(edges, n, time_budget=10.0):`
    # That changes the signature technically? Usually "signature" means name and required args.
    # But to be safe, I'll use a global variable for time budget?
    # Or I'll just hardcode a reasonable split in construct and call a new helper?
    
    # I will rename the internal solver to `_solve_instance_sa` and have `_solve_instance` call it?
    # No, I'll just modify `_solve_instance` to use a global `_TIME_BUDGET_FOR_THIS_CALL`.
    
    # Actually, I can just change `construct` to call `_solve_instance_coloring` directly with specific time budgets.
    # And I can keep `_solve_instance` as a wrapper or remove it?
    # The prompt says "Keep the function names... of the seed program".
    # The seed program has `_solve_instance`.
    # I will keep `_solve_instance` but make it use a global time slice.
    
    # Let's define a global `_CURRENT_TIME_BUDGET`.
    
    return _solve_instance_coloring(edges, n, _CURRENT_TIME_BUDGET)


_CURRENT_TIME_BUDGET = 10.0

def construct():
    """
    Entry point.
    """
    global _CURRENT_TIME_BUDGET
    
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Split time
    # n=32 is much faster.
    # Give 20s to n=32, 85s to n=64.
    # Total 105s. Deadline is 115s.
    
    time_left = _DEADLINE - time.time() - 5.0
    
    # Proportional to n^2?
    # 32^2 = 1024, 64^2 = 4096. Ratio 1:4.
    # So 20% and 80%.
    
    budget1 = time_left * 0.2
    budget2 = time_left * 0.8
    
    _CURRENT_TIME_BUDGET = budget1
    res1 = _solve_instance(edges1, 32)
    
    _CURRENT_TIME_BUDGET = budget2
    res2 = _solve_instance(edges2, 64)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
