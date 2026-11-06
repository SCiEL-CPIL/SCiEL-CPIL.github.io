import time
import random
import itertools

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 145.0
START_TIME = None

def _find_violation(adj_masks, n):
    """
    Find a subset S of size >= 6 such that the induced subgraph on S is regular.
    Returns the subset as a tuple of vertices, or None.
    Uses bitmasks for fast degree calculation.
    """
    # Check size 6 first, as these are the most numerous and likely to exist
    # If a size 6 violation exists, we must fix it.
    for combo in itertools.combinations(range(n), 6):
        # Construct mask for this subset
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        # Check if all vertices in combo have the same degree within the subset
        # Degree of v in subset S is popcount(adj[v] & mask_S)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        if is_regular:
            return combo

    # Check larger subsets
    # Note: A regular induced subgraph of size k > 6 is not necessarily composed of
    # regular induced subgraphs of size 6. So we must check them.
    # However, if we fix all size 6 violations, larger ones might disappear or be easier to find.
    # For n <= 22, C(22, 7) is ~170k, C(22, 8) is ~320k. Total is manageable if optimized.
    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            is_regular = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_regular = False
                    break
            if is_regular:
                return combo
    return None

def _edges_from_masks(adj_masks, n):
    """Convert adjacency masks to edge list."""
    edges = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                edges.append([u, v])
    return edges

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    deadline = START_TIME + DEADLINE_SECONDS
    rng = random.Random(42) # Fixed seed for reproducibility
    
    best_n = 0
    best_edges = []
    
    # Try to find a graph with n vertices, starting from the maximum possible
    # The problem states n <= 22.
    for n_target in range(22, 5, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Number of attempts for this n. Larger n is harder, but we have less time left.
        # Let's try a few random restarts.
        max_attempts = 10 if n_target <= 20 else 20
        
        found_valid = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize random graph
            # Edge probability p. 
            # Regular subgraphs are more likely in random graphs. 
            # We want to avoid them. 
            # Let's try p=0.5 initially, maybe vary it.
            p = 0.5
            adj_masks = [0] * n_target
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if rng.random() < p:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
            
            # Local Search
            # We repeatedly find a violation and flip an edge to break it.
            # Strategy:
            # 1. Find a violating subset S.
            # 2. If S is not the whole graph, pick u in S and w not in S, flip edge (u, w).
            #    This changes the degree of u in S by +/- 1, breaking regularity of S.
            #    It does not change the degrees of other vertices in S relative to each other.
            # 3. If S is the whole graph, flip a random internal edge.
            
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > deadline - 1.0:
                    break
                    
                viol = _find_violation(adj_masks, n_target)
                if viol is None:
                    # No violation found!
                    found_valid = True
                    best_n = n_target
                    best_edges = _edges_from_masks(adj_masks, n_target)
                    break
                
                # viol is a tuple of vertices
                v_set = set(viol)
                
                if len(v_set) == n_target:
                    # Violation is the whole graph. Flip a random internal edge.
                    # Pick two random vertices from the set
                    u, w = rng.sample(list(v_set), 2)
                else:
                    # Violation is a proper subset. Flip an edge between S and V\S.
                    # This is more effective because it only changes the degree of one vertex in S.
                    u = rng.choice(list(v_set))
                    # Pick a vertex outside S
                    outside = [v for v in range(n_target) if v not in v_set]
                    w = rng.choice(outside)
                
                # Flip edge (u, w)
                adj_masks[u] ^= (1 << w)
                adj_masks[w] ^= (1 << u)
                
                # Optional: If we get stuck in a loop, we might want to randomize more,
                # but the boundary flip is quite strong.
            
            if found_valid:
                break
        
        if found_valid:
            # Since we iterate n_target in descending order, the first valid one is the largest.
            break
            
    if best_n == 0:
        # Fallback: return a small valid graph
        # Any graph with n < 6 is valid.
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
