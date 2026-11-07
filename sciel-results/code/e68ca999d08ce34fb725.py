"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
N = 15
ALL_PAIRS = list(combinations(range(N), 2))
PAIR_INDEX = {pair: i for i, pair in enumerate(ALL_PAIRS)}

def _build_adj_and_deg(edge_set):
    """Build adjacency list (as sets) and degree array from a set of edges."""
    adj = [set() for _ in range(N)]
    deg = [0] * N
    for (u, v) in edge_set:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _is_valid(edge_set):
    """Check if the graph satisfies the constraint.
    For every simple path s-u-v-t (i.e., edges (s,u), (u,v), (v,t) with s,t,u,v distinct),
    we must have deg(s) != deg(t)."""
    adj, deg = _build_adj_and_deg(edge_set)
    for (u, v) in edge_set:
        # s is in adj[u], s != v
        # t is in adj[v], t != u
        # s != t (simple path requires distinct vertices)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second buffer before 90s kill

    # Start with a valid base: a star centered at 0
    base_edges = set((0, v) for v in range(1, N))
    best_edge_set = base_edges.copy()
    best_count = len(best_edge_set)

    # Helper to check validity
    def check_validity(edges):
        return _is_valid(edges)

    # First, try to add edges greedily
    # Order of pairs: try to add non-star edges first? Or random order?
    # Let's try a deterministic greedy pass first to establish a good baseline.
    # Then random local search.

    # Greedy pass 1: add edges if possible
    # We iterate through all non-base pairs.
    non_base_pairs = [p for p in ALL_PAIRS if p not in base_edges]
    random.shuffle(non_base_pairs)

    current_edges = best_edge_set.copy()
    
    for pair in non_base_pairs:
        if time.time() > deadline:
            break
        candidate = current_edges | {pair}
        if check_validity(candidate):
            current_edges = candidate
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edge_set = current_edges.copy()

    # Random Local Search
    # State: current_edges
    # Move: Add a random missing edge, or remove a random existing edge (to escape local optima), or swap.
    # Strategy:
    # 1. Try adding random edges.
    # 2. If stuck (no additions possible in recent tries), try removing an edge from a "conflict" area or random edge.
    # 3. Accept if valid. If it improves best, update best.
    
    current_edges = best_edge_set.copy()
    stuck_counter = 0
    
    # Precompute list of all pairs for random selection
    all_pairs_list = ALL_PAIRS
    
    while time.time() < deadline:
        # Determine available edges to add
        missing = [p for p in all_pairs_list if p not in current_edges]
        
        if not missing:
            # No more edges to add. Try removal to enable other additions?
            # This is a local optimum for addition.
            # Try a random removal to escape.
            if current_edges:
                remove_pair = random.choice(list(current_edges))
                candidate = current_edges - {remove_pair}
                if check_validity(candidate):
                    current_edges = candidate
                    stuck_counter = 0
            else:
                break
        
        # Try adding a random missing edge
        if missing:
            add_pair = random.choice(missing)
            candidate = current_edges | {add_pair}
            if check_validity(candidate):
                current_edges = candidate
                stuck_counter = 0
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edge_set = current_edges.copy()
            else:
                stuck_counter += 1
                # If stuck for too long, try a removal to perturb state
                if stuck_counter > 50 and current_edges:
                    # Remove a random edge
                    remove_pair = random.choice(list(current_edges))
                    candidate = current_edges - {remove_pair}
                    if check_validity(candidate):
                        current_edges = candidate
                        stuck_counter = 0
        
        # Occasionally, try a "swap": remove one, add another
        # This is implicitly handled by the removal perturbation above,
        # but explicit swaps might be faster.
        # Let's add a 10% chance of attempting a swap if we have some edges.
        if time.time() < deadline and current_edges and random.random() < 0.1:
            remove_pair = random.choice(list(current_edges))
            missing = [p for p in all_pairs_list if p not in current_edges]
            if missing:
                add_pair = random.choice(missing)
                # Ensure we don't remove and add the same edge (impossible by construction)
                candidate = (current_edges - {remove_pair}) | {add_pair}
                if check_validity(candidate):
                    current_edges = candidate
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edge_set = current_edges.copy()

    # Convert best_edge_set to list of lists
    # Ensure integers are Python ints
    result = [[int(u), int(v)] for (u, v) in best_edge_set]
    return result

# EVOLVE-BLOCK-END
