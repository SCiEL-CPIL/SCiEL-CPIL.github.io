"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
import itertools

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses Branch and Bound on the compatibility graph to find a maximum clique.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # Build adjacency matrix/list
    # Two nodes u, v are connected if lcm(u, v) <= 512
    # lcm(u, v) = u * v // gcd(u, v)
    
    # Precompute gcds? Or just check on the fly. 
    # N=512 is small enough for O(N^2) setup.
    
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            l = (i * j) // gcd(i, j)
            if l <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Heuristic: Order nodes by degree (descending) to prune B&B effectively
    # We want to pick high-degree nodes first.
    # However, B&B on Max Clique usually works best if we process vertices in a specific order.
    # Let's create a list of candidates sorted by degree descending.
    # Actually, standard Tomita algorithm for Max Clique uses a specific vertex ordering.
    # Let's use a simple recursive B&B with a global best set.
    
    best_clique = []
    
    # Initial heuristic: Greedy clique to get a good lower bound
    current_clique = []
    remaining = set(nodes)
    while remaining:
        # Pick node with highest degree in remaining
        # For speed, just pick max degree
        max_deg_node = max(remaining, key=lambda x: len(adj[x] & remaining))
        current_clique.append(max_deg_node)
        # Intersect remaining with neighbors of max_deg_node
        remaining = remaining & adj[max_deg_node]
    
    best_clique = current_clique
    best_size = len(best_clique)
    
    # Branch and Bound
    # We will search for a clique larger than best_size.
    # State: (current_clique, candidates)
    # candidates are nodes that can be added to current_clique (adjacent to all in current_clique)
    
    # To improve B&B, we should order the initial candidates.
    # Let's sort nodes by degree descending.
    sorted_nodes = sorted(nodes, key=lambda x: len(adj[x]), reverse=True)
    
    def bnb(clique, candidates):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            return
        
        # Pruning: If even taking all remaining candidates doesn't beat best_size, stop
        if len(clique) + len(candidates) <= best_size:
            return
        
        # Greedy Coloring for tighter upper bound
        cand_list = list(candidates)
        
        # Compute degrees in the candidate subgraph for sorting
        # This helps the greedy coloring produce a tighter bound
        degs = {}
        for u in cand_list:
            degs[u] = len(adj[u] & candidates)
        
        # Sort by degree descending
        sorted_cands = sorted(cand_list, key=lambda x: degs[x], reverse=True)
        
        # Assign colors
        color_of = {}
        color_sets = []
        
        for u in sorted_cands:
            used_colors = set()
            for v in adj[u]:
                if v in candidates and v in color_of:
                    used_colors.add(color_of[v])
            
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[u] = c
            if c >= len(color_sets):
                color_sets.append(set())
            color_sets[c].add(u)
            
        num_colors = len(color_sets)
        
        # Upper bound check
        if len(clique) + num_colors <= best_size:
            return
            
        # Optimization: If num_colors is 1, max clique size in candidates is 1
        if num_colors == 1:
            clique.append(sorted_cands[0])
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            clique.pop()
            return
            
        # Expansion: Pick the vertex with highest degree in candidate subgraph
        u = sorted_cands[0]
        
        # Branch 1: Include u
        clique.append(u)
        new_candidates = candidates - {u}
        new_candidates &= adj[u]
        bnb(clique, new_candidates)
        clique.pop()
        
        # Branch 2: Exclude u
        if len(clique) + len(candidates - {u}) > best_size:
            bnb(clique, candidates - {u})

    # Start B&B
    # Initial candidates: All nodes
    bnb([], set(nodes))
    
    # Convert set to list for return
    # The problem asks for a subset. Order doesn't matter, but list is standard.
    return sorted(best_clique)

# EVOLVE-BLOCK-END
