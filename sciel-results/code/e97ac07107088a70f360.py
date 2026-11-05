import math
from functools import lru_cache

def construct():
    N = 512
    
    # 1. Generate all numbers 1..N
    nodes = list(range(1, N + 1))
    
    # 2. Build conflict graph
    # Edge (u, v) if lcm(u, v) > N
    # We can precompute gcds or just use math.gcd
    # lcm(u, v) = u * v // gcd(u, v)
    
    # To speed up, we can represent the graph as a list of bitsets or just lists of neighbors
    # 512 nodes is small enough for list of lists
    
    neighbors = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            g = math.gcd(u, v)
            l = u * v // g
            if l > N:
                neighbors[u].append(v)
                neighbors[v].append(u)
                
    # 3. Greedy Coloring for Upper Bound
    # Order nodes by degree descending? Or just 1..N
    # We need a coloring of the graph to bound the independent set size.
    # The independence number is <= the chromatic number? No, <= sum of colors?
    # Actually, for a graph, alpha(G) <= n / (min_degree + 1) is weak.
    # Better: Greedy coloring provides an upper bound on the size of the independent set?
    # No, the chromatic number is a LOWER bound on the clique number, not directly an upper bound on independent set.
    # Wait. The size of the largest independent set is at most the number of colors used in a valid coloring?
    # Yes. In any valid coloring, no two adjacent nodes have the same color.
    # An independent set can contain at most one node of each color.
    # So |IS| <= number of colors used.
    
    # Greedy Coloring
    # We process nodes in a specific order.
    # To get a tight bound, we often order by degree.
    
    # Let's implement a simple greedy coloring
    def greedy_coloring(order):
        colors = {}
        next_color = 0
        for u in order:
            used_colors = set()
            for v in neighbors[u]:
                if v in colors:
                    used_colors.add(colors[v])
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            colors[u] = c
            if c > next_color:
                next_color = c
        return next_color + 1 # Number of colors used
        
    # Order nodes by degree descending to get tighter bound?
    # Usually, high degree nodes first in coloring leads to more colors?
    # Actually, we want FEW colors to get a tight upper bound?
    # No, we want the upper bound to be as LOW as possible to prune.
    # So we want a coloring that uses FEW colors.
    # Greedy with high degree first (Welsh-Powell) tends to use fewer colors.
    
    degrees = [len(neighbors[i]) for i in range(1, N + 1)]
    order = sorted(range(1, N + 1), key=lambda x: degrees[x], reverse=True)
    
    upper_bound = greedy_coloring(order)
    
    # 4. Branch and Bound
    # We want to maximize the size of the independent set.
    # Current set S. Candidate set C.
    # If len(S) + len(C) <= best, prune.
    # If len(S) + upper_bound(C) <= best, prune.
    
    # To compute upper_bound(C), we can re-color the subgraph induced by C.
    # This is expensive.
    # Alternative: Use the global coloring?
    # If we have a global coloring, the number of colors available in C is an upper bound on the IS size in C.
    # Let's compute a global coloring once.
    
    # Re-compute coloring with the specific order
    global_colors = {}
    next_c = 0
    for u in order:
        used = set()
        for v in neighbors[u]:
            if v in global_colors:
                used.add(global_colors[v])
        c = 0
        while c in used:
            c += 1
        global_colors[u] = c
        if c > next_c:
            next_c = c
            
    max_color = next_c + 1
    
    # For a subset C, the upper bound on IS size is the number of distinct colors in C?
    # No. An independent set can pick at most one node of each color.
    # So yes, the number of distinct colors present in C is an upper bound.
    
    best_set = set()
    # Initialize with divisors of 360 (size 24)
    M_init = 360
    for i in range(1, int(M_init**0.5) + 1):
        if M_init % i == 0:
            best_set.add(i)
            if i != M_init // i:
                best_set.add(M_init // i)
                
    best_size = len(best_set)
    
    # B&B Search
    # We pick nodes one by one.
    # To make it faster, we can restrict the search to nodes that are "compatible" with the current best set?
    # No, we need to find a better set.
    
    # State: current_set, available_nodes
    # We can use recursion.
    # To avoid deep recursion or slow execution, we need good pruning.
    
    # Let's try a simple recursive search with the global color bound.
    
    def solve(current_set, available_nodes):
        nonlocal best_set, best_size
        
        if not available_nodes:
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
            return
            
        # Upper bound: len(current_set) + number of distinct colors in available_nodes
        colors_in_avail = set(global_colors[u] for u in available_nodes)
        ub = len(current_set) + len(colors_in_avail)
        
        if ub <= best_size:
            return
            
        # Pick a node
        # Heuristic: Pick node with highest degree in available subgraph?
        # Or just the first one.
        # To improve, sort available by degree?
        
        # Let's pick the first node in the sorted order (high degree first)
        # But we need to maintain the set of available nodes.
        
        # Optimization: If we include u, we must remove all neighbors of u from available.
        # If we exclude u, we just remove u.
        
        # We iterate through candidates.
        # To speed up, we can branch on a specific node.
        # Let's pick the node in available_nodes that has the highest degree in the induced subgraph?
        # Calculating degree in induced subgraph is O(N^2) if done naively.
        # With 512 nodes, it's fine.
        
        # Find pivot
        pivot = None
        max_deg = -1
        for u in available_nodes:
            deg = sum(1 for v in available_nodes if v in neighbors[u])
            if deg > max_deg:
                max_deg = deg
                pivot = u
                
        if pivot is None:
             # All isolated?
             if len(current_set) + len(available_nodes) > best_size:
                 best_set = current_set.union(available_nodes)
                 best_size = len(best_set)
             return

        # Branch 1: Include pivot
        new_available_include = [u for u in available_nodes if u not in neighbors[pivot] and u != pivot]
        solve(current_set.union([pivot]), new_available_include)
        
        # Branch 2: Exclude pivot
        new_available_exclude = [u for u in available_nodes if u != pivot]
        solve(current_set, new_available_exclude)
        
    # To prevent timeout, we might need to limit the depth or use a heuristic.
    # 512 nodes is small, but the branching factor is high.
    # The color bound is strong if the graph is dense.
    # The conflict graph: u-v edge if lcm(u,v) > 512.
    # How dense is it?
    # For small u, v, lcm is small.
    # For large u, v, lcm is likely large.
    # So the graph is somewhat dense for large numbers.
    
    # Initial available: all nodes
    # But we already have a good best_set.
    # We can start with available = all nodes.
    
    # To make it faster, we can remove nodes that are "dominated" or just rely on B&B.
    # Let's run it.
    
    # One issue: `neighbors` is a list of lists. Checking `u in neighbors[pivot]` is O(N).
    # Using sets for neighbors is faster.
    
    neighbor_sets = [set(neighbors[i]) for i in range(N + 1)]
    
    def solve_fast(current_set, available_set):
        nonlocal best_set, best_size
        
        if not available_set:
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
            return
            
        colors_in_avail = set(global_colors[u] for u in available_set)
        ub = len(current_set) + len(colors_in_avail)
        
        if ub <= best_size:
            return
            
        # Pick pivot
        pivot = None
        max_deg = -1
        for u in available_set:
            # Degree in available subgraph
            deg = len(neighbor_sets[u].intersection(available_set))
            if deg > max_deg:
                max_deg = deg
                pivot = u
                
        if pivot is None:
            if len(current_set) + len(available_set) > best_size:
                best_set = current_set.union(available_set)
                best_size = len(best_set)
            return

        # Branch 1: Include pivot
        new_avail_inc = available_set - neighbor_sets[pivot] - {pivot}
        solve_fast(current_set.union({pivot}), new_avail_inc)
        
        # Branch 2: Exclude pivot
        new_avail_exc = available_set - {pivot}
        solve_fast(current_set, new_avail_exc)

    solve_fast(set(), set(range(1, N + 1)))
    
    return list(best_set)
