import time
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))
    
    if not bad_triples:
        return elements
    
    # Build pairwise conflict graph: two nodes conflict if they appear together in any bad triple
    # We use this as a relaxation: any valid solution is an independent set in this graph
    # (but not every independent set is valid for the hypergraph)
    # However, for the branch-and-bound, we use the hypergraph constraint directly
    
    # For each node, store which other nodes form bad triples with it (as pairs)
    pairs_by_node = [set() for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_node[a].add((b, c))
        pairs_by_node[b].add((a, c))
        pairs_by_node[c].add((a, b))
    
    # Build adjacency bitmask: adj[i] has bit j set if i and j are in some bad triple
    adj = [0] * n
    for a, b, c in bad_triples:
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    # Degree of each node in the conflict graph
    degree = [bin(adj[i]).count('1') for i in range(n)]
    
    # Upper bound: use greedy coloring on the conflict graph for a tight bound
    # We'll use a simple greedy coloring heuristic
    def greedy_color_bound(available_mask):
        """Compute a greedy coloring upper bound on the size of MIS in the subgraph induced by available_mask."""
        # Get list of available nodes
        nodes = [i for i in range(n) if (available_mask >> i) & 1]
        if not nodes:
            return 0
        
        # Simple greedy coloring: assign colors greedily
        color = [-1] * n
        num_colors = 0
        for node in nodes:
            # Find smallest color not used by neighbors
            used_colors = set()
            for neighbor in range(n):
                if (adj[node] >> neighbor) & 1 and (available_mask >> neighbor) & 1:
                    if color[neighbor] >= 0:
                        used_colors.add(color[neighbor])
            c = 0
            while c in used_colors:
                c += 1
            color[node] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors
    
    # Branch and bound for maximum independent set
    best_size = 0
    best_mask = 0
    time_limit_exceeded = False
    
    def can_add_to_mask(selected_mask, node):
        """Check if adding node to selected_mask doesn't complete any bad triple."""
        for other1, other2 in pairs_by_node[node]:
            if (selected_mask & (1 << other1)) and (selected_mask & (1 << other2)):
                return False
        return True
    
    def branch_and_bound(available_mask, selected_mask, current_size):
        nonlocal best_size, best_mask, time_limit_exceeded
        
        if time.time() > deadline - 5:
            time_limit_exceeded = True
            return
        
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
        
        # Upper bound: current_size + greedy coloring bound
        ub = current_size + greedy_color_bound(available_mask)
        if ub <= best_size:
            return
        
        # Pick a node with high degree for branching
        # Find node with max degree in the induced subgraph
        best_node = -1
        best_deg = -1
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            node = lsb.bit_length() - 1
            deg = bin(adj[node] & available_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_node = node
            temp_mask ^= lsb
        
        if best_node == -1:
            return
        
        # Branch 1: include best_node
        if can_add_to_mask(selected_mask, best_node):
            new_selected = selected_mask | (1 << best_node)
            # Remove best_node and all its neighbors from available
            new_available = available_mask & ~(1 << best_node) & ~adj[best_node]
            branch_and_bound(new_available, new_selected, current_size + 1)
        
        if time_limit_exceeded:
            return
        
        # Branch 2: exclude best_node
        new_available = available_mask & ~(1 << best_node)
        branch_and_bound(new_available, selected_mask, current_size)
    
    # Initial greedy solution to set a good lower bound
    # Use multiple orderings
    orderings = [
        sorted(range(n), key=lambda i: degree[i]),
        sorted(range(n), key=lambda i: degree[i], reverse=True),
        list(range(n)),
        list(range(n - 1, -1, -1)),
    ]
    
    for order in orderings:
        if time.time() > deadline - 30:
            break
        selected_mask = 0
        for node in order:
            if can_add_to_mask(selected_mask, node):
                selected_mask |= (1 << node)
        sz = bin(selected_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = selected_mask
    
    # Run branch and bound
    all_mask = (1 << n) - 1
    branch_and_bound(all_mask, 0, 0)
    
    # Final local search on best solution
    if not time_limit_exceeded and time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for node in range(n):
                if time.time() > deadline - 10:
                    break
                if not (best_mask & (1 << node)):
                    if can_add_to_mask(best_mask, node):
                        best_mask |= (1 << node)
                        improved = True
    
    return [i + 1 for i in range(n) if (best_mask >> i) & 1]
