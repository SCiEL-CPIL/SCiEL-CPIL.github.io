"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict
import random

# EVOLVE-BLOCK-START

def _has_c4(n, adj):
    """Check if graph has a C4 using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            # Count common neighbors using bit operations
            common = adj[u] & adj[v]
            if common & (common - 1):  # More than one bit set
                return True
    return False

def _max_independent_set_size(n, adj, target=11):
    """Check if there exists an independent set of size >= target using branching with pruning."""
    # Use greedy coloring upper bound for pruning
    def greedy_color_bound(vertices):
        """Upper bound on independence number using greedy coloring."""
        if not vertices:
            return 0
        colors = [0] * len(vertices)
        max_color = 0
        for i, v in enumerate(vertices):
            used_colors = set()
            for j in range(i):
                if colors[j] > 0 and (adj[v] >> vertices[j]) & 1:
                    used_colors.add(colors[j])
            c = 1
            while c in used_colors:
                c += 1
            colors[i] = c
            max_color = max(max_color, c)
        return max_color
    
    def search(vertices, current_size, best):
        if current_size + len(vertices) <= best:
            return best
        if not vertices:
            return max(best, current_size)
        if current_size >= target:
            return target
        
        # Upper bound
        ub = greedy_color_bound(vertices)
        if current_size + ub <= best:
            return best
        
        # Pick vertex with highest degree in the subgraph
        v = vertices[0]
        rest = vertices[1:]
        
        # Option 1: include v
        new_rest = [u for u in rest if not ((adj[v] >> u) & 1)]
        best = search(new_rest, current_size + 1, best)
        
        # Option 2: exclude v
        best = search(rest, current_size, best)
        
        return best
    
    return search(list(range(n)), 0, 0)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check C4
    if _has_c4(n, adj):
        return False
    
    # Check independent set of size 11
    if _max_independent_set_size(n, adj, 11) >= 11:
        return False
    
    return True

def _construct_known_39():
    """Construct a known 39-vertex C4-free graph with alpha < 11."""
    # Use a construction based on the polarity graph of a projective plane
    # or a specific known construction
    
    # One approach: use vertices as (i, j) for i in range(6), j in range(6) plus 3 extra
    # This is complex. Let me try a different known construction.
    
    # Actually, let me try to build it systematically.
    # Start with a 36-vertex graph (6x6 grid structure) and add 3 more.
    
    n = 39
    edges = []
    adj = [0] * n
    
    # Create a base structure: 6 groups of 6 vertices
    # Within each group, make a complete graph (clique of size 6)
    # Between groups, add edges carefully to avoid C4
    
    # Actually, let me try a different approach: use a known good seed
    # and do local search.
    
    return None  # Will be filled by search

def _greedy_construct(n_target):
    """Greedy construction of a C4-free graph."""
    adj = [0] * n_target
    edges = []
    
    # Try all edges in random order, add if doesn't create C4
    all_pairs = list(itertools.combinations(range(n_target), 2))
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        # Check if adding (u,v) creates C4
        # C4 is created if there are two distinct common neighbors of u and v
        common = adj[u] & adj[v]
        if common & (common - 1):  # More than one common neighbor
            continue
        
        # Add the edge
        edges.append([u, v])
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    return n_target, edges, adj

def _local_search(n, adj, edges, deadline):
    """Local search to minimize independence number."""
    best_alpha = _max_independent_set_size(n, adj, 11)
    best_adj = adj[:]
    best_edges = edges[:]
    
    while time.time() < deadline:
        improved = False
        
        # Try adding random edges
        for _ in range(100):
            if time.time() > deadline:
                break
            
            u, v = random.sample(range(n), 2)
            if (adj[u] >> v) & 1:
                continue  # Already an edge
            
            # Check if adding creates C4
            common = adj[u] & adj[v]
            if common & (common - 1):
                continue
            
            # Try adding
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            new_alpha = _max_independent_set_size(n, new_adj, 11)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_adj = new_adj
                best_edges = edges + [[u, v]]
                adj = new_adj
                improved = True
                if best_alpha < 11:
                    return best_adj, best_edges
        
        # Try removing random edges (to allow adding others)
        if not improved:
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                if not edges:
                    break
                
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                
                # Remove edge
                new_adj = adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                # Now try to add some edges
                new_edges = edges[:idx] + edges[idx+1:]
                all_pairs = list(itertools.combinations(range(n), 2))
                random.shuffle(all_pairs)
                
                for a, b in all_pairs:
                    if time.time() > deadline:
                        break
                    if (new_adj[a] >> b) & 1:
                        continue
                    common = new_adj[a] & new_adj[b]
                    if common & (common - 1):
                        continue
                    new_adj[a] |= (1 << b)
                    new_adj[b] |= (1 << a)
                    new_edges.append([a, b])
                
                new_alpha = _max_independent_set_size(n, new_adj, 11)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_adj = new_adj
                    best_edges = new_edges
                    adj = new_adj
                    edges = new_edges
                    improved = True
                    if best_alpha < 11:
                        return best_adj, best_edges
    
    return best_adj, best_edges

def _try_construction():
    """Try to find a C4-free graph with no independent set of size 11."""
    start_time = time.time()
    deadline = start_time + 110  # Leave margin for verification
    
    best_n = 0
    best_edges = []
    best_adj = None
    
    # Try different sizes
    for n_try in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + n_try)
            
            n, edges, adj = _greedy_construct(n_try)
            
            # Check current alpha
            alpha = _max_independent_set_size(n, adj, 11)
            
            if alpha < 11:
                if n_try > best_n:
                    best_n = n_try
                    best_edges = edges
                    best_adj = adj
                    if n_try >= 39:
                        return {'n': best_n, 'edges': best_edges}
            else:
                # Try local search
                adj, edges = _local_search(n, adj, edges, deadline)
                alpha = _max_independent_set_size(n, adj, 11)
                if alpha < 11:
                    if n_try > best_n:
                        best_n = n_try
                        best_edges = edges
                        best_adj = adj
                        if n_try >= 39:
                            return {'n': best_n, 'edges': best_edges}
        
        # If we found a good one, stop trying smaller sizes
        if best_n >= n_try:
            break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    # Try to find a better construction
    result = _try_construction()
    
    # Verify the result
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
