"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a dense triangle-free graph on 16 vertices that is not 3-colorable.
    Uses a bitset-based greedy fill with random restarts on a fixed 4-chromatic core.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all possible edges as bitmasks
    # Edge (u, v) where u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)  # 120
    
    # For each edge, precompute the set of vertices it's incident to
    # and the set of edges that would form a triangle if added
    # Edge index mapping
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # For each edge, compute which other edges are incompatible (would form a triangle)
    # If we have edge (u,v) and edge (v,w), then we cannot have edge (u,w)
    # So for edge (u,v), incompatible edges are those (u,w) or (w,v) for w != u,v
    # Actually, triangle (u,v,w) means edges (u,v), (v,w), (u,w) all present
    # So edge (u,v) is incompatible with (u,w) AND (v,w) simultaneously
    # But we need to track: if we add edge (u,v), we must NOT have both (u,w) and (v,w) for any w
    
    # Better approach: maintain adjacency as bitmasks
    # adj[v] is a bitmask of neighbors of v
    
    # Core: Grötzsch graph on vertices 0-10
    # This is 4-chromatic and triangle-free
    # Cycle: 0-1-2-3-4-0
    # Then 5,6,7,8,9 connected to form Mycielski
    # Vertex 10 is the new apex
    
    def make_groetzsch_core():
        """Return adjacency bitmasks for Grötzsch graph on vertices 0-10."""
        adj = [0] * n
        # Cycle 0-1-2-3-4-0
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Mycielski: for each edge (u,w) in cycle, add (u, 5+w) and (w, 5+u)
        # Actually, the standard Mycielski construction:
        # For cycle vertices 0..4, create new vertices 5..9
        # For each edge (u,v) in cycle, connect u to 5+v and v to 5+u
        # Wait, let me use the standard construction
        # Cycle: 0-1-2-3-4-0
        # New vertices 5-9 where 5+i is the copy of i
        # Connect 5+i to all neighbors of i in the cycle
        # Then vertex 10 connects to all 5..9
        
        # Reset
        adj = [0] * n
        cycle_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Mycielski vertices 5-9
        for i in range(5):
            # Vertex 5+i connects to neighbors of i in the cycle
            for j in range(5):
                if (1 << j) & adj[i]:
                    adj[5+i] |= (1 << j)
                    adj[j] |= (1 << (5+i))
        
        # Vertex 10 connects to 5-9
        for i in range(5, 10):
            adj[10] |= (1 << i)
            adj[i] |= (1 << 10)
        
        return adj
    
    def is_triangle_free(adj):
        """Check if graph with adjacency bitmasks is triangle-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << v) & adj[u]:
                    # Check common neighbors
                    common = adj[u] & adj[v]
                    if common:
                        return False
        return True
    
    def is_3_colorable(adj):
        """Check if graph is 3-colorable using backtracking with bitsets."""
        color = [-1] * n
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: bin(adj[v]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try colors 0, 1, 2
            for c in range(3):
                # Check if color c is available for v
                conflict = False
                for u in range(n):
                    if color[u] == c and (1 << u) & adj[v]:
                        conflict = True
                        break
                if not conflict:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def count_edges(adj):
        return sum(bin(adj[v]).count('1') for v in range(n)) // 2
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) to adj. Returns True if added without creating triangle."""
        if (1 << v) & adj[u]:
            return True  # Already present
        # Check if adding (u,v) creates a triangle
        # Triangle exists if there's w such that w is neighbor of both u and v
        common = adj[u] & adj[v]
        if common:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def greedy_fill(adj, rng):
        """Greedily add edges in random order, maintaining triangle-free property."""
        adj = list(adj)  # Copy
        edges_to_try = list(range(n * (n-1) // 2))
        rng.shuffle(edges_to_try)
        
        for idx in edges_to_try:
            # Convert linear index to (u, v)
            # This is a bit tricky; let's just iterate over all pairs
            pass
        
        # Better: iterate over all possible edges in random order
        pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(pairs)
        
        for u, v in pairs:
            if not ((1 << v) & adj[u]):
                common = adj[u] & adj[v]
                if not common:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj
    
    def local_search(adj, rng, max_iter=1000):
        """Try to add more edges via local search."""
        adj = list(adj)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            pairs = [(u, v) for u in range(n) for v in range(u+1, n) if not ((1 << v) & adj[u])]
            rng.shuffle(pairs)
            for u, v in pairs:
                common = adj[u] & adj[v]
                if not common:
                    # Adding this edge is safe
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        
        return adj
    
    def adj_to_edges(adj):
        """Convert adjacency bitmasks to list of edges."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << v) & adj[u]:
                    edges.append([u, v])
        return edges
    
    # Start with Grötzsch core
    best_adj = make_groetzsch_core()
    best_count = count_edges(best_adj)
    
    # Verify core is valid
    if not is_triangle_free(best_adj):
        raise ValueError("Core is not triangle-free")
    if is_3_colorable(best_adj):
        raise ValueError("Core is 3-colorable, should not be")
    
    # Random restarts with greedy fill
    rng = random.Random(42)
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Start from core
        adj = list(best_adj)
        
        # Greedy fill
        adj = greedy_fill(adj, rng)
        
        # Check if still 4-chromatic (not 3-colorable)
        if not is_3_colorable(adj):
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(adj)
        
        # Try local search from this state
        if time.time() < deadline:
            adj2 = local_search(adj, rng)
            if not is_3_colorable(adj2):
                cnt2 = count_edges(adj2)
                if cnt2 > best_count:
                    best_count = cnt2
                    best_adj = list(adj2)
        
        # Also try starting from a perturbed core
        if num_restarts % 5 == 0 and time.time() < deadline:
            adj3 = list(make_groetzsch_core())
            # Randomly remove some edges from core
            for _ in range(rng.randint(1, 5)):
                u = rng.randint(0, 10)
                neighbors = bin(adj3[u]).count('1')
                if neighbors > 0:
                    # Pick a random neighbor
                    nbrs = [v for v in range(11) if (1 << v) & adj3[u]]
                    if nbrs:
                        v = rng.choice(nbrs)
                        adj3[u] &= ~(1 << v)
                        adj3[v] &= ~(1 << u)
            adj3 = greedy_fill(adj3, rng)
            if not is_3_colorable(adj3):
                cnt3 = count_edges(adj3)
                if cnt3 > best_count:
                    best_count = cnt3
                    best_adj = list(adj3)
    
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
