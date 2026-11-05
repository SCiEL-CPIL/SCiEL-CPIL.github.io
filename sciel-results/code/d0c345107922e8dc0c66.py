"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """Construct a graph with 31 vertices and 91 edges minimizing C4 count.
    
    Strategy: Build the polarity graph (C4-free, 90 edges) using a known
    algebraic construction, then add the 91st edge that creates the fewest
    new C4s.
    """
    start_time = time.time()
    deadline = 140.0
    n = 31
    q = 5
    
    # Construction of the polarity graph for q=5 (n=31 vertices, 90 edges, 0 C4s)
    # Using the standard construction from finite geometry.
    # 
    # Vertices: 
    #   Type 1: (a, b) for a,b in GF(5) -> 25 vertices, labels 0..24
    #   Type 2: (a, inf) for a in GF(5) -> 5 vertices, labels 25..29  
    #   Type 3: (inf, inf) -> 1 vertex, label 30
    #
    # The adjacency is defined by a symmetric bilinear form (polarity).
    # For this to work correctly and give 90 edges with 0 C4s, we use:
    #
    # Let me use a verified construction. I'll define the graph using
    # modular arithmetic with the following rules:
    
    def is_adjacent(u, v):
        """Determine if vertices u and v are adjacent in the polarity graph."""
        if u == v:
            return False
        
        # Determine types
        u_type = 1 if u < 25 else (2 if u < 30 else 3)
        v_type = 1 if v < 25 else (2 if v < 30 else 3)
        
        if u_type == 3:
            u, v = v, u
            u_type, v_type = v_type, u_type
        
        if u_type == 1 and v_type == 1:
            # Both finite: (a1,b1) and (a2,b2)
            a1, b1 = u // 5, u % 5
            a2, b2 = v // 5, v % 5
            # Adjacency: a1*a2 + b1*b2 + b1 + b2 ≡ 0 (mod 5)
            return (a1 * a2 + b1 * b2 + b1 + b2) % 5 == 0
        
        elif u_type == 1 and v_type == 2:
            # (a,b) and (c,inf)
            a, b = u // 5, u % 5
            c = v - 25
            # Adjacency: b ≡ c (mod 5)
            return b == c
        
        elif u_type == 1 and v_type == 3:
            # (a,b) and (inf,inf)
            a, b = u // 5, u % 5
            # Adjacency: a^2 + b ≡ 0 (mod 5)  [i.e., b ≡ -a^2]
            return (a * a + b) % 5 == 0
        
        elif u_type == 2 and v_type == 2:
            # (a,inf) and (b,inf)
            a = u - 25
            b = v - 25
            # Not adjacent
            return False
        
        elif u_type == 2 and v_type == 3:
            # (a,inf) and (inf,inf)
            a = u - 25
            # Adjacency: a^2 ≡ -1 ≡ 4 (mod 5)
            return (a * a) % 5 == 4
        
        return False
    
    # Build the polarity graph edges
    edges = set()
    for i in range(31):
        for j in range(i + 1, 31):
            if is_adjacent(i, j):
                edges.add((i, j))
    
    # Verify edge count
    edge_count = len(edges)
    
    if edge_count != 90:
        # If our construction doesn't give exactly 90 edges, 
        # use a fallback greedy construction
        # Try to build a C4-free graph greedily
        adj = [set() for _ in range(31)]
        edges = set()
        
        # Try all possible edges in a smart order
        all_pairs = [(i, j) for i in range(31) for j in range(i+1, 31)]
        random.seed(42)
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if len(edges) >= 90:
                break
            # Check if adding (u,v) creates a C4
            # A C4 is created if there exists w such that (u,w) and (v,w) are both in edges
            # and w is not u or v, AND there's another common neighbor... 
            # Actually, adding edge (u,v) creates a C4 iff u and v have at least 2 common neighbors
            common = adj[u] & adj[v]
            if len(common) < 2:
                # Safe to add
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        # If we still don't have 90, try different random seeds
        if len(edges) < 90:
            for seed in range(100):
                if time.time() - start_time > deadline - 10:
                    break
                adj = [set() for _ in range(31)]
                edges = set()
                all_pairs = [(i, j) for i in range(31) for j in range(i+1, 31)]
                random.seed(seed)
                random.shuffle(all_pairs)
                
                for u, v in all_pairs:
                    if len(edges) >= 90:
                        break
                    common = adj[u] & adj[v]
                    if len(common) < 2:
                        edges.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                
                if len(edges) == 90:
                    break
    
    # Rebuild adjacency from edges
    adj = [set() for _ in range(31)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find the best edge to add (minimizes C4s created)
    best_edge = None
    best_c4s = float('inf')
    
    for u in range(31):
        for v in range(u + 1, 31):
            if v not in adj[u]:
                common = len(adj[u] & adj[v])
                c4s = common * (common - 1) // 2 if common >= 2 else 0
                if c4s < best_c4s:
                    best_c4s = c4s
                    best_edge = (u, v)
    
    if best_edge is not None:
        edges.add(best_edge)
    
    # Ensure we have exactly 91 edges
    if len(edges) > 91:
        # Remove excess edges (shouldn't happen, but safety)
        edge_list = sorted(edges)
        edges = set(edge_list[:91])
    elif len(edges) < 91:
        # Add more edges if needed
        adj = [set() for _ in range(31)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        all_pairs = [(i, j) for i in range(31) for j in range(i+1, 31)]
        for u, v in all_pairs:
            if len(edges) >= 91:
                break
            if (u, v) not in edges and (v, u) not in edges:
                edges.add((u, v))
    
    result = [[u, v] for u, v in sorted(edges)]
    return result
# EVOLVE-BLOCK-END
