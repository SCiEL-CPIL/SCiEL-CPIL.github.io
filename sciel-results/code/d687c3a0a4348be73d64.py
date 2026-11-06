import random

def construct():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Fast validity check using adjacency and edge colors
    # We maintain:
    # edge_colors: dict (u,v) -> c
    # adj: list of sets
    # For fast checking, we can precompute 4-cycles? No, too many.
    # We check on the fly.
    
    def is_valid_addition(edge_colors, adj, u, v, c):
        # Potential 4-cycles: u-a-b-v-u
        # a in adj[u], b in adj[v], a-b in edge_colors
        # a != v, b != u, a != b
        for a in adj[u]:
            if a == v:
                continue
            c_ua = edge_colors.get((min(u, a), max(u, a)))
            for b in adj[v]:
                if b == u or b == a:
                    continue
                key_ab = (min(a, b), max(a, b))
                c_ab = edge_colors.get(key_ab)
                if c_ab is None:
                    continue
                c_bv = edge_colors.get((min(b, v), max(b, v)))
                # Check if c, c_ua, c_ab, c_bv are distinct
                if c == c_ua or c == c_ab or c == c_bv or c_ua == c_ab or c_ua == c_bv or c_ab == c_bv:
                    return False
        return True

    def add_edge(edge_colors, adj, u, v, c):
        key = (min(u, v), max(u, v))
        edge_colors[key] = c
        adj[u].add(v)
        adj[v].add(u)

    def run_greedy(order):
        edge_colors = {}
        adj = [set() for _ in range(n)]
        for u, v in order:
            key = (min(u, v), max(u, v))
            if key in edge_colors:
                continue
            for c in range(20):
                if is_valid_addition(edge_colors, adj, u, v, c):
                    add_edge(edge_colors, adj, u, v, c)
                    break
        return len(edge_colors), edge_colors

    best_edges = 0
    best_dict = {}
    
    # Strategy 1: Random orders
    for _ in range(20):
        random.shuffle(all_edges)
        cnt, dct = run_greedy(all_edges)
        if cnt > best_edges:
            best_edges = cnt
            best_dict = dct

    # Strategy 2: Bipartite First (K10,10)
    # This often yields high density
    a_part = list(range(10))
    b_part = list(range(10, 20))
    bip_edges = [(u, v) for u in a_part for v in b_part]
    intra_a = [(u, v) for u in a_part for v in a_part if u < v]
    intra_b = [(u, v) for u in b_part for v in b_part if u < v]
    
    # Try different shuffles of bipartite edges
    for _ in range(20):
        random.shuffle(bip_edges)
        random.shuffle(intra_a)
        random.shuffle(intra_b)
        # Interleave or sequence?
        # Sequence: Bipartite first, then intra
        order = bip_edges + intra_a + intra_b
        cnt, dct = run_greedy(order)
        if cnt > best_edges:
            best_edges = cnt
            best_dict = dct
            
        # Try mixing
        mixed = bip_edges + intra_a + intra_b
        random.shuffle(mixed)
        cnt, dct = run_greedy(mixed)
        if cnt > best_edges:
            best_edges = cnt
            best_dict = dct

    # Strategy 3: Local Search / Hill Climbing on the best solution
    # Try to add edges that were rejected
    # Or swap colors to allow more edges
    # Given time, stick to multiple greedy runs with different heuristics
    
    # Convert best_dict to list
    result = [[u, v, c] for (u, v), c in best_dict.items()]
    return result
