import random
import time
import sys

def _has_c6(u, v, adj_bits, n):
    """
    Check if there's a simple path of length 5 from u to v.
    Path: u -> a -> b -> c -> d -> v, all distinct.
    """
    # Try all d in N(v)
    # For each d, check if there's a simple path u -> a -> b -> c -> d of length 4
    # avoiding v
    
    # Precompute neighbors as lists for faster iteration
    # Use bitmask for visited
    
    def has_path_len4(start, end, forbidden):
        """Check if simple path of length 4 from start to end avoiding forbidden."""
        # Path: start -> a -> b -> c -> end
        # a in N(start), b in N(a), c in N(b), end in N(c)
        # All of {start, a, b, c, end} distinct and != forbidden
        
        start_mask = 1 << start
        end_mask = 1 << end
        forbidden_mask = 1 << forbidden
        
        # a must be in N(start), a != start, a != end, a != forbidden
        for a in range(n):
            if a == start or a == end or a == forbidden:
                continue
            if not ((adj_bits[start] >> a) & 1):
                continue
            # b must be in N(a), b != start, b != a, b != end, b != forbidden
            for b in range(n):
                if b == start or b == a or b == end or b == forbidden:
                    continue
                if not ((adj_bits[a] >> b) & 1):
                    continue
                # c must be in N(b), c != start, c != a, c != b, c != end, c != forbidden
                for c in range(n):
                    if c == start or c == a or c == b or c == end or c == forbidden:
                        continue
                    if not ((adj_bits[b] >> c) & 1):
                        continue
                    # Check if end in N(c)
                    if (adj_bits[c] >> end) & 1:
                        return True
        return False
    
    # d must be in N(v), d != u, d != v
    for d in range(n):
        if d == u or d == v:
            continue
        if not ((adj_bits[v] >> d) & 1):
            continue
        # Check if simple path u -> a -> b -> c -> d avoiding v
        if has_path_len4(u, d, v):
            return True
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def build_adj_bits(edges):
        adj_bits = [0] * n
        edge_set = set()
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edge_set.add((u, v))
            edge_set.add((v, u))
        return adj_bits, edge_set
    
    def greedy_add(order, initial_edges):
        adj_bits, edge_set = build_adj_bits(initial_edges)
        added = []
        for u, v in order:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if not _has_c6(u, v, adj_bits, n):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                added.append((u, v))
        return initial_edges + added, adj_bits, edge_set
    
    def local_search_add(edges, adj_bits, edge_set, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for u, v in all_pairs:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if not _has_c6(u, v, adj_bits, n):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edge_set.add((u, v))
                    edge_set.add((v, u))
                    edges.append((u, v))
                    improved = True
                    if time.time() > time_limit:
                        break
        return edges, adj_bits, edge_set
    
    # Initial seed: 8 disjoint K5s
    seed_edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                seed_edges.append((5 * k + a, 5 * k + b))
    
    best_edges = seed_edges[:]
    best_count = len(best_edges)
    
    while time.time() < deadline:
        order = all_pairs[:]
        random.shuffle(order)
        
        # From seed
        cur_edges, adj_bits, edge_set = greedy_add(order, seed_edges)
        cur_edges, adj_bits, edge_set = local_search_add(cur_edges, adj_bits, edge_set, deadline - 10)
        
        if len(cur_edges) > best_count:
            best_count = len(cur_edges)
            best_edges = cur_edges[:]
        
        # From empty
        cur_empty, adj_bits_e, edge_set_e = greedy_add(order, [])
        cur_empty, adj_bits_e, edge_set_e = local_search_add(cur_empty, adj_bits_e, edge_set_e, deadline - 10)
        
        if len(cur_empty) > best_count:
            best_count = len(cur_empty)
            best_edges = cur_empty[:]
        
        if time.time() > deadline - 30:
            break
    
    # Final polish
    if time.time() < deadline - 5:
        adj_bits_b, edge_set_b = build_adj_bits(best_edges)
        best_edges, _, _ = local_search_add(best_edges, adj_bits_b, edge_set_b, deadline - 1)
    
    return [[u, v] for u, v in best_edges]
