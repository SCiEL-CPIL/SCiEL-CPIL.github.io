import random
import time
import sys

def construct():
    n = 40
    deadline = time.time() + 140.0  # Leave 10s margin
    
    # Bitset representation
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge
    adj = [0] * n
    
    def get_neighbors(i):
        """Return list of neighbors of i"""
        bits = adj[i]
        res = []
        while bits:
            lsb = bits & -bits
            j = lsb.bit_length() - 1
            res.append(j)
            bits ^= lsb
        return res

    def has_c6_with_edge(u, v):
        """
        Check if adding edge (u, v) creates a C6.
        This is equivalent to checking if there is a simple path of length 4 between u and v.
        Path: u - a - b - c - d - v
        """
        # We need to find a-b-c-d such that:
        # (u, a) in E, (a, b) in E, (b, c) in E, (c, d) in E, (d, v) in E
        # And all vertices distinct.
        
        # Optimization: 
        # N(u) is set of a's.
        # N(v) is set of d's.
        # We need path a-b-c-d where a in N(u), d in N(v).
        # This is a path of length 3 between N(u) and N(v).
        
        # Let's iterate over a in N(u)
        for a in get_neighbors(u):
            if a == v: continue # shouldn't happen if u,v not connected
            for b in get_neighbors(a):
                if b == u or b == v: continue
                for c in get_neighbors(b):
                    if c == u or c == v or c == a: continue
                    # Check if any neighbor of c is in N(v)
                    # And distinct from u, v, a, b
                    neighbors_c = get_neighbors(c)
                    for d in neighbors_c:
                        if d == u or d == v or d == a or d == b: continue
                        if adj[d] & (1 << v): # d is connected to v
                            return True
        return False

    def is_valid_edge(u, v):
        if u == v: return False
        if adj[u] & (1 << v): return False # already exists
        return not has_c6_with_edge(u, v)

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges():
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    # Strategy:
    # 1. Start with a random graph
    # 2. Remove all edges that are part of a C6 (or just check validity and remove invalid ones)
    # 3. Greedy add
    # 4. Local search swaps

    best_edges = 0
    best_adj = None

    for restart in range(20): # Limit restarts
        if time.time() > deadline: break
        
        random.seed(restart)
        # Initialize with random graph
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.2:
                    add_edge(i, j)
        
        # Clean up: remove edges that create C6
        # We iterate over all edges and check if removing them is necessary?
        # No, we check if the current graph has C6.
        # It's easier to just try to add edges and if we start with random, it might have C6.
        # Let's start with empty graph and add greedily.
        # Reset adj
        adj = [0] * n
        
        # Greedy add
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline: break
            if is_valid_edge(u, v):
                add_edge(u, v)
        
        current_count = count_edges()
        if current_count > best_edges:
            best_edges = current_count
            best_adj = adj[:]
        
        # Local Search
        # Try to swap edges to improve
        # We don't really need swaps if we just want max edges, but local optima exist.
        # However, simple greedy with random order often works well for small n.
        # Let's do a few passes of "try to add any valid edge".
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if time.time() > deadline: break
                    if not (adj[u] & (1 << v)):
                        if is_valid_edge(u, v):
                            add_edge(u, v)
                            improved = True
        
        current_count = count_edges()
        if current_count > best_edges:
            best_edges = current_count
            best_adj = adj[:]

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        bits = best_adj[i]
        while bits:
            lsb = bits & -bits
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            bits ^= lsb
            
    return edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(len(edges))
