import time
import random

def construct():
    N = 40
    deadline = time.time() + 110.0
    
    def creates_c6_bitset(u, v, adj):
        # adj is a list of N integers, each representing the adjacency set.
        # Check if there is a path of length 5 from u to v.
        # visited is a bitmask.
        visited = 1 << u
        current_level = 1 << u
        
        for _ in range(5):
            next_level = 0
            # Union of neighbors of all nodes in current_level
            # We can iterate through set bits in current_level
            temp = current_level
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                node = lsb.bit_length() - 1
                # Neighbors of node
                next_level |= adj[node]
                temp -= lsb
            
            # Remove visited nodes
            next_level &= ~visited
            if next_level == 0:
                break
            
            visited |= next_level
            current_level = next_level
            
        # Check if v is in current_level (distance 5)
        return (current_level >> v) & 1

    best_edges = 0
    best_graph = []
    
    # Seed
    seed_edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                seed_edges.append([5 * k + a, 5 * k + b])
    best_edges = len(seed_edges)
    best_graph = seed_edges[:]

    while time.time() < deadline:
        adj = [0] * N
        edges = []
        
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if not creates_c6_bitset(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = edges[:]
            if best_edges >= 105:
                break

    return best_graph
