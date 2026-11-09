import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    n = 20
    # Precompute all triples of distinct vertices
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    def is_valid(adj):
        """
        adj: list of n integers (bitmasks)
        Returns True if graph is K3,3-free
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            # Count bits
            if common & (common - 1) and (common & (common - 1)) & (common & (common - 1)) - 1:
                 # This is a quick check for >= 3 bits? 
                 # Let's just use bin().count('1') or a lookup table for 20 bits
                 pass
            # Faster:
            # if common has >= 3 bits set
            if common != 0:
                # Remove lowest set bit
                c = common
                c &= c - 1
                if c != 0:
                    c &= c - 1
                    if c != 0:
                        return False
        return True

    def edge_count(adj):
        # Sum of bits in all adj[i] / 2
        total = 0
        for mask in adj:
            total += bin(mask).count('1')
        return total // 2

    best_edges = 0
    best_adj = None
    
    # Multiple restarts
    while time.time() < deadline:
        # Initialize random graph
        # Start with a moderately dense graph to have more edges to play with
        # p = 0.5 is good
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove existing K3,3s
        # Loop until valid
        # To speed up, find a violating triple and remove a random edge from it
        valid = False
        while not valid:
            if time.time() > deadline:
                break
            valid = is_valid(adj)
            if valid:
                break
            # Find a violating triple
            # We can just re-scan to find one
            # Or modify is_valid to return the triple
            # For simplicity, let's just remove a random edge and re-check
            # This is slow. Better to find the specific conflict.
            
            # Let's optimize: find first conflict
            conflict = None
            for i, j, k in triples:
                common = adj[i] & adj[j] & adj[k]
                if common != 0:
                    c = common
                    c &= c - 1
                    if c != 0:
                        c &= c - 1
                        if c != 0:
                            conflict = (i, j, k, common)
                            break
            if conflict:
                # Remove a random edge incident to one of the vertices in the triple
                # or one of the common neighbors?
                # The K3,3 involves {i,j,k} and 3 neighbors in 'common'.
                # Removing any edge from the bipartite graph between {i,j,k} and common breaks it.
                # Edges are (i, c), (j, c), (k, c) for c in common.
                # Let's pick a random c in common and remove (i, c)
                # Actually, removing (i, c) only breaks K3,3s that include i and c.
                # It's safer to remove an edge between the two parts.
                # Parts: A={i,j,k}, B=subset of common.
                # Any edge (a, b) with a in A, b in B is part of the K3,3.
                # Let's pick a random a in A and b in B (from common)
                a = random.choice([i, j, k])
                # Pick a bit from common
                bits = []
                c = common
                while c:
                    lsb = c & -c
                    bits.append(lsb.bit_length() - 1)
                    c ^= lsb
                b = random.choice(bits)
                # Remove edge (a, b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
        
        if not valid:
            continue
            
        current_edges = edge_count(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
        # Local search: try to add edges
        # Shuffle edge list
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if adj[u] & (1 << v):
                    continue
                # Try adding (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_valid(adj):
                    improved = True
                    # Keep the edge
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
            if improved:
                # Restart the pass to catch new opportunities
                random.shuffle(all_edges)
                
        # Update best
        current_edges = edge_count(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    # Convert best_adj to list of edges
    if best_adj is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges
