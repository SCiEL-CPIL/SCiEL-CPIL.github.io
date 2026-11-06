import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds

    def popcount(x):
        return bin(x).count('1')

    def has_independent_set_of_size_k(n, adj, k):
        """
        Returns True if there is an independent set of size k.
        adj: list of bitmasks.
        """
        def dfs(cands, size):
            if size == k:
                return True
            if not cands:
                return False
            if size + popcount(cands) < k:
                return False
            
            # Heuristic: pick a vertex with high degree in the induced subgraph
            # to reduce candidates faster when excluding?
            # Or low degree to include?
            # Let's pick the first available vertex for simplicity, or one with max neighbors in cands.
            # Max neighbors in cands:
            # v = argmax_{u in cands} popcount(adj[u] & cands)
            # This is expensive. Let's just pick the lowest set bit.
            v = (cands & -cands).bit_length() - 1
            
            # Branch 1: Include v
            # New cands: cands & ~adj[v] & ~(1 << v)
            new_cands_incl = cands & ~adj[v] & ~(1 << v)
            if dfs(new_cands_incl, size + 1):
                return True
            
            # Branch 2: Exclude v
            new_cands_excl = cands & ~(1 << v)
            if dfs(new_cands_excl, size):
                return True
            
            return False

        return dfs((1 << n) - 1, 0)

    def is_valid(n, adj):
        if time.time() - start_time > DEADLINE - 5:
            return False
        # Check triangle-free
        for i in range(n):
            # Neighbors of i
            neighbors = adj[i]
            # For each pair of neighbors, check if they are connected
            # This is O(sum deg^2).
            # Better: for each j in neighbors, check if adj[j] & neighbors has any bits other than i?
            # Actually, if there is a triangle i-j-k, then j and k are neighbors.
            # So for each i, if any two neighbors are connected, triangle exists.
            # We can check: for each j in neighbors:
            # if adj[j] & neighbors:
            #    # Check if the intersection contains any vertex other than i?
            #    # adj[j] & neighbors includes i? No, adj[j] has i. neighbors has i? No, neighbors is mask of N(i).
            #    # So adj[j] & neighbors contains k if k is neighbor of j and k is neighbor of i.
            #    # Since graph is simple, i is not in N(i).
            #    # So if (adj[j] & neighbors) != 0, there is a k != i such that k in N(i) and k in N(j).
            #    # This forms triangle i-j-k.
            pass
        
        # Efficient triangle check:
        for i in range(n):
            ni = adj[i]
            # Iterate through bits in ni
            x = ni
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                # Check if any neighbor of j is also a neighbor of i
                # adj[j] & ni
                if adj[j] & ni:
                    return False
                x ^= lsb
        return True

    def try_construct(n):
        # Initialize with a random bipartite graph or just empty
        # Let's start with a cycle C_n to ensure connectivity and some structure
        adj = [0] * n
        for i in range(n):
            adj[i] |= 1 << ((i + 1) % n)
            adj[i] |= 1 << ((i - 1) % n)
        
        # Randomly add edges
        # Generate all possible non-edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Check if adding (u, v) creates a triangle
            # Triangle if N(u) & N(v) != 0
            if (adj[u] & adj[v]) == 0:
                # Add edge
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                
                # Periodically check validity
                # Checking every 100 edges or so
                # We can check alpha < 10
                # If we find an IS of size 10, we should remove some edges?
                # Or just restart?
                # For now, just check if it's valid.
                # If valid, return.
                # If not valid (has IS of size 10), we can't just continue adding edges
                # because adding edges only *decreases* alpha.
                # So if alpha >= 10, adding more edges might bring it down.
                # So we should keep adding edges.
                
                # Check if alpha < 10
                # This is expensive. Do it every 50 edges.
                # We need a counter.
                pass
        
        # After loop, check final validity
        if is_valid(n, adj):
            if not has_independent_set_of_size_k(n, adj, 10):
                return adj
        return None

    # Try n from 40 down to 39
    for n in [40, 39]:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Multiple restarts
        for attempt in range(50):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            adj = [0] * n
            # Start with random bipartite? Or empty?
            # Empty graph has alpha = n.
            # We want to add edges to reduce alpha.
            # Start with a random triangle-free graph.
            # Let's start with a cycle.
            for i in range(n):
                adj[i] |= 1 << ((i + 1) % n)
                adj[i] |= 1 << ((i - 1) % n)
            
            # Generate candidate edges
            candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
            random.shuffle(candidates)
            
            edges_added = 0
            for u, v in candidates:
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges_added += 1
                    
                    if edges_added % 20 == 0:
                        # Check if we already have alpha < 10?
                        # If we do, we can stop adding and return.
                        # But we need to ensure no triangles (guaranteed by check).
                        # And alpha < 10.
                        if not has_independent_set_of_size_k(n, adj, 10):
                            # Valid!
                            edges_list = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges_list.append([i, j])
                            return {'n': n, 'edges': edges_list}
            
            # If we finished candidates without finding valid, try next restart
            # But maybe we should check if the final graph is valid?
            # If alpha is still >= 10, it's invalid.
            # We can't do more with this restart.

    # Fallback to n=26 circulant
    n1 = 26
    edges1 = []
    for i in range(n1):
        for j in range(i + 1, n1):
            diff = (j - i) % n1
            if diff % 3 == 1:
                edges1.append([i, j])
    return {'n': n1, 'edges': edges1}
