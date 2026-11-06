import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Initial graph: 8 disjoint K4s
    n = 32
    # Build adjacency matrix as list of sets or bitmasks
    # Bitmasks are efficient for n <= 64
    adj = [0] * n
    # K4 blocks
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    best_n = n
    best_adj = adj[:]
    
    # Helper: Check if graph has C5
    def has_c5(adj, n):
        # For n <= 35, we can check all 5-tuples? 35^5 is too big.
        # But we can check cycles.
        # A faster way:
        # For each edge (u, v), look for path of length 3 between u and v.
        # If u-v-w-x-y-u, then w-x-y is path from v to u?
        # Let's use a standard C5 detection.
        # Since we are adding vertices one by one, we can optimize.
        # But for the initial check or general check:
        
        # Simple check:
        # For every pair of non-adjacent vertices (u, v) that are distance 2?
        # Actually, C5 is an induced subgraph? No, subgraph.
        # If G contains C5, it contains a cycle of length 5.
        
        # Optimized for small n:
        # Iterate over all 5-cycles.
        # Number of 5-cycles is small in sparse graphs, but our graph is dense in blocks.
        
        # Let's use a bitmask approach for path finding.
        # For each edge (u, v), we want to know if there is a path of length 3 from v to u.
        # Path v-a-b-u.
        # a != u, b != v, a != b.
        # Also u-v is an edge.
        # If such path exists, and {u,v,a,b} are distinct, we have a C5?
        # No, v-a-b-u-v is a 4-cycle.
        # We need v-a-b-c-u-v.
        # So for edge (u, v), we need path of length 4 from v to u?
        # v-a-b-c-u.
        # If u-v is edge, then v-a-b-c-u-v is C5.
        
        # So: For each edge (u, v), check if there is a path of length 4 from v to u.
        # This is equivalent to checking if u is in the 4-neighborhood of v.
        
        # Let's compute 4-step reachability.
        # n=35, so 35x35 matrix.
        # M = adj matrix.
        # M2 = M^2. M4 = M2^2.
        # If M4[v][u] > 0 and M[v][u] == 1, and the path is simple?
        # Bitmasks don't guarantee simple paths easily.
        
        # Given the small size, let's just do a DFS/BFS for each edge?
        # Or, since we start from 32 and add 1, we only check cycles involving the new node.
        
        pass

    # We will implement a specific check for the added vertex.
    
    def check_c5_with_new_vertex(adj, n, new_v):
        # Check all cycles of length 5 involving new_v.
        # new_v - a - b - c - d - new_v
        # a,b,c,d are from 0..n-2.
        # Edges: (new_v, a), (a, b), (b, c), (c, d), (d, new_v).
        # a, b, c, d distinct.
        
        neighbors = []
        for i in range(n - 1):
            if adj[new_v] & (1 << i):
                neighbors.append(i)
        
        # We need to pick a, d from neighbors.
        # And b, c from the rest such that a-b, b-c, c-d exist.
        
        # To be fast, iterate a, d.
        for a in neighbors:
            for d in neighbors:
                if a == d: continue
                # Find b adjacent to a
                # Find c adjacent to d and b
                # And c != a, c != d, c != new_v
                # And b != a, b != d, b != c, b != new_v
                
                # Iterate b
                b_mask = adj[a] & ~(1 << new_v) & ~(1 << a)
                while b_mask:
                    b = (b_mask & -b_mask).bit_length() - 1
                    b_mask &= b_mask - 1
                    
                    # b must be distinct from d
                    if b == d: continue
                    
                    # c must be neighbor of b and d
                    c_mask = adj[b] & adj[d]
                    c_mask &= ~(1 << new_v)
                    c_mask &= ~(1 << a)
                    c_mask &= ~(1 << b)
                    c_mask &= ~(1 << d)
                    
                    if c_mask:
                        return True
        return False

    def check_alpha(adj, n):
        # Check if there is an independent set of size 9.
        # This is NP-hard, but n is small.
        # We can use a greedy or backtracking.
        # Given we start from valid graph, and add 1 vertex,
        # we just need to check if the new vertex creates an independent set of size 9.
        # An independent set of size 9 either contains new_v or not.
        # If not, it was already there (so we assume previous graph was valid).
        # So we only need to check if there is an independent set of size 8 in G \ {new_v}
        # that is non-adjacent to new_v.
        
        non_neighbors = []
        for i in range(n - 1):
            if not (adj[new_v] & (1 << i)):
                non_neighbors.append(i)
        
        # Find independent set of size 8 in G[non_neighbors]
        # G[non_neighbors] is a subgraph of the previous graph.
        # We can use a recursive search.
        
        def find_independent_set(vertices, k, adj, n):
            if k == 0: return True
            if len(vertices) < k: return False
            if k == 1: return True
            
            # Pick a vertex v from vertices
            # Branch: include v or exclude v
            
            # Optimized:
            # Sort vertices by degree in the subgraph?
            
            v = vertices[0]
            rest = vertices[1:]
            
            # Include v:
            # Remove v and its neighbors from rest
            new_rest = []
            for u in rest:
                if not (adj[v] & (1 << u)):
                    new_rest.append(u)
            if find_independent_set(new_rest, k - 1, adj, n):
                return True
            
            # Exclude v:
            if find_independent_set(rest, k, adj, n):
                return True
            
            return False

        return find_independent_set(non_neighbors, 8, adj, n)

    # Try to add vertices
    current_n = 32
    current_adj = best_adj
    
    for target_n in range(33, 36):
        if time.time() > deadline:
            break
            
        new_v = current_n
        # Try different connection patterns for new_v
        # We want to maximize edges to reduce alpha, but avoid C5.
        
        # Heuristic: Connect to as many as possible.
        # Start with full connection, remove edges if C5 detected.
        
        # Actually, let's try to find ANY valid connection.
        # Randomize?
        
        # Try all subsets of neighbors? 2^32 is too big.
        # Try specific patterns.
        
        # Pattern 1: Connect to 1 vertex in each K4?
        # Pattern 2: Connect to 2 vertices in each K4?
        
        # Let's just try a random search for the new vertex's adjacency.
        
        found = False
        for _ in range(100): # 100 attempts
            if time.time() > deadline: break
            
            # Random adjacency
            new_adj = current_adj[:] + [0]
            for i in range(current_n):
                if random.random() < 0.5:
                    new_adj[new_v] |= (1 << i)
                    new_adj[i] |= (1 << new_v)
            
            # Check C5
            if check_c5_with_new_vertex(new_adj, current_n + 1, new_v):
                continue
                
            # Check Alpha
            if check_alpha(new_adj, current_n + 1):
                continue
                
            # Valid!
            current_n = target_n
            current_adj = new_adj
            best_n = current_n
            best_adj = current_adj[:]
            found = True
            break
            
        if not found:
            # Maybe try deterministic patterns?
            # If random fails, it might be impossible.
            # Break out if we can't grow.
            pass

    # Return the best graph found
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
