import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_c4_free(adj, n):
        # Check if graph is C4-free
        # Two vertices u, v have at most 1 common neighbor
        # Using bitsets for efficiency
        # adj is list of bitmasks
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):
                    return False
        return True

    def count_common_neighbors(adj, u, v):
        common = adj[u] & adj[v]
        return bin(common).count('1')

    def can_add_edge(adj, u, v, n):
        # Check if adding edge (u, v) creates a C4
        # A C4 is formed if there exists w, z such that
        # w is neighbor of u, z is neighbor of v, and w=z? No.
        # C4: u-w-v-z-u? No, u-v is the new edge.
        # A C4 involving new edge (u,v) requires a path of length 3 from u to v.
        # i.e., u-w-x-v where w, x are distinct and w != v, x != u.
        # Actually, if we add (u,v), a C4 exists if there is a vertex w
        # such that w is neighbor of u and w is neighbor of v? No, that's a triangle.
        # C4: u - a - b - v - u.
        # So we need a path u-a-b-v.
        # This means there exists a in N(u) and b in N(v) such that a-b is an edge?
        # No, the cycle is u-a-b-v-u.
        # So a must be neighbor of u, b must be neighbor of v, and a must be neighbor of b.
        # AND a != v, b != u, a != b.
        # Wait, the standard definition: No two vertices have more than 1 common neighbor.
        # Adding (u,v) creates a C4 if and only if there exists a vertex w
        # such that w is a common neighbor of u and v? No.
        # If u and v already have a common neighbor w, adding (u,v) creates a triangle, not necessarily C4.
        # But if u and v have TWO common neighbors w1, w2, then u-w1-v-w2-u is a C4?
        # No, u-w1-v and u-w2-v. The cycle is u-w1-v-w2-u.
        # Yes! If u and v have at least 2 common neighbors, adding the edge (u,v)
        # creates a C4 (u-w1-v-w2-u) IF the graph was C4-free before?
        # Wait. If the graph is C4-free, u and v have at most 1 common neighbor.
        # If they have 1 common neighbor w, adding (u,v) creates triangle u-w-v-u.
        # Does it create a C4? No.
        # If they have 0 common neighbors, adding (u,v) creates no C4 involving u-v as a chord?
        # Let's re-evaluate.
        # A C4 is a cycle of length 4.
        # If we add edge (u,v), any new C4 must use this edge.
        # So the C4 is u-v-x-y-u.
        # This requires v-x edge, x-y edge, y-u edge.
        # So x must be a neighbor of v, y must be a neighbor of u, and x-y must be an edge.
        # Also x != u, y != v, x != y.
        # So we need to check if there exists x in N(v) and y in N(u) such that (x,y) is an edge.
        # This is equivalent to: (N(v) & ~{u}) and (N(u) & ~{v}) have an edge between them?
        # More simply: Let A = N(u), B = N(v).
        # We need an edge (x,y) with x in B, y in A.
        # Note: if x=y, then x is a common neighbor. If x=y, the path is u-x-v-x-u? No, that's a triangle if u-v exists.
        # If x=y, the cycle is u-x-v-x-u? No, edges are u-v, v-x, x-x (no), x-u.
        # So x and y must be distinct.
        # So we check if there is an edge between set A and set B, excluding self-loops?
        # Actually, if there is a common neighbor w, and w is in A and B.
        # If there is another neighbor x in B and y in A such that x-y is edge.
        # Efficient check:
        # For each x in N(v):
        #   If (adj[x] & adj[u]) is not empty (and the bit is not v? v is not in N(u) usually unless edge exists)
        #   Actually, if x is in N(v), we check if any neighbor of x is in N(u).
        #   Let C = adj[x] & adj[u].
        #   If C is not 0, there is a y in N(u) connected to x.
        #   If y == v? v is not in N(u) (we are adding u-v).
        #   If y == x? x is not in N(x).
        #   So if C is not 0, we have a C4.
        #   Exception: if the only such y is v? No, v not in N(u).
        #   So just check if any x in N(v) has a neighbor in N(u).
        
        # Optimization:
        # Iterate over smaller of N(u) and N(v)?
        # We can just check: 
        # union_N = 0
        # for x in bits(N(v)):
        #    if adj[x] & adj[u]: return False
        # return True
        
        # Even faster:
        # Let S = N(v).
        # We want to know if any x in S has a neighbor in N(u).
        # This is equivalent to checking if (OR of adj[x] for x in S) & N(u) is non-zero?
        # No, that's not right. That checks if ANY x in S is connected to ANY y in N(u).
        # Yes, that is exactly what we want.
        # Let R = 0
        # for each bit x set in N(v): R |= adj[x]
        # if R & N(u): return False
        
        # However, doing this for every pair is O(n^3) in worst case if not careful.
        # Given n<=43, N(u) is small-ish.
        
        # Let's implement the simple loop for clarity and correctness first.
        # n is small.
        u_mask = adj[u]
        v_mask = adj[v]
        
        # We need x in v_mask, y in u_mask, such that adj[x] has bit y.
        # And x != y.
        
        # Iterate x in v_mask
        temp = v_mask
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            temp ^= lsb
            
            # Check if x has any neighbor in u_mask
            # If adj[x] & u_mask is not 0, we found a C4.
            # Note: if x is in u_mask (common neighbor), and adj[x] has another bit in u_mask?
            # If x is common neighbor, x in u_mask.
            # If adj[x] & u_mask has bit x? No, no self loops.
            # So if adj[x] & u_mask is non-zero, it contains some y != x.
            # This y is in u_mask (neighbor of u) and neighbor of x.
            # So u-y-x-v-u is a cycle?
            # Edges: u-y, y-x, x-v, v-u.
            # Yes, this is a C4 if u!=y, y!=x, x!=v, v!=u.
            # u!=v (new edge).
            # y!=x (no self loop).
            # y!=v? v is not in u_mask (u,v not connected yet).
            # x!=u? u is not in v_mask (u,v not connected yet).
            # So yes, any neighbor of x in u_mask creates a C4.
            
            if adj[x] & u_mask:
                return False
                
        return True

    def has_independent_set_of_size_k(adj, n, k):
        # Returns True if there is an independent set of size k
        # Use branch and bound / backtracking
        # Order vertices by degree (descending) to prune faster?
        # Or use greedy coloring bound?
        # For n=43, k=11, simple backtracking might be slow if not pruned.
        # We can use a heuristic: try to find a set of size k.
        
        # Sort vertices by degree descending
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Map original index to position in order
        pos_in_order = [0]*n
        for i, v in enumerate(order):
            pos_in_order[v] = i
            
        # Recursive search
        # State: current set of vertices (as bitmask), current size, next vertex to consider (index in order)
        # Pruning: if size + (remaining vertices) < k: return False
        # Also, if we can't add any more vertices from remaining to reach k?
        
        # To speed up, we can use a greedy upper bound on independence number of remaining subgraph?
        # For now, simple backtracking with good ordering.
        
        def dfs(idx, current_mask, size):
            if size >= k:
                return True
            if idx == n:
                return False
            
            # Pruning: even if we take all remaining, can we reach k?
            if size + (n - idx) < k:
                return False
            
            v = order[idx]
            
            # Option 1: Include v
            # Check if v is independent with current_mask
            if not (adj[v] & current_mask):
                if dfs(idx + 1, current_mask | (1 << v), size + 1):
                    return True
            
            # Option 2: Exclude v
            if dfs(idx + 1, current_mask, size):
                return True
                
            return False
            
        return dfs(0, 0, 0)

    def try_construct(n, time_limit):
        # Try to find a C4-free graph on n vertices with alpha < 11
        # Start with a random graph, make it C4-free, then try to reduce alpha by adding edges
        
        # Initialize random graph
        # Prob of edge ~ 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Step 1: Enforce C4-free
        # While not C4-free, remove an edge from a C4
        # To be fast, we can just remove random edges until C4-free?
        # Or detect C4s and remove one edge.
        
        # Simple loop:
        # Check C4-free. If not, find a C4 and remove an edge.
        # Finding a C4:
        # for u < v:
        #   common = adj[u] & adj[v]
        #   if common has >= 2 bits:
        #      pick two bits w, z from common
        #      remove edge (u, w) or (v, w) etc?
        #      Removing (u, w) breaks the C4 u-w-v-z-u?
        #      Yes.
        
        while True:
            if is_c4_free(adj, n):
                break
            # Find a violation
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    if common & (common - 1): # at least 2 bits
                        # Pick two common neighbors
                        c1 = common & -common
                        c2 = (common ^ c1) & -(common ^ c1)
                        w1 = c1.bit_length() - 1
                        w2 = c2.bit_length() - 1
                        # Remove edge (u, w1)
                        adj[u] &= ~(1 << w1)
                        adj[w1] &= ~(1 << u)
                        found = True
                        break
                if found:
                    break
            if not found:
                break # Should not happen if is_c4_free is accurate
                
        # Step 2: Check alpha
        if not has_independent_set_of_size_k(adj, n, 11):
            return adj, True
            
        # Step 3: Try to add edges to reduce alpha
        # We want to add edges that do not create C4s.
        # We prioritize adding edges between vertices that are in large independent sets?
        # Or just random non-edges that are C4-safe.
        
        # Generate list of potential edges (u, v) where not connected and can_add_edge
        # This list changes as we add edges.
        
        # Loop until time limit
        while time.time() < time_limit:
            # Find a non-edge that can be added
            added = False
            # Shuffle potential edges? Or just iterate
            # To be random, we can pick a random u, v
            for _ in range(100): # Try 100 random pairs per iteration
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if adj[u] & (1 << v): continue
                if can_add_edge(adj, u, v, n):
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if not added:
                # No more edges can be added without creating C4?
                # Or we just didn't find any?
                # If we can't add any, we are stuck.
                # Break? Or restart?
                break
                
            # Check alpha again
            if not has_independent_set_of_size_k(adj, n, 11):
                return adj, True
                
    # Try n from 43 down to 39
    for n in range(43, 38, -1):
        if time.time() > deadline:
            break
        # Try a few times for each n
        success = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            t_limit = min(deadline, time.time() + 5)
            adj, ok = try_construct(n, t_limit)
            if ok:
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
    # Fallback: Return a known valid small graph or seed
    # Seed was 39? Or 30?
    # The prompt says "best known construction has 39 vertices".
    # If we fail, we should return something valid.
    # Let's return a simple valid graph, e.g., n=10, no edges? No, alpha=10.
    # n=10, complete graph K10? Alpha=1. C4-free? K10 has many C4s.
    # We need a C4-free graph with alpha < 11.
    # Empty graph on 10 vertices: alpha=10. Valid.
    # Let's return n=10, edges=[]
    return {'n': 10, 'edges': []}

if __name__ == "__main__":
    result = construct()
    print(result['n'])
