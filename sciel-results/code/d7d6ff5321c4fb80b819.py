import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Helper: Check if adding edge (u, v) creates a C4
    # A C4 is created if u and v have a common neighbor w such that w is connected to some x, 
    # and x is connected to u or v? 
    # Actually, C4-free condition: No two vertices have more than 1 common neighbor.
    # So, adding edge (u,v) is safe iff |N(u) ∩ N(v)| <= 1.
    # Wait, if we add (u,v), the new common neighbors of u and v are N(u) ∩ N(v).
    # If |N(u) ∩ N(v)| >= 2, say w1, w2, then u-w1-v-w2-u is a C4? 
    # No, u-w1-v and u-w2-v form a C4 u-w1-v-w2-u only if w1-w2 is an edge? 
    # No, a C4 is a cycle of length 4. u-w1-v-w2-u requires edges (u,w1), (w1,v), (v,w2), (w2,u).
    # If u and v are NOT connected, and they share w1, w2, then u-w1-v-w2-u is a C4.
    # So the condition for a graph to be C4-free is: for any non-adjacent u,v, |N(u) ∩ N(v)| <= 1.
    # For adjacent u,v, they share no "common neighbor" in the sense of forming a C4 with each other?
    # Actually, if u~v, and they share w, then u-w-v is a path of length 2. 
    # A C4 involving u,v would be u-w-x-v-u? No, u and v are adjacent.
    # The standard definition: G is C4-free iff for any distinct u,v, |N(u) ∩ N(v)| <= 1.
    # Let's verify. If |N(u) ∩ N(v)| >= 2, let w1, w2 be two common neighbors.
    # If u~v, then u-w1-v-w2-u is a 4-cycle? Edges: (u,w1), (w1,v), (v,w2), (w2,u). Yes.
    # If !u~v, then u-w1-v-w2-u is a 4-cycle? Edges: (u,w1), (w1,v), (v,w2), (w2,u). Yes.
    # So the condition is simply: For ALL pairs u,v, |N(u) ∩ N(v)| <= 1.
    
    # Helper: Check independence number
    # We need alpha(G) < 11.
    
    # State representation:
    # n: current number of vertices
    # adj: list of sets or bitmasks for neighbors. Bitmasks are faster for intersection.
    
    # We will try to grow n from a small base.
    # Start with n=39 (best known) or try to find a valid graph at n=40, 41, 42, 43.
    # Strategy: 
    # 1. Try to construct a valid graph for n=43 down to 39.
    # 2. Use a greedy + local search approach.
    
    # Bitmask representation for neighbors.
    # n <= 43, so we can use Python integers (arbitrary precision) as bitmasks.
    
    def is_c4_free(adj, n):
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if common & (common - 1): # has more than 1 bit
                    return False
        return True

    def max_independent_set_size(adj, n):
        # Exact search for max independent set using bitmask DP or branch and bound.
        # For n <= 43, a simple recursive branch and bound with bitmasks is fast enough if optimized.
        # Or use a simple greedy + local search to estimate, but we need exact for validity.
        # Actually, we just need to check if alpha < 11.
        # We can use a branch and bound to see if an independent set of size 11 exists.
        
        def has_independent_set_of_size_k(adj, n, k):
            # Returns True if there is an independent set of size k.
            # State: available vertices (bitmask), current size
            # We can prune if current_size + popcount(available) < k
            
            best = 0
            # Simple recursive search
            # To speed up, order vertices by degree?
            # Let's use a standard backtracking.
            
            # Optimization: If popcount(available) < k - current_size, return False.
            
            def backtrack(avail, size):
                if size == k:
                    return True
                if avail == 0:
                    return False
                if size + bin(avail).count('1') < k:
                    return False
                
                # Pick a vertex v from avail
                # Heuristic: pick vertex with highest degree in induced subgraph? Or lowest?
                # For independent set, picking a vertex with high degree removes more options.
                # Let's just pick the lowest set bit for simplicity, or random?
                # Randomization might help escape local minima in the "search for independent set" if we were optimizing, 
                # but here we are just checking existence.
                # Let's pick the first available vertex.
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                
                # Option 1: Include v
                # If we include v, we cannot include its neighbors.
                new_avail = avail & ~adj[v] & ~v_bit
                if backtrack(new_avail, size + 1):
                    return True
                
                # Option 2: Exclude v
                new_avail = avail & ~v_bit
                if backtrack(new_avail, size):
                    return True
                
                return False
            
            return backtrack((1 << n) - 1, 0)
        
        return has_independent_set_of_size_k(adj, n, 11)

    def verify_graph(adj, n):
        if not is_c4_free(adj, n):
            return False
        if max_independent_set_size(adj, n):
            return False
        return True

    # Main Strategy:
    # 1. Start with a known valid graph if possible, or random.
    # 2. Try to add vertices one by one.
    # 3. When adding a vertex, try all possible connections to existing vertices that maintain C4-free.
    # 4. If stuck, backtrack or restart with different random choices.
    
    # Since n=39 is known, let's try to build up to 43.
    # However, constructing from scratch via random addition is hard.
    # Let's try a different approach: 
    # Start with n=43.
    # Randomly initialize edges to satisfy C4-free.
    # This is hard.
    
    # Alternative: 
    # Use the polarity graph of order 5 (n=31) as a seed? 
    # Or start with disjoint triangles (n=30) and add vertices.
    
    # Let's try a "Grow and Repair" strategy.
    # 1. Start with n=39 (or lower) valid graph.
    # 2. Try to add vertex n.
    # 3. Connect it to a subset of existing vertices such that C4-free is maintained.
    # 4. Check independence number.
    
    # To get a starting point, let's use the disjoint triangles (n=30) which is fast to verify.
    # Then try to add vertices 30, 31, ... up to 43.
    
    def add_vertex(adj, n, new_idx):
        # Try to add new vertex with random connections
        # To maintain C4-free:
        # For any two existing neighbors u, v of new_idx, they must not share any common neighbor other than new_idx?
        # No, the condition is global.
        # If we connect new_idx to u and v, then u and v now share new_idx as a common neighbor.
        # They must not share any OTHER common neighbor.
        # So, for any pair (u, v) in N(new_idx), |N(u) ∩ N(v) \ {new_idx}| must be 0.
        # Since new_idx is new, it's not in N(u) or N(v) yet.
        # So we require |N(u) ∩ N(v)| == 0 for all u, v in N(new_idx).
        # Wait, if u and v are already connected, they share no common neighbors in a C4-free graph?
        # No, adjacent vertices can share common neighbors? 
        # If u~v and they share w, then u-w-v-u is a triangle, not C4.
        # But if they share w1, w2, then u-w1-v-w2-u is a C4.
        # So in a C4-free graph, ANY two vertices (adjacent or not) can share at most 1 common neighbor.
        
        # So, if we add new_idx connected to S = {u1, u2, ...}, then for any ui, uj in S,
        # ui and uj now share new_idx.
        # They must not share any other common neighbor.
        # So we must ensure that for all ui, uj in S, |N(ui) ∩ N(uj)| == 0.
        
        # This is a strong constraint. It means the neighbors of new_idx must form an independent set in the "common neighbor graph"?
        # No, it means the subgraph induced by S must have no edges between vertices that share a neighbor?
        # Actually, if ui and uj are in S, and they share a neighbor w (w != new_idx), then C4 is formed.
        # So S must be a set of vertices such that no two vertices in S share a neighbor in the existing graph.
        
        # Let's try to find such a set S.
        # Greedy: Pick vertices one by one.
        
        candidates = list(range(n))
        random.shuffle(candidates)
        
        S = []
        for v in candidates:
            # Check if v can be added to S
            ok = True
            for u in S:
                if adj[u] & adj[v]: # They share a neighbor
                    ok = False
                    break
            if ok:
                S.append(v)
                
        # If S is too small, independence number might increase.
        # We want S to be large enough to keep alpha < 11.
        # Actually, adding a vertex with many neighbors reduces the independence number of the new graph?
        # No, the new vertex is not in any independent set if it has neighbors in that set.
        # If we connect new_idx to S, then any independent set containing new_idx cannot contain any vertex in S.
        
        # Let's try a few random sets S.
        for attempt in range(50):
            candidates = list(range(n))
            random.shuffle(candidates)
            S = []
            for v in candidates:
                ok = True
                for u in S:
                    if adj[u] & adj[v]:
                        ok = False
                        break
                if ok:
                    S.append(v)
            
            # Create new adj
            new_adj = adj + [0]
            new_adj[n] = 0
            for v in S:
                new_adj[n] |= (1 << v)
                new_adj[v] |= (1 << n)
            
            # Verify
            if verify_graph(new_adj, n + 1):
                return new_adj, True
            
        return adj, False

    # Start with n=30 (disjoint triangles)
    adj = [0] * 30
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    current_n = 30
    
    # Try to grow
    while current_n < 43:
        if time.time() > deadline - 10:
            break
            
        new_adj, success = add_vertex(adj, current_n, current_n)
        if success:
            adj = new_adj
            current_n += 1
        else:
            # Stuck. Try random restarts or different strategies?
            # For now, just break or try a few more times with different random seeds
            # We can try to remove some edges to make room?
            # Let's just break for simplicity in this first iteration.
            # Or try to add with a different heuristic.
            
            # Let's try one more time with a different random seed
            random.seed(current_n * 1000 + time.time_ns() % 10000)
            new_adj, success = add_vertex(adj, current_n, current_n)
            if success:
                adj = new_adj
                current_n += 1
            else:
                break

    # Convert adj to edges list
    edges = []
    for i in range(current_n):
        for j in range(i+1, current_n):
            if adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': current_n, 'edges': edges}

# Ensure function is accessible
if __name__ == "__main__":
    result = construct()
    print(result['n'])
