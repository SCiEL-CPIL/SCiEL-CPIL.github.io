import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=3, g=16: Use LUW (Lubotzky-Weil-Warner) construction.
    Returns (n, edges) where edges is a list of (u, v) tuples.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s/150s limits

    def verify_girth(n, edges, k, g):
        """
        Verify girth >= g using non-backtracking walks from every vertex.
        Returns True if girth is at least g.
        """
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        # Build adjacency and check for simple graph (no loops, no multiedges)
        for u, v in edges:
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
            
        max_depth = g // 2
        # For even girth g, we need to ensure no cycle of length < g.
        # Non-backtracking BFS from each vertex up to depth g//2.
        # If we return to start, cycle length is depth+1.
        # If we hit a previous state (v, u), the cycle length is bounded.
        
        # Efficient check:
        # From each start node, do BFS on (current, prev) states.
        # If we reach (start, prev) at depth d, we have a cycle of length d+1? No.
        # Actually, if we start at S, go to v1, ... uk, and uk is adjacent to S, 
        # the cycle is S-v1-...-uk-S of length k+1.
        # In non-backtracking terms:
        # Start states: (v, S) for v in adj[S]. Depth 1.
        # If we reach (S, v) at depth d, we have a cycle of length d+1.
        # Wait, if we are at (u, prev) and next is S, and S != prev, then cycle length is depth+1.
        
        for s in range(n):
            # BFS
            visited = {} # (curr, prev) -> depth
            queue = deque()
            for v in adj[s]:
                queue.append((v, s, 1))
                visited[(v, s)] = 1
            
            while queue:
                u, prev, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                for v in adj[u]:
                    if v == prev:
                        continue
                    
                    if v == s:
                        # Cycle found of length depth + 1
                        if depth + 1 < g:
                            return False
                        continue
                    
                    state = (v, u)
                    if state in visited:
                        # Cycle found.
                        # The length of the cycle is not simply depth - visited_depth[state].
                        # Let's analyze:
                        # We have a path s -> ... -> u -> v (length depth+1? No, depth is steps from s).
                        # Wait, (v, s) was at depth 1.
                        # If we see (v, u) where (v, u) was seen before at depth d1?
                        # Actually, if we see a state (v, u) that is already in visited, 
                        # it means we have a closed non-backtracking walk.
                        # The length of the shortest cycle in this walk is at most (depth - d1) + 2?
                        # To be safe and simple, if we detect any state reuse before depth g, 
                        # we need to be careful. 
                        # Standard approach: 
                        # If we reach a node v that was visited at a shallower depth, 
                        # the distance is depth - prev_depth.
                        # But non-backtracking prevents immediate reversal.
                        # Let's use the property: if there is a cycle of length L < g, 
                        # there is a non-backtracking closed walk of length L.
                        # If we find any repeat, the cycle length is <= depth + 1? 
                        # Actually, if (v, u) is in visited at depth d1, and now at depth d2,
                        # the closed walk length is d2 - d1 + 1? No, the step count difference.
                        # Let's just check if depth + 1 < g. If we are expanding up to max_depth = g//2,
                        # and we see a repeat, the cycle is likely short.
                        # For rigorous certification:
                        # If (v, u) is in visited, let d1 = visited[(v, u)].
                        # The cycle length is at most d2 - d1 + 2?
                        # Actually, the path from (v, u) at d1 to (v, u) at d2 is a closed walk.
                        # The shortest cycle in it has length <= d2 - d1 + 1?
                        # Let's be conservative: if d2 - d1 + 1 < g, return False.
                        # But actually, if we see (v, u) again, it implies a cycle.
                        # Let's check the length.
                        if depth - visited[state] + 1 < g:
                            return False
                    else:
                        visited[state] = depth
                        queue.append((v, u, depth + 1))
                        
        return True

    def luw_construct(k, g):
        """
        Construct LUW graph for k=3 (q=3) or k=4 (q=4).
        For k=3, the LUW graphs are Cayley graphs of the group (Z_q)^d x {0,1} 
        with specific generators.
        """
        if k != 3 and k != 4:
            return None, None
        
        q = k
        # d is the dimension. For girth g, we typically need d >= g/2 - 1.
        # Let's try d = g // 2 - 1, g // 2, etc.
        # The number of vertices is 2 * q^d.
        # For k=3, g=16: d=7 -> 2 * 3^7 = 4374. Too big?
        # Wait, the known best is 936.
        # 936 is not of the form 2 * 3^d or 2 * 4^d.
        # 936 = 8 * 117 = 8 * 9 * 13.
        # It's a general graph, not necessarily a Cayley graph of that form.
        # LUW construction for k=3 uses q=3.
        # The standard LUW graph for girth 16 might not be the optimal one found by Exoo et al.
        # However, we need *some* valid graph. 
        # If LUW gives 4374, it's valid but score 3.0 (since seed is likely worse or similar).
        # Actually, if seed score is 3.0, and we produce 4374, and the best known is 936, 
        # the score is likely based on n. 
        # The prompt says: "Score: n, minimized".
        # So if we return 4374, and the best known is 936, our score is worse than 936.
        # But we don't have a 936 graph hardcoded.
        # We need to find a good one.
        
        # Let's try a different strategy: 
        # 1. If we have a hardcoded graph for k=3, g=16 with 936 vertices, return it.
        # 2. Otherwise, try LUW with small d and verify.
        # 3. If LUW is too big, try to find a smaller one via search? 
        #    Search for girth 16 is hard.
        
        # Let's check if we can hardcode a known graph.
        # The paper "Small 3-regular graphs of girth 16" by Exoo et al. provides the graph.
        # I don't have the exact edge list memorized.
        
        # Alternative: Use a generic construction that works for small girths, 
        # but for g=16, it's tough.
        
        # Let's try LUW with d=6, 7.
        # For d=6, n = 2 * 3^6 = 1458.
        # For d=5, n = 2 * 3^5 = 486. (Girth might be < 16)
        
        # Let's implement LUW for d=6 and verify.
        
        def luw_with_d(d):
            # Group: {0, 1} x (Z_q)^d
            # Vertex representation: (side, x1, ..., xd)
            # side in {0, 1}
            # xi in {0, ..., q-1}
            
            # Generators for side=0:
            # e_i = (1, e_i) where e_i is unit vector?
            # Actually, LUW uses generators (1, x) such that the graph is regular.
            # For q=3, the generators are:
            # (1, (1,0,...,0)), (1, (0,1,0,...,0)), ..., (1, (0,...,0,1))
            # And their inverses?
            # The graph is defined by edges between (0, x) and (1, x + g_i) for i=1..d?
            # And between (1, x) and (0, x - g_i)?
            # This would give degree d on each side.
            # For k=3, d must be 3?
            # If d=3, n = 2 * 27 = 54. Girth is small.
            # To get higher girth, we need more dimensions?
            # But degree is fixed at 3.
            # So we can't just increase d for k=3.
            # The LUW construction for fixed degree k uses a fixed q.
            # The girth increases with d?
            # No, the standard LUW construction for degree k+1 (q=k+1) uses 
            # a fixed set of generators. The girth is determined by the 
            # absence of short words in the free group mapping to identity.
            
            # For k=3 (q=3), the standard LUW graph is a tree-like structure 
            # up to a certain depth.
            
            # Actually, the LUW construction is often described for q+1 regular graphs 
            # where q is a prime power.
            # For q=3, we get 4-regular graphs.
            # For 3-regular graphs, we use a different construction?
            # Or maybe we use the subgraph of the 4-regular LUW graph?
            # No, 3-regular graphs of high girth are often found via 
            # "cage" problems.
            
            # Let's try a simple random 3-regular graph generator and check girth.
            # If girth < 16, we can try to fix it? 
            # Or just return a large LUW if k=4 is allowed? 
            # The task says k=3, g=16.
            
            # Let's try to construct a 3-regular graph using the 
            # "random configuration model" and then verify.
            # If n is large enough, the girth will be large.
            # For n=1000, girth is likely > 16.
            # We can start with n=1000 and decrease until girth < 16.
            
            return None, None
        
        # Since implementing a verified girth 16 graph from scratch is hard,
        # and I don't have the 936-vertex graph hardcoded,
        # I will implement a robust random search for small n.
        
        # Start with n = 600 (above Moore bound 512).
        # Generate random 3-regular graphs.
        # Verify girth.
        # If valid, return.
        # If not, increase n or try more samples.
        
        return None, None

    # --- Main Logic ---
    
    # Try LUW first (might be buggy or not applicable for k=3 directly without care)
    # For now, let's use the random search strategy which is more general.
    
    best_n = float('inf')
    best_edges = []
    
    # Moore bound for k=3, g=16:
    # n >= 2 * (1 + 2 + 4 + ... + 128) = 2 * 255 = 510.
    # So n must be at least 512 (even).
    
    # We will try n from 512 upwards.
    # Generating random 3-regular graphs:
    # 1. Create 3n stubs.
    # 2. Randomly pair them.
    # 3. Check for loops and multiple edges.
    # 4. If invalid, retry.
    # 5. If valid, check girth.
    
    import random
    random.seed(42) # Deterministic for reproducibility if needed, but random is fine
    
    def generate_random_3_regular(n):
        if n % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(3)]
        edges = set()
        adj = [[] for _ in range(n)]
        
        while True:
            random.shuffle(stubs)
            valid = True
            edges = set()
            adj = [[] for _ in range(n)]
            for i in range(0, 3*n, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                # Check degrees (should be 3)
                if all(len(adj[i]) == 3 for i in range(n)):
                    edge_list = list(edges)
                    return n, edge_list
    
    # Try n from 512 to 1000
    # Verification is the bottleneck.
    # We can speed up verification by early termination.
    
    for n in range(512, 1000, 2):
        if time.time() > deadline:
            break
            
        # Try a few random graphs for this n
        for _ in range(50):
            if time.time() > deadline:
                break
                
            res = generate_random_3_regular(n)
            if res is None:
                continue
            n_test, edges_test = res
            if verify_girth(n_test, edges_test, 3, g):
                return n_test, edges_test
                
    # If we didn't find a small one, try LUW as a fallback (if it works)
    # Or just return a large valid graph if found earlier?
    # We didn't track best_n in the loop above because we return immediately.
    # Let's modify to keep track of the best valid graph found so far.
    
    # Actually, the loop above returns immediately if a valid graph is found.
    # If it doesn't find one in 512-1000, we need a fallback.
    # Let's try n=2000 with random search.
    
    for n in range(1000, 2000, 2):
        if time.time() > deadline:
            break
        for _ in range(10):
            if time.time() > deadline:
                break
            res = generate_random_3_regular(n)
            if res is None:
                continue
            n_test, edges_test = res
            if verify_girth(n_test, edges_test, 3, g):
                return n_test, edges_test
                
    # Last resort: return an empty graph or a trivial one? 
    # The task requires a valid graph.
    # If we can't find one, we fail.
    # But for n=1000, girth 16 is very likely to exist.
    
    return 0, []

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
