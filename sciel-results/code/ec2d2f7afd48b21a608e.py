import random
import time

def _has_c5(adj, n):
    """Check if graph has a 5-cycle. adj is list of sets."""
    # For each pair of non-adjacent vertices, check if they have two common neighbors
    # that are not adjacent to each other (forming a 5-cycle through them)
    # Actually, a 5-cycle v0-v1-v2-v3-v4-v0 exists iff there exist v0, v1 adjacent,
    # and v2, v3, v4 such that v1-v2-v3-v4-v0 is a path of length 3.
    # Simpler: for each edge (u,v), check if there's a path of length 3 from v back to u
    # that doesn't use u or v in the middle.
    
    # More efficient: count common neighbors. If u and v are non-adjacent,
    # and they have at least 2 common neighbors a, b where a and b are not adjacent,
    # then u-a-v-b-u is a 4-cycle, not 5.
    
    # Let me think again. A 5-cycle is v0-v1-v2-v3-v4-v0.
    # Equivalently, there exist two non-adjacent vertices u, v such that
    # there are two internally disjoint paths of length 3 between u and v.
    
    # Alternative: iterate over all pairs of edges (a,b) and (c,d) that share no vertices,
    # and check if b-c and d-a are edges. That's O(n^4) which is too slow.
    
    # Best approach: for each vertex v, consider its neighborhood N(v).
    # A 5-cycle through v exists iff there exist a, b in N(v) (a != b) such that
    # a and b are not adjacent, and there exists c not in N(v) and c != v such that
    # c is adjacent to both a and b.
    # Wait, that's a 5-cycle v-a-c-b-v? No, that's v-a-c-b-v which is 4 edges = 4-cycle.
    
    # Let me be precise. 5-cycle: v0-v1-v2-v3-v4-v0.
    # Fix v0. Then v1, v4 are neighbors of v0. v1-v2-v3-v4 is a path of length 3.
    # So: for each v0, for each pair of neighbors (v1, v4) of v0 with v1 != v4,
    # check if there's a path of length 3 from v1 to v4 not using v0.
    # Path of length 3: v1-v2-v3-v4 where v2 adjacent to v1, v3 adjacent to v2, v4 adjacent to v3.
    # And v2 != v0, v3 != v0, v2 != v4, v3 != v1.
    
    # For each v0:
    for v0 in range(n):
        nbrs0 = adj[v0]
        if len(nbrs0) < 2:
            continue
        # For each pair of neighbors v1, v4 of v0
        nbrs_list = list(nbrs0)
        for i in range(len(nbrs_list)):
            v1 = nbrs_list[i]
            for j in range(i + 1, len(nbrs_list)):
                v4 = nbrs_list[j]
                # Check if there's a path v1-v2-v3-v4 of length 3 not through v0
                # v2 must be neighbor of v1, v2 != v0, v2 != v4
                # v3 must be neighbor of v2 and neighbor of v4, v3 != v0, v3 != v1
                found = False
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v4:
                        continue
                    # v2 is a candidate for the second vertex
                    # Check if v2 and v4 have a common neighbor v3 (v3 != v0, v3 != v1)
                    common = adj[v2] & adj[v4]
                    # Remove v0 and v1 from common
                    common.discard(v0)
                    common.discard(v1)
                    if common:
                        found = True
                        break
                if found:
                    return True
    return False

def _alpha_ge_9(adj, n):
    """Check if independence number >= 9 using greedy + backtracking."""
    # Quick check: if there's a vertex with degree <= n-1-8 = n-9, it can be in an independent set
    # with 8 others. But this is complex. Let's use a simple branch and bound.
    
    # Greedy lower bound first
    remaining = set(range(n))
    independent = []
    while remaining:
        # Pick vertex with minimum degree in remaining induced subgraph
        min_deg = float('inf')
        min_v = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        independent.append(min_v)
        remaining.discard(min_v)
        remaining -= adj[min_v]
        if len(independent) >= 9:
            return True
    
    # If greedy didn't find 9, try a more systematic search
    # This is exponential in worst case, but for n<=35 and alpha < 9, it should be fast
    # if the graph is dense enough.
    
    # Branch and bound
    best = 0
    def backtrack(remaining_set, current_independent):
        nonlocal best
        if len(current_independent) >= 9:
            return True
        if not remaining_set:
            return False
        # Upper bound: current + len(remaining)
        if len(current_independent) + len(remaining_set) < 9:
            return False
        
        # Pick a vertex to branch on
        v = next(iter(remaining_set))
        
        # Branch 1: include v
        new_remaining = remaining_set - {v} - adj[v]
        if backtrack(new_remaining, current_independent + [v]):
            return True
        
        # Branch 2: exclude v
        new_remaining2 = remaining_set - {v}
        if backtrack(new_remaining2, current_independent):
            return True
        
        return False
    
    return backtrack(set(range(n)), [])

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with the known 32-vertex solution: 8 disjoint K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    # Try to find a 33-vertex C5-free graph with alpha <= 8
    # Strategy: start from 32-vertex graph and add a vertex with appropriate connections
    # or use random search
    
    random.seed(42)
    
    def try_add_vertex(edges_32, n_32=32):
        """Try to add a 33rd vertex to the 32-vertex graph."""
        # Build adjacency from edges
        adj = [set() for _ in range(n_32)]
        for u, v in edges_32:
            adj[u].add(v)
            adj[v].add(u)
        
        # The new vertex 32 needs to be connected such that:
        # 1. No C5 is formed
        # 2. No independent set of size 9 is formed
        
        # For no C5: the new vertex can connect to vertices, but we need to ensure
        # no 5-cycle uses the new vertex. A 5-cycle through vertex 32 would be
        # 32-a-b-c-d-32 where a,d are neighbors of 32 and a-b-c-d is a path of length 3.
        
        # For no independent set of size 9: if the new vertex is not connected to some vertex v,
        # then v and the new vertex can both be in an independent set. The original graph has
        # alpha = 8, so if we don't connect the new vertex to at least 9 vertices, we might
        # get an independent set of size 9.
        
        # Actually, the new vertex must be connected to at least n - 8 + 1 = 26 vertices
        # to ensure that any independent set containing the new vertex has at most 8 other vertices
        # from the original graph that are not adjacent to it.
        
        # Let's try different connection patterns
        # The new vertex should connect to a set S of vertices.
        # Condition 1 (no C5): For any two non-adjacent vertices a, d in S, there should be no path
        # of length 3 from a to d in the original graph.
        # Condition 2 (no alpha >= 9): |V \ S| <= 8, i.e., |S| >= 25.
        
        # Let's try S = all vertices except a set T of size 8.
        # The 8 disjoint K4s: each K4 is a clique.
        # If T picks one vertex from each K4, then V\S = T has 8 vertices, one from each K4.
        # These 8 vertices form an independent set in the original graph.
        # The new vertex is not connected to T, so T ∪ {32} is an independent set of size 9. BAD.
        
        # So T cannot be an independent set in the original graph.
        # We need T to have alpha(T) <= 7 in the original graph, so that T ∪ {32} has alpha <= 8.
        
        # Let's try T with 2 vertices from 4 different K4s and 0 from the other 4 K4s.
        # Then alpha(T) <= 4 (one from each of the 4 K4s that have 2 vertices in T).
        # Wait, if T has 2 vertices from K4_i, those 2 are adjacent (clique), so they can't both be in an independent set.
        # So alpha(T) <= 4 if T has at most 2 vertices from each of 4 K4s.
        
        # Let's try: T = {0, 1, 8, 9, 16, 17, 24, 25} (two from each of first 4 K4s)
        # Then S = V \ T, |S| = 24. But we need |S| >= 25. So this doesn't work.
        
        # Let's try |T| = 7. Then |S| = 25.
        # T should have alpha(T) <= 7, which is always true since |T| = 7.
        # But we also need to check C5 condition.
        
        # This is getting complex. Let's just try random connection patterns and check.
        
        for _ in range(1000):
            # Choose a random set S of size 25-32
            s_size = random.randint(25, 32)
            S = set(random.sample(range(n_32), s_size))
            
            # Check C5 condition: for any two non-adjacent a, d in S, no path of length 3 from a to d
            c5_ok = True
            S_list = list(S)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    d = S_list[j]
                    if d in adj[a]:
                        continue  # a and d are adjacent, no C5 through them
                    # Check if there's a path of length 3 from a to d
                    # a-b-c-d where b adjacent to a, c adjacent to b and d
                    found_path = False
                    for b in adj[a]:
                        if b in S and b != d:
                            # b is in S, but that's ok for the path
                            common = adj[b] & adj[d]
                            if common:
                                found_path = True
                                break
                    if found_path:
                        c5_ok = False
                        break
                if not c5_ok:
                    break
            
            if not c5_ok:
                continue
            
            # Check alpha condition: new vertex not connected to V\S
            # V\S has size n_32 - s_size
            # The new vertex can be in an independent set with any independent set of V\S
            # So we need alpha(V\S) <= 8
            # Since the original graph has alpha = 8, alpha(V\S) <= 8 is always true
            # Wait, that's not right. alpha(V\S) could be up to 8, and adding the new vertex
            # (which is not connected to V\S) gives an independent set of size 9.
            # So we need alpha(V\S) <= 7.
            
            # Check if V\S has an independent set of size 8
            vs = set(range(n_32)) - S
            if len(vs) < 8:
                # Can't have independent set of size 8
                pass
            else:
                # Check if vs has alpha >= 8
                # Quick check: if vs has 8 vertices that are pairwise non-adjacent
                vs_list = list(vs)
                alpha_vs = 0
                # Simple greedy
                remaining = set(vs_list)
                ind = []
                while remaining:
                    min_deg = float('inf')
                    min_v = None
                    for v in remaining:
                        deg = len(adj[v] & remaining)
                        if deg < min_deg:
                            min_deg = deg
                            min_v = v
                    ind.append(min_v)
                    remaining.discard(min_v)
                    remaining -= adj[min_v]
                    if len(ind) >= 8:
                        break
                if len(ind) >= 8:
                    continue
            
            # Both conditions satisfied, construct the new graph
            new_edges = list(edges_32)
            for v in S:
                new_edges.append([32, v])
            return new_edges
        
        return None
    
    # Try to extend the 32-vertex graph
    if time.time() < deadline - 5:
        new_edges = try_add_vertex(best_edges)
        if new_edges is not None:
            # Verify
            n_test = 33
            adj_test = [set() for _ in range(n_test)]
            for u, v in new_edges:
                adj_test[u].add(v)
                adj_test[v].add(u)
            if not _has_c5(adj_test, n_test) and not _alpha_ge_9(adj_test, n_test):
                best_n = 33
                best_edges = new_edges
    
    return {'n': best_n, 'edges': best_edges}
