import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    def is_valid_circulant(n, S):
        """
        Checks if the circulant graph on n vertices with connection set S 
        is K4-free and has independence number <= 5.
        S is a set of integers in [1, n//2].
        """
        if time.time() - start_time > time_limit:
            return None

        # Build adjacency bitmasks for efficiency
        adj = [0] * n
        for i in range(n):
            for s in S:
                j = (i + s) % n
                adj[i] |= (1 << j)
                k = (i - s) % n
                adj[i] |= (1 << k)

        # Check for K4
        # A K4 exists if there are 4 vertices a,b,c,d such that all pairs are edges.
        # In a circulant graph, if we fix vertex 0, a K4 containing 0 corresponds to
        # a set {s1, s2, s3} in S such that s2-s1, s3-s1, s3-s2 are also in S (modulo n).
        # More generally, we can just iterate.
        # Optimization: Iterate over all pairs (u, v) that are edges.
        # Then look for w in N(u) \cap N(v). If such w exists, check if N(u) \cap N(v) \cap N(w) is non-empty.
        
        # For n=35, n^2 is 1225. Pairs of neighbors can be many.
        # Let's use the property: K4 exists iff there is a 4-clique.
        # We can check for K4 by iterating over all triples? No, too slow.
        # Iterate over pairs (u, v) where u < v and u~v.
        # Common neighbors C = N(u) & N(v).
        # For each w in C, check if N(u) & N(v) & N(w) != 0.
        
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Remove u and v from common? 
                    # common contains neighbors of both u and v.
                    # u is not neighbor of u, v is not neighbor of v.
                    # But u is neighbor of v, so u is in N(v). v is in N(u).
                    # So u is in N(u) & N(v)? No, u not in N(u).
                    # So common does not contain u or v?
                    # adj[u] has bit v set. adj[v] has bit u set.
                    # common = adj[u] & adj[v].
                    # Bit u in common? adj[u] bit u is 0. So u not in common.
                    # Bit v in common? adj[v] bit v is 0. So v not in common.
                    # So common contains only vertices w != u, v.
                    
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c &= c - 1
                        # Check if there is an x in N(u) & N(v) & N(w)
                        # This x would form K4 with u, v, w.
                        # Note: x must be distinct from u, v, w.
                        # Since u,v not in N(u), N(v) respectively in a way that excludes them?
                        # u not in N(u). v not in N(v).
                        # w is in N(u) and N(v).
                        # If x = u, is u in N(w)? Yes, since w in N(u) implies u in N(w).
                        # But u is not in N(u). So u is not in N(u) & N(v) & N(w).
                        # Similarly v is not in the intersection.
                        # What about w? w is in N(w)? No.
                        # So any x in the intersection is a valid 4th vertex.
                        if (adj[u] & adj[v] & adj[w]):
                            return False
        
        # Check for Independent Set of size 6
        # Equivalent to K6 in the complement graph.
        # Complement adjacency: comp_adj[i] = ((1 << n) - 1) - (1 << i) - adj[i]
        # We need to find if there is a K6 in the complement.
        # This is a standard clique search. For n=35, we can use a backtracking search.
        
        # To speed up clique search, we can use a greedy coloring bound or simple pruning.
        # Given the small n, a simple recursive search with bitmasks is likely fast enough
        # if we order vertices by degree in complement.
        
        # Degrees in complement: n - 1 - degree_in_graph
        degs = [bin(adj[i]).count('1') for i in range(n)]
        comp_degs = [n - 1 - d for d in degs]
        
        # Order vertices by comp_degree descending
        order = sorted(range(n), key=lambda x: comp_degs[x], reverse=True)
        
        # Map original vertex to position in order? 
        # It's easier to just pass the list of candidates.
        
        def find_k6(candidates):
            if len(candidates) < 6:
                return False
            
            # Pruning: if we can't form a clique of size 6 from candidates, return False
            # Simple bound: if max degree in induced subgraph is too low?
            # Let's just backtrack.
            
            if not candidates:
                return False
            
            # Pick the first vertex
            v = candidates[0]
            rest = candidates[1:]
            
            # New candidates must be neighbors of v in the complement graph
            # i.e., non-neighbors of v in the original graph
            new_cands = []
            for u in rest:
                if not (adj[v] >> u) & 1:
                    new_cands.append(u)
            
            if find_k6(new_cands):
                return True
            
            # Option 2: Exclude v
            return find_k6(rest)

        # To make this faster, we can add a check: if len(candidates) < 6, return False.
        # Also, we can check if the subgraph induced by candidates can even contain a K6.
        # A simple necessary condition: the number of edges in the complement induced by candidates 
        # must be at least 6*5/2 = 15.
        # Calculating this every step might be expensive.
        
        # Let's try the simple backtrack first. With n=35, depth 6, branching factor might be high.
        # But we prune heavily: new_cands are neighbors in complement.
        
        # Optimization: Check if any vertex in candidates has comp_degree < 5? 
        # Actually, we are looking for K6. Every vertex in the K6 must have degree >= 5 within the K6.
        # So in the induced subgraph of candidates, every vertex in the potential K6 must have degree >= 5.
        # If a vertex has degree < 5 in the induced subgraph of candidates, it can't be part of a K6.
        
        def find_k6_opt(candidates):
            if len(candidates) < 6:
                return False
            
            # Filter candidates that have at least 5 neighbors in the complement within candidates
            # This is a strong prune.
            valid_cands = []
            for v in candidates:
                deg_in_cands = 0
                for u in candidates:
                    if u != v and not (adj[v] >> u) & 1:
                        deg_in_cands += 1
                if deg_in_cands >= 5:
                    valid_cands.append(v)
            
            if len(valid_cands) < 6:
                return False
            
            v = valid_cands[0]
            rest = valid_cands[1:]
            
            new_cands = []
            for u in rest:
                if not (adj[v] >> u) & 1:
                    new_cands.append(u)
            
            if find_k6_opt(new_cands):
                return True
            
            return find_k6_opt(rest)

        if find_k6_opt(order):
            return False
            
        return True

    # Search for a valid circulant graph
    # Try n=35 first, then n=36 if time permits.
    # For n=35, S is a subset of {1, ..., 17}.
    # There are 2^17 = 131072 possible sets. We can iterate through them or sample them.
    # Given the time limit, we should sample or iterate quickly.
    # Checking validity takes some time, so we can't check all 131k.
    # Let's try a random search with a high number of attempts.
    
    best_n = 17
    best_edges = None
    
    # Fallback graph: Paley graph of order 17? 
    # Paley(17) is K4-free? 
    # R(4,6) > 17 is known.
    # Let's construct the Paley graph of order 17 as fallback.
    # Squares mod 17: 1, 4, 9, 16, 8, 2, 15, 13
    squares = {1, 4, 9, 16, 8, 2, 15, 13}
    fallback_edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            diff = (j - i) % 17
            if diff in squares:
                fallback_edges.append([i, j])
    best_edges = fallback_edges

    # Try n=35
    target_n = 35
    half_n = target_n // 2
    all_residues = list(range(1, half_n + 1))
    
    attempts = 0
    max_attempts = 5000 # Adjust based on speed
    
    # Use random sampling of subsets
    # We want S to be "dense" enough to break independent sets, but "sparse" enough to avoid K4.
    # Average degree ~ 10-15 is typical for such Ramsey graphs.
    # |S| determines degree. deg = 2*|S|.
    # If |S|=5, deg=10. If |S|=7, deg=14.
    
    random.seed(42)
    
    while time.time() - start_time < time_limit - 5:
        # Pick a random size for S
        # Try sizes 4 to 8
        s_size = random.randint(4, 8)
        S = set(random.sample(all_residues, s_size))
        
        valid = is_valid_circulant(target_n, S)
        attempts += 1
        
        if valid:
            # Construct edges
            edges = []
            for i in range(target_n):
                for s in S:
                    j = (i + s) % target_n
                    if i < j:
                        edges.append([i, j])
            best_n = target_n
            best_edges = edges
            break
        
    # If we found n=35, we can try to see if n=36 is possible quickly?
    # Unlikely to find a circulant for 36 quickly if 35 was hard, but let's try a few.
    if best_n < 36 and time.time() - start_time < time_limit - 10:
        target_n = 36
        half_n = target_n // 2
        all_residues = list(range(1, half_n + 1))
        
        for _ in range(200): # Fewer attempts for 36
            if time.time() - start_time > time_limit - 5:
                break
            s_size = random.randint(4, 8)
            S = set(random.sample(all_residues, s_size))
            valid = is_valid_circulant(target_n, S)
            if valid:
                edges = []
                for i in range(target_n):
                    for s in S:
                        j = (i + s) % target_n
                        if i < j:
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                break

    return {'n': best_n, 'edges': best_edges}
