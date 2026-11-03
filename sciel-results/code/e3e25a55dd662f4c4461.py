import random
import numpy as np
from itertools import combinations, permutations

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using Cayley graph
    construction over finite fields (Galois Fields).
    """
    # Target: minimize n.
    # For k=5, g=9, we need a 5-regular graph.
    # Cayley graphs on (Z_q)^d are k-regular if we pick k generators (plus inverses, or symmetric set).
    # If we pick a set S of generators such that S = -S and 0 not in S, degree is |S|.
    # For odd k, we can't have S = -S without 0. So we must use a different structure or even k?
    # Wait, Cayley graph of a group G with generating set S is k-regular if |S|=k.
    # If G is abelian and S is symmetric (s in S => -s in S), then k must be even.
    # For odd k=5, we can use a non-abelian group or a non-symmetric set?
    # If S is not symmetric, the graph is directed? No, simple graph requires undirected edges.
    # Usually, for undirected Cayley graphs, we require S = S^-1.
    # If k is odd, we cannot have S = S^-1 in an abelian group without 0 in S (which creates loops) or pairing issues.
    # Actually, if k is odd, we can pick a set S where some elements are self-inverse (order 2) and others come in pairs.
    # In (Z_q)^d with q odd, only 0 is self-inverse. So we can't have odd degree in (Z_q)^d Cayley graphs?
    # Correct. (Z_q)^d with q odd has only one element of order 2 (the identity).
    # So for k=5, we need a group with more elements of order 2, or a non-abelian group.
    # Alternatively, we can use a circulant graph on Z_n?
    # Circulant graphs are Cayley graphs of Z_n. Same issue: if n is odd, only 0 is self-inverse.
    # If n is even, we have n/2 elements of order 2.
    # Let's try Circulant Graphs on Z_n.
    # We need 5 generators.
    # Let S = {a1, -a1, a2, -a2, a3}. For this to be symmetric, a3 must be self-inverse, i.e., 2*a3 = 0 mod n.
    # So a3 = n/2. This requires n to be even.
    # So, construct a circulant graph on Z_n with n even.
    # Generators: {d1, n-d1, d2, n-d2, n/2}.
    # Degree = 5.
    # We need to find n and d1, d2 such that girth >= g.
    # n should be as small as possible.
    
    # Moore bound for k=5, g=9 is 428.
    # Let's search for n in a range.
    # Known best is 1116. We hope to find something smaller or comparable.
    
    def _verify_girth_circulant(n, generators, g):
        """
        Verify girth of a circulant graph defined by n and generators.
        Generators are steps. Edges are (i, i+g) mod n for g in generators.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for gen in generators:
                j = (i + gen) % n
                if j not in adj[i]:
                    adj[i].append(j)
        
        # Check simple
        for i in range(n):
            if len(adj[i]) != len(set(adj[i])):
                return False
            if 0 in adj[i]: # Loop
                return False
        
        # BFS for girth
        # Since it's a circulant graph, we can just check from vertex 0?
        # No, girth is global. But due to symmetry, the shortest cycle through any vertex
        # is the same as through 0? Not necessarily, but often.
        # To be safe, check from all vertices or just 0 if we assume symmetry holds for girth?
        # Actually, in a vertex-transitive graph, the girth is the length of the shortest cycle.
        # We can just find the shortest cycle from 0.
        
        # BFS from 0 with non-backtracking
        # State: (current, prev)
        # visited: dict (current, prev) -> dist
        # We want to find a cycle.
        # A cycle of length L exists if we can return to 0 in L steps without backtracking.
        
        max_depth = g - 1
        # Queue: (current, prev, dist)
        # Start: (0, -1, 0)
        # But we need to track where we came from to prevent backtracking.
        # For the first step, prev is None.
        
        from collections import deque
        q = deque()
        # (current, prev, dist)
        q.append((0, None, 0))
        visited = {(0, None): 0}
        
        while q:
            curr, prev, dist = q.popleft()
            if dist >= max_depth:
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                # If we return to 0, we found a cycle
                if nb == 0 and dist + 1 >= 3:
                    return False
                # Standard BFS: if we haven't visited this (nb, curr) state
                # Wait, non-backtracking BFS state is (curr, prev).
                # We are moving from (curr, prev) to (nb, curr).
                # We should mark (nb, curr) as visited.
                if (nb, curr) not in visited:
                    visited[(nb, curr)] = dist + 1
                    q.append((nb, curr, dist + 1))
                    
        return True

    # Search for n and generators
    # n must be even.
    # Generators: d1, n-d1, d2, n-d2, n/2.
    # d1, d2 must be distinct, not 0, not n/2.
    # Also {d1, n-d1} != {d2, n-d2}.
    
    # Start n from Moore bound upwards.
    # Moore bound ~ 428.
    # Let's try n from 430 to 1200.
    # Step size 2 (to keep even).
    
    # Optimization:
    # For small n, the number of pairs (d1, d2) is small.
    # We can iterate n, then iterate d1, d2.
    
    # To speed up, we can precompute or use heuristics.
    # Let's just brute force with a time limit or attempt limit.
    
    import time
    start_time = time.time()
    timeout = 110 # Leave 10s for overhead
    
    # Range of n
    # Lower bound 428.
    # Upper bound 1200 (known best 1116).
    
    found = False
    best_n = None
    best_edges = None
    
    # We will try n from 430 to 1116 (known best) step 2.
    # If we find one, we return it.
    # If we don't find one smaller than 1116, we should return the known best?
    # The task says "smallest known... 1116".
    # If our search doesn't find anything < 1116, we should ideally return 1116 if we can construct it.
    # But constructing the specific 1116 graph might be hard if it's not a simple circulant.
    # However, if we find *any* valid graph, we return it.
    # If we find nothing, we must return *something* valid.
    # The previous code had a fallback.
    # I will implement a robust search. If it finds nothing, I will use a larger n that is likely to work (e.g., 1200) or a known valid construction.
    
    # Actually, for k=5, g=9, circulant graphs might not reach girth 9 with n < 1116?
    # The (5,9)-cage is not necessarily a circulant graph.
    # But let's try.
    
    # Heuristic for generators:
    # To avoid small cycles, generators should be "spread out".
    # d1 ~ n/3, d2 ~ n/4?
    
    for n in range(430, 1118, 2):
        if time.time() - start_time > timeout:
            break
            
        # Generate candidate generators
        # d1 in [1, n/2), d2 in [1, n/2)
        # Exclude 0, n/2.
        # Exclude d1 == d2, d1 == n-d2.
        
        # To reduce search space, we can restrict d1, d2 to a subset?
        # Or just iterate.
        # Number of pairs is ~ (n/2)^2 / 2. For n=500, 125k pairs.
        # Verification is BFS O(n * k * g).
        # 125k * 500 * 5 * 9 ~ 2.8 * 10^9 operations. Too slow in Python.
        
        # We need a faster girth check or fewer candidates.
        # Faster check:
        # In a circulant graph, a cycle corresponds to a sequence of generators summing to 0 mod n.
        # We can search for minimal length sequences of generators summing to 0.
        # This is a shortest path problem in a graph of residues?
        
        # Alternative:
        # Use the property that if there is a short cycle, it involves small combinations of generators.
        # We can just check linear combinations of length < g.
        # But BFS is safer.
        
        # Let's optimize the BFS.
        # Instead of full BFS for every (d1, d2), we can early terminate.
        # Also, we can skip many (d1, d2) if they are "close" to each other.
        
        # Let's try a smarter search:
        # Randomly sample (d1, d2) pairs for each n?
        # Or iterate d1, and for each d1, try a few d2.
        
        # Let's limit the number of pairs tested per n.
        # If n is small, test all. If n is large, test random samples.
        
        max_pairs_tested = 2000
        total_pairs = (n//2 - 1) * (n//2 - 2) // 2 # Approx
        
        if total_pairs <= max_pairs_tested:
            # Test all
            pairs = []
            half = n // 2
            for d1 in range(1, half):
                for d2 in range(d1 + 1, half):
                    # Check distinctness
                    if d1 == n - d2: continue
                    pairs.append((d1, d2))
            # Shuffle to avoid bias?
            random.shuffle(pairs)
        else:
            # Random sample
            pairs = []
            half = n // 2
            for _ in range(max_pairs_tested):
                d1 = random.randint(1, half - 1)
                d2 = random.randint(1, half - 1)
                if d1 == d2 or d1 == n - d2:
                    continue
                pairs.append((d1, d2))
        
        for d1, d2 in pairs:
            gens = [d1, n - d1, d2, n - d2, n // 2]
            # Check for duplicates in gens
            if len(set(gens)) != 5:
                continue
            
            if _verify_girth_circulant(n, gens, g):
                # Construct edges
                edges = []
                for i in range(n):
                    for gen in gens:
                        j = (i + gen) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        # If we didn't find one for this n, continue to next n
        
    # If no graph found in range, fallback to a known valid larger graph
    # Or just return a valid graph from a larger n
    # Let's try n = 1200 with random generators
    n = 1200
    half = n // 2
    for _ in range(1000):
        d1 = random.randint(1, half - 1)
        d2 = random.randint(1, half - 1)
        if d1 == d2 or d1 == n - d2:
            continue
        gens = [d1, n - d1, d2, n - d2, n // 2]
        if len(set(gens)) != 5:
            continue
        if _verify_girth_circulant(n, gens, g):
            edges = []
            for i in range(n):
                for gen in gens:
                    j = (i + gen) % n
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    # Absolute fallback: return a simple valid graph if possible, or raise
    # But we must return a valid graph.
    # If we failed, we might return a non-optimal but valid graph.
    # For k=5, g=9, n=1200 should work with high probability.
    # If it didn't, something is wrong with the verifier or search.
    
    # Re-try with fixed seed for reproducibility in fallback
    random.seed(42)
    n = 1200
    half = n // 2
    for _ in range(10000):
        d1 = random.randint(1, half - 1)
        d2 = random.randint(1, half - 1)
        if d1 == d2 or d1 == n - d2:
            continue
        gens = [d1, n - d1, d2, n - d2, n // 2]
        if len(set(gens)) != 5:
            continue
        if _verify_girth_circulant(n, gens, g):
            edges = []
            for i in range(n):
                for gen in gens:
                    j = (i + gen) % n
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    # If all else fails, return a dummy valid graph (might be large)
    # Construct a 5-regular graph on n=1000 using configuration model as last resort
    return _fallback_random_regular(1000, k, g)

def _fallback_random_regular(n, k, g):
    import random
    for _ in range(100):
        # Configuration model
        stubs = []
        for i in range(n):
            for j in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = set()
        valid = True
        adj = [[] for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            continue
        if all(len(a) == k for a in adj):
            # Check girth
            if _verify_girth_circulant(n, None, g) is None: # Placeholder
                pass
            # Use the BFS verifier from before
            if _verify_girth(adj, g):
                return n, list(edges)
    # If failed, return a simple cycle? No, must be k-regular.
    # Just return the last generated one even if girth check failed? No, must be valid.
    # This should rarely happen.
    raise Exception("Failed to construct valid graph")

def _verify_girth(graph, g):
    """
    Verify girth >= g using non-backtracking BFS from each vertex.
    """
    n = len(graph)
    max_depth = g - 1
    
    for s in range(n):
        # BFS from s
        # State: (curr, prev)
        visited = set()
        queue = [(s, -1, 0)]
        visited.add((s, -1))
        
        while queue:
            new_queue = []
            for curr, prev, dist in queue:
                if dist >= max_depth:
                    continue
                for nb in graph[curr]:
                    if nb == prev:
                        continue
                    if nb == s and dist + 1 >= 3:
                        return False
                    if (nb, curr) not in visited:
                        visited.add((nb, curr))
                        new_queue.append((nb, curr, dist + 1))
            queue = new_queue
    return True
