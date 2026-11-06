import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit
    
    # The task asks for a C4-free graph on n vertices with min degree >= n - 39.
    # We want to maximize n. The theoretical bound is around 46/47.
    # For n=46, min degree >= 7.
    # For n=47, min degree >= 8.
    
    # Strategy:
    # 1. Try to construct a C4-free graph on n=47 with min degree >= 8.
    # 2. If that fails, try n=46 with min degree >= 7.
    # 3. If that fails, try n=45 with min degree >= 6.
    # ... and so on.
    
    # A good starting point for C4-free graphs with high degree is often found in:
    # - Polarity graphs of projective planes (e.g., n = q^2+q+1, deg ~ q)
    # - Circulant graphs with specific difference sets.
    # - Random regular graphs with C4 removal (but removal is hard).
    
    # Let's use a hybrid approach:
    # 1. Generate a random regular graph (or near-regular) using a fast method.
    # 2. Use a local search to remove C4s and balance degrees.
    
    # However, removing C4s from a dense graph is difficult.
    # A better approach for small n is to use a SAT-like solver or backtracking,
    # but that might be slow.
    
    # Alternative: Use known constructions.
    # For n=46, we can try a circulant graph.
    # A circulant graph C_n(S) is C4-free if the set S has no solutions to
    # s1 - s2 = s3 - s4 for distinct s_i in S (in Z_n).
    
    # Let's try to find a difference set or similar structure.
    # Or, simply use a randomized local search starting from a random graph,
    # but with a strong bias towards maintaining high degree.
    
    # Actually, the most robust method for n <= 50 is often a simple backtracking
    # or a very aggressive local search.
    
    # Let's implement a local search:
    # 1. Start with a random graph G.
    # 2. While G contains a C4:
    #    a. Pick a C4.
    #    b. Remove an edge from the C4 that causes the least damage to min degree.
    # 3. Check min degree. If too low, try to add edges that don't create C4s.
    
    # This "remove then add" cycle is tricky.
    
    # Let's try a different structural approach:
    # **Finite Geometry Construction**
    # The polarity graph of the projective plane of order q has n = q^2 + q + 1 vertices.
    # For q=6, n=43. Min degree is roughly q-1=5 or q=6?
    # The degree is q-1 for one type of vertex and q for others?
    # Actually, for a polarity graph, the number of self-orthogonal points is fixed.
    # This might be too specific.
    
    # Let's try **Circulant Graphs** first. They are fast to generate and check.
    # We want a C4-free circulant graph on n vertices with high degree.
    
    def is_c4_free_circulant(n, S):
        # S is a set of integers in 1..n//2 (symmetric)
        # C4 exists if there are distinct a,b,c,d in S such that a-b = c-d mod n
        # or a+b = c+d mod n (if we consider undirected steps carefully)
        # Actually, a C4 in a circulant graph corresponds to a solution to
        # x1 + x2 = x3 + x4 (mod n) where xi are steps.
        # Since the graph is undirected, steps are +/- s.
        # A C4 is a sequence of 4 steps s1, s2, s3, s4 summing to 0.
        # s1+s2+s3+s4 = 0 mod n.
        # The vertices are 0, s1, s1+s2, s1+s2+s3, 0.
        # For it to be a simple C4, no intermediate vertex can be 0 or repeat.
        
        S_set = set(S)
        # Generate all sums of two elements from S (with signs)
        # This is getting complicated.
        # Simpler: Just build the adjacency matrix and check for C4 using numpy.
        # n=47 is small enough.
        
        adj = np.zeros((n, n), dtype=int)
        for s in S:
            for i in range(n):
                j = (i + s) % n
                if i != j:
                    adj[i, j] = 1
        adj = np.triu(adj, 1) + np.triu(adj, 1).T
        # Ensure symmetry and no self loops
        np.fill_diagonal(adj, 0)
        
        A2 = adj @ adj
        return not np.any(A2 >= 2), adj, np.min(adj.sum(axis=1))
        
    # We want to find S for n=47 such that min degree >= 8 and no C4.
    # Degree of vertex i in C_n(S) is 2*|S| (if no s = n/2).
    # So if |S|=4, degree is 8.
    # We need a set S of size 4 in Z_47 that is C4-free.
    
    # Let's search for such an S.
    # There are C(23, 4) = 8855 possible sets of size 4 (since s and n-s are equivalent in undirected graph).
    # This is very small! We can brute force it.
    
    def find_circulant(n, target_deg):
        k = target_deg // 2
        # We need |S| = k.
        # Candidates for S are subsets of {1, ..., n//2}
        max_s = n // 2
        candidates = list(range(1, max_s + 1))
        
        # We want to pick k numbers from candidates.
        # To be C4-free, the set S should be a "Sidon set" in Z_n roughly.
        # Let's iterate through combinations.
        
        import itertools
        
        # To speed up, we can prune. But 8855 is small.
        # If target_deg is 8, k=4.
        # If target_deg is 7, k=3.5 -> not possible for regular.
        # Circulant graphs are regular.
        # So for n=47, we can only get even degrees.
        # Min degree >= 8 implies degree must be at least 8.
        # So we look for a 8-regular C4-free circulant graph on 47 vertices.
        
        best_S = None
        # Iterate through combinations of size k
        for S in itertools.combinations(candidates, k):
            S_tuple = tuple(S)
            # Quick check: No two differences equal.
            # Difference set D = { (a-b) mod n | a,b in S, a!=b }
            # If |D| < 2*k*(k-1)/2, there is a collision -> C4?
            # Actually, a collision in differences s1-s2 = s3-s4 implies s1+s4 = s2+s3.
            # This forms a C4 if the vertices are distinct.
            
            diffs = []
            valid = True
            for i in range(len(S_tuple)):
                for j in range(i+1, len(S_tuple)):
                    d = (S_tuple[i] - S_tuple[j]) % n
                    diffs.append(d)
            
            # Check for duplicates in diffs (considering symmetry? No, directed diffs)
            # If d1 == d2, then s_i - s_j = s_k - s_l.
            # s_i + s_l = s_j + s_k.
            # This is a solution to the C4 equation.
            if len(set(diffs)) != len(diffs):
                continue
                
            # It's a Sidon set! Likely C4-free.
            # Let's verify with the matrix check just in case.
            is_free, adj, min_deg = is_c4_free_circulant(n, S_tuple)
            if is_free and min_deg >= target_deg:
                return S_tuple, adj
                
        return None, None

    # Try n=47 first
    n_target = 47
    target_deg = n_target - 39 # 8
    
    S, adj = find_circulant(n_target, target_deg)
    
    if adj is not None:
        # Verify once more
        A = adj.astype(int)
        A2 = A @ A
        if not np.any(A2 >= 2) and np.min(A.sum(axis=1)) >= target_deg:
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
            
    # If n=47 failed, try n=46
    # For n=46, target_deg = 7.
    # Circulant graphs are regular, so degree must be even.
    # So we look for degree 8 (which is >= 7).
    n_target = 46
    target_deg = 8 # Must be even
    
    S, adj = find_circulant(n_target, target_deg)
    
    if adj is not None:
        A = adj.astype(int)
        A2 = A @ A
        if not np.any(A2 >= 2) and np.min(A.sum(axis=1)) >= n_target - 39:
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
            
    # If circulant fails, try n=45
    # target_deg = 6. Even.
    n_target = 45
    target_deg = 6
    
    S, adj = find_circulant(n_target, target_deg)
    
    if adj is not None:
        A = adj.astype(int)
        A2 = A @ A
        if not np.any(A2 >= 2) and np.min(A.sum(axis=1)) >= n_target - 39:
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
            
    # Fallback: Return a valid small graph
    # n=39, empty graph. Min degree 0 >= 0.
    return {'n': 39, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
