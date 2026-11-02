"""
Strategy: Maximum Clique via Branch and Bound on D11 Lattice Candidates.

The problem is to find the largest set of vectors in R^11 such that the minimum
pairwise distance is at least the maximum vector norm.
Let S be the set of vectors. For all u, v in S:
||u - v||^2 >= max(||u||^2, ||v||^2)

We restrict to vectors from the D11 lattice: vectors with entries in {0, 1, -1}
and an even number of non-zero entries? No, D_n is even weight.
Actually, the standard kissing number constructions often use:
1. Weight 2 vectors: norm^2 = 2.
   There are C(11,2) * 4 = 220 such vectors.
   As analyzed in the current program, ALL weight-2 vectors are mutually compatible?
   Let's re-verify.
   u, v weight 2. ||u-v||^2 >= 2.
   If u, v share 2 coords with same signs: u.v = 2. ||u-v||^2 = 2+2-4=0. Conflict.
   So we cannot take ALL weight-2 vectors. We can take at most 1 from each "direction" pair?
   Actually, if u = (1,1,0..), v=(1,1,0..) is conflict.
   If u = (1,1,0..), v=(1,-1,0..), u.v=0. ||u-v||^2 = 4. OK.
   If u = (1,1,0..), v=(1,0,1..), u.v=1. ||u-v||^2 = 2. OK.
   
   So the conflict graph for weight-2 vectors is a matching? No.
   Two weight-2 vectors conflict ONLY if they are identical or negatives?
   Wait. u=(1,1,0..), v=(1,1,0..). u.v=2. Conflict.
   u=(1,1,0..), v=(-1,-1,0..). u.v=-2. ||u-v||^2 = 2+2-2(-2)=8. OK.
   u=(1,1,0..), v=(1,1,0..) is the only conflict for a specific u among weight-2?
   What about u=(1,1,0..), v=(1,1,0..)? Yes.
   Are there other conflicts?
   u=(1,1,0..), v=(1,1,0..) -> conflict.
   u=(1,1,0..), v=(1,1,0..) is the same vector.
   So if we just pick distinct weight-2 vectors, do they conflict?
   If u != v, can u.v = 2?
   u.v = 2 implies u and v have same signs on 2 coords and 0 elsewhere.
   Since they are weight-2, they MUST be identical.
   So, ANY set of DISTINCT weight-2 vectors is a valid kissing configuration?
   Let's check u=(1,1,0..), v=(1,1,0..) is not distinct.
   Let u=(1,1,0..), v=(1,1,0..) is the same.
   Let u=(1,1,0..), v=(1,1,0..) -> u.v=2.
   If u != v, max u.v is 1?
   If they share 1 coord: u.v = 1 or -1.
   If they share 0 coords: u.v = 0.
   If they share 2 coords, they are identical (if same signs) or not?
   If they share indices {i,j}, u[i]=1, u[j]=1.
   v[i]=1, v[j]=1 => v=u.
   v[i]=1, v[j]=-1 => u.v=0.
   v[i]=-1, v[j]=1 => u.v=0.
   v[i]=-1, v[j]=-1 => u.v=-2.
   
   So YES. The set of all 220 distinct weight-2 vectors is a valid kissing configuration.
   Score = 220.
   
   Can we do better?
   The known kissing number for d=11 is likely higher.
   Lower bound is often cited around 504+ for d=10/11?
   Actually, for d=10, k=504 is proven? No, d=8 is 240. d=9 is 366?
   Let's look at the D11 lattice vectors with weight 4.
   Weight 4 vectors have norm^2 = 4.
   Condition: ||u-v||^2 >= 4 (if we mix with weight 2, max norm is 2, so dist >= 2, dist^2 >= 4).
   Wait, if we have a weight-4 vector (norm 2) and a weight-2 vector (norm sqrt(2)~1.41),
   Max norm is 2. Min dist must be >= 2.
   dist^2 >= 4.
   
   If we use ONLY weight-4 vectors:
   Norm^2 = 4.
   Condition: ||u-v||^2 >= 4.
   4+4 - 2 u.v >= 4 => 8 - 2 u.v >= 4 => u.v <= 2.
   
   How many weight-4 vectors are there?
   C(11,4) * 2^4 = 330 * 16 = 5280.
   This is a much larger pool.
   If we can find a large subset of these 5280 vectors where pairwise dot products are <= 2, we win.
   
   Note: If u and v are weight-4 and identical, u.v=4. Conflict.
   If u and v differ in one sign: u.v = 4-2 = 2. OK.
   If u and v differ in two signs: u.v = 4-4 = 0. OK.
   If u and v share 3 coords:
   u = (1,1,1,1,0..), v = (1,1,1,-1,0..). u.v=2. OK.
   u = (1,1,1,1,0..), v = (1,1,1,1,0..) identical. Conflict.
   u = (1,1,1,1,0..), v = (1,1,1,0,1..). Share 3. u.v=3. CONFLICT.
   
   So, weight-4 vectors conflict if they share 3 or 4 coordinates with the same signs.
   Specifically, if they share 3 coordinates with the same signs, u.v >= 3.
   If they share 4 coordinates (identical), u.v = 4.
   
   So the conflict condition for weight-4 vectors is:
   u.v > 2.
   
   We want the Maximum Independent Set in the graph where edges connect vectors with u.v > 2.
   Equivalently, Maximum Clique in the graph where edges connect vectors with u.v <= 2.
   
   The number of candidates is 5280.
   Building the adjacency matrix is 5280^2 ~ 27M entries. This is feasible in memory (bool array).
   Solving Max Clique on 5280 nodes is hard, but the graph is dense?
   Most pairs have small dot products.
   If the graph is very dense, Max Clique is large.
   
   Let's try to implement a fast Max Clique solver (Tomita's algorithm) on the "compatibility" graph of weight-4 vectors.
   If that yields > 220, we return it.
   We can also mix weight-2 and weight-4, but let's start with pure weight-4 as it has a larger pool.
   
   Actually, we can include weight-2 vectors too.
   But weight-2 vectors have norm^2=2.
   If we mix, max norm is 2 (from weight-4).
   So weight-2 vectors must be distance >= 2 from all weight-4 vectors.
   dist^2(w2, w4) = 2 + 4 - 2 u.v = 6 - 2 u.v.
   Need 6 - 2 u.v >= 4 => u.v <= 1.
   
   So a weight-2 vector is compatible with a weight-4 vector if their dot product is <= 1.
   
   Strategy:
   1. Generate all weight-4 vectors (5280).
   2. Generate all weight-2 vectors (220).
   3. Build compatibility graph.
      - Edge between w4_i and w4_j if dot(w4_i, w4_j) <= 2.
      - Edge between w2_i and w2_j if dot(w2_i, w2_j) <= 1. (Always true for distinct w2s? Yes, max dot is 1).
      - Edge between w2_i and w4_j if dot(w2_i, w4_j) <= 1.
   4. Find Maximum Clique in this graph.
   
   Since 5280+220 = 5500 nodes.
   The graph is likely quite dense.
   Tomita's algorithm is efficient for dense graphs.
   
   Let's implement Tomita's Max Clique.
"""

import numpy as np
import time
import itertools
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Generate Candidates
    # Weight 4 vectors
    w4_indices = []
    w4_vectors = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            w4_vectors.append(v)
    w4_vectors = np.array(w4_vectors)
    
    # Weight 2 vectors
    w2_vectors = []
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            w2_vectors.append(v)
    w2_vectors = np.array(w2_vectors)
    
    # Combine into one pool
    # To simplify, we can just solve Max Clique on the combined pool.
    # However, weight-2 vectors are "easier" to place.
    # Let's stick to the combined pool for a unified Max Clique.
    
    all_vectors = np.vstack([w4_vectors, w2_vectors])
    N = len(all_vectors)
    
    # 2. Build Compatibility Matrix
    # Condition: ||u-v||^2 >= max(||u||^2, ||v||^2)
    # Let n_u = ||u||^2, n_v = ||v||^2.
    # n_u + n_v - 2 u.v >= max(n_u, n_v)
    # 2 u.v <= min(n_u, n_v)
    # u.v <= min(n_u, n_v) / 2
    
    norms_sq = np.sum(all_vectors * all_vectors, axis=1)
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    limits = min_norms_sq / 2.0
    
    G = all_vectors @ all_vectors.T
    
    # Compatibility: G[i,j] <= limits[i,j]
    # Note: Diagonal is always compatible with itself?
    # We want a set of DISTINCT vectors.
    # In Max Clique, we pick a subset of nodes such that all pairs are connected.
    # So we build the adjacency matrix A where A[i,j] = True if i and j are compatible.
    A = G <= limits
    # Remove self-loops for clique finding (or handle in algo)
    np.fill_diagonal(A, False)
    
    # 3. Maximum Clique via Branch and Bound (Tomita)
    # We implement a recursive search.
    # To make it fast, we use bitsets or boolean arrays.
    # Given N ~ 5500, a full NxN boolean matrix is ~30MB. Acceptable.
    
    # Tomita's algorithm (MCQ)
    # We maintain a list of candidates R (current clique) and P (possible candidates).
    # We order P to improve pruning.
    
    # For efficiency in Python, we will use a list of indices for P.
    # A simple recursive implementation might be too slow for N=5500 if not optimized.
    # However, the graph is very dense.
    # Average degree is high.
    
    # Let's try a greedy heuristic first to get a good lower bound (initial clique size).
    # Greedy: Pick node with highest degree in P, add to clique, remove non-neighbors.
    
    def greedy_clique(A, N):
        # Simple greedy
        clique = []
        P = list(range(N))
        while P:
            # Pick node with max degree in induced subgraph P
            # This is O(|P|^2) per step.
            # Optimization: Just pick first? No, max degree is better.
            # For speed, let's just pick a random node or first node?
            # Let's try picking the node with the most neighbors in P.
            # To speed up, we can sample or use a static degree heuristic.
            # Let's use static degree for the initial greedy.
            
            # Actually, a simple greedy that picks nodes in order of global degree is often good.
            # Let's sort nodes by global degree once.
            pass
            
        # Let's implement a simple static-order greedy.
        degrees = np.sum(A, axis=1)
        order = np.argsort(-degrees)
        
        clique = []
        P_mask = np.ones(N, dtype=bool)
        
        for v in order:
            if not P_mask[v]:
                continue
            clique.append(v)
            # Remove non-neighbors of v from P
            # Neighbors of v are A[v] == True
            # Non-neighbors are ~A[v]
            P_mask[~A[v]] = False
            P_mask[v] = False
            
        return clique
    
    initial_clique = greedy_clique(A, N)
    best_clique = initial_clique
    best_size = len(best_clique)
    
    # Now, attempt to improve using local search or a limited B&B.
    # Given the time limit (145s), a full B&B on 5500 nodes is risky in Python.
    # But the graph is dense.
    # Let's try a "Greedy with Random Restarts" on the ordering.
    
    rng = np.random.default_rng(42)
    
    # If we have time, do random restarts of the greedy.
    # Each restart: perturb the degree ordering.
    
    for _ in range(100):
        if time.time() > deadline - 5:
            break
            
        # Perturb degrees
        perturbed_degrees = degrees + rng.random(N) * 10
        order = np.argsort(-perturbed_degrees)
        
        clique = []
        P_mask = np.ones(N, dtype=bool)
        
        for v in order:
            if not P_mask[v]:
                continue
            clique.append(v)
            P_mask[~A[v]] = False
            P_mask[v] = False
            
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique
    
    # If best_size is still small, we might have a bug or the graph is sparse?
    # Weight-4 vectors should be dense.
    # Let's check the size.
    
    # One more strategy: If the greedy is stuck, try to add more vectors from the weight-2 set
    # to the weight-4 clique?
    # Actually, the greedy already considers all nodes.
    
    # Return the best clique found
    if not best_clique:
        # Fallback: return a single vector
        return np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return all_vectors[best_clique]

# EVOLVE-BLOCK-END
