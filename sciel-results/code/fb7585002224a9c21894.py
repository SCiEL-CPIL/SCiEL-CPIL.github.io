import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # We aim for n=35, 34, ... down to 20
    # Spectral construction: 
    # 1. Generate random symmetric matrix A
    # 2. Compute top k eigenvectors
    # 3. Form new adjacency from these eigenvectors (e.g., sign of inner product or threshold)
    # 4. Repair locally
    
    best_result = None
    
    # Try different n values
    for n in range(35, 19, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Multiple restarts with different spectral strategies
        for restart in range(30):
            if time.time() - start_time > time_limit:
                break
                
            adj = None
            
            # Strategy 1: Random symmetric matrix eigenvectors
            # Generate random symmetric matrix
            A = np.random.randn(n, n)
            A = (A + A.T) / 2.0
            
            # Compute top k eigenvectors
            # k should be small, e.g., 2 to 5
            k = random.randint(2, 5)
            try:
                # Use eigh for symmetric matrices
                eigenvalues, eigenvectors = np.linalg.eigh(A)
                # Take top k (largest eigenvalues)
                # eigh returns in ascending order, so take last k
                top_vecs = eigenvectors[:, -k:]
                
                # Construct adjacency from these vectors
                # Method: Sign of the sum of outer products? Or just take the first eigenvector?
                # Let's try: adj[i,j] = 1 if sum(top_vecs[i,:] * top_vecs[j,:]) > 0 else 0?
                # Or simpler: Use the first eigenvector as a partition?
                # Let's try binarizing the eigenvector matrix itself.
                # If we use the first eigenvector (principal component), it often gives a bipartite-like structure.
                # For Ramsey graphs, we want a balance.
                
                # Alternative: Use the matrix of inner products of the top k eigenvectors?
                # Let's try: B = top_vecs.T @ top_vecs. Then binarize B?
                # Or just take the first eigenvector v. adj[i,j] = 1 if v[i]*v[j] < 0? (Bipartite)
                # But we need no K4 and no Ind6. Bipartite graphs have no odd cycles, so no K4.
                # But Ind6 is easy to find in bipartite graphs if not dense enough.
                
                # Let's try a different spectral method:
                # Generate random matrix, take top eigenvector, use sign to define a cut.
                # Then add random edges within parts to break Ind6, and remove edges to break K4.
                
                v = eigenvectors[:, -1] # Principal eigenvector
                # Binarize based on sign
                # Threshold at 0
                mask = v > 0
                # Create adjacency: 1 if same sign? Or different?
                # Let's start with a random graph and use spectral info to guide initialization?
                # Actually, the prompt suggests "binarizing the signs to form adjacency".
                # Let's try: adj[i,j] = 1 if v[i] and v[j] have same sign? 
                # This creates two cliques? No, v[i] > 0 and v[j] > 0 -> 1. 
                # This is a union of two cliques. Cliques have K4. Bad.
                
                # Let's try: adj[i,j] = 1 if v[i] * v[j] < 0? (Bipartite)
                # Bipartite graphs are K4-free. We just need to ensure no Ind6.
                # A bipartite graph with parts of size ~n/2.
                # Max independent set in bipartite graph is max(part sizes) + ...?
                # Actually, in a bipartite graph, an independent set can take all vertices from one part.
                # So if n/2 >= 6, we have an Ind6. 
                # So simple bipartite from 1 eigenvector is bad for n>=12.
                
                # Let's use a combination of eigenvectors.
                # Let's try the "Spectral Clustering" style adjacency.
                # Or simply: Generate random graph, but initialize using random walk?
                
                # Let's stick to the prompt's hint: "top-k eigenvectors ... binarizing the signs".
                # Maybe: Project vertices onto the top k eigenvectors, then use distance?
                # If distance < threshold, edge?
                # Let's try:
                # X = top_vecs (n x k)
                # dist = ||X[i] - X[j]||^2
                # adj[i,j] = 1 if dist < median(dist)
                
                X = top_vecs
                # Compute pairwise distances
                # Efficient way: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
                norms = np.sum(X**2, axis=1)
                dot = X @ X.T
                dist_sq = norms[:, None] + norms[None, :] - 2 * dot
                np.fill_diagonal(dist_sq, np.inf)
                
                # Threshold: median of non-inf values
                valid_dists = dist_sq[np.isfinite(dist_sq)]
                if len(valid_dists) > 0:
                    threshold = np.median(valid_dists)
                    adj = (dist_sq < threshold).astype(np.int8)
                    np.fill_diagonal(adj, 0)
                    # Symmetrize (should be symmetric already)
                    adj = (adj + adj.T) > 0
                    adj = adj.astype(np.int8)
                
            except Exception:
                adj = None
            
            # Strategy 2: Paley-like construction if n is a prime power
            if adj is None:
                # Check if n is a prime power suitable for Paley
                # Paley graphs exist for q = p^k where p is prime, q ≡ 1 (mod 4)
                # Smallest such q > 19: 17, 29, 37...
                # 17 is Paley. 29 is Paley.
                # Let's try n=29 if n==29
                if n == 29:
                    q = 29
                    # Quadratic residues mod 29
                    squares = set((x * x) % q for x in range(q))
                    adj = np.zeros((q, q), dtype=np.int8)
                    for i in range(q):
                        for j in range(i + 1, q):
                            if (j - i) % q in squares:
                                adj[i, j] = 1
                                adj[j, i] = 1
                elif n == 17:
                    q = 17
                    squares = set((x * x) % q for x in range(q))
                    adj = np.zeros((q, q), dtype=np.int8)
                    for i in range(q):
                        for j in range(i + 1, q):
                            if (j - i) % q in squares:
                                adj[i, j] = 1
                                adj[j, i] = 1
                else:
                    # Fallback: Random graph
                    adj = (np.random.rand(n, n) < 0.5).astype(np.int8)
                    np.fill_diagonal(adj, 0)
                    adj = (adj + adj.T) > 0
                    adj = adj.astype(np.int8)

            # Local Search / Repair
            # We need to remove K4s and add edges to break Ind6s
            # This is a constraint satisfaction problem.
            
            # We will iterate:
            # 1. Find K4. If found, remove a random edge from it.
            # 2. Find Ind6. If found, add a random edge within it.
            # 3. Repeat until no violations or time out.
            
            # To make it fast, we need efficient find_k4 and find_ind6.
            
            success = False
            for step in range(2000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Check K4
                k4 = find_k4_fast(adj)
                if k4:
                    # Remove a random edge from the K4
                    edges_in_k4 = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                                   (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                    u, v = random.choice(edges_in_k4)
                    adj[u, v] = 0
                    adj[v, u] = 0
                    continue
                
                # Check Ind6
                ind6 = find_ind6_fast(adj)
                if ind6:
                    # Add a random edge within the Ind6
                    pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
                    u, v = random.choice(pairs)
                    if adj[u, v] == 0:
                        adj[u, v] = 1
                        adj[v, u] = 1
                    continue
                
                # No violations
                # Verify exactly
                if not find_k4_fast(adj) and not find_ind6_fast(adj):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    # This is a valid graph for n
                    return {'n': n, 'edges': edges}
            
    # Fallback: Return a known small graph
    # Paley 17 is valid
    q = 17
    squares = set((x * x) % q for x in range(q))
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': 17, 'edges': edges}

def find_k4_fast(adj):
    n = adj.shape[0]
    # Iterate over pairs
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                # Common neighbors
                common = np.where(adj[i] & adj[j])[0]
                # Exclude i and j
                common = common[(common != i) & (common != j)]
                if len(common) < 2:
                    continue
                # Check if any edge exists in common
                # We need to check if the subgraph induced by common has an edge
                # common is an array of indices
                # We can check adj[common[:, None], common[None, :]]
                # But simpler: loop
                for k_idx in range(len(common)):
                    k = common[k_idx]
                    for l_idx in range(k_idx + 1, len(common)):
                        l = common[l_idx]
                        if adj[k, l]:
                            return [i, j, k, l]
    return None

def find_ind6_fast(adj):
    n = adj.shape[0]
    # Backtracking with pruning
    # Order vertices by degree (ascending) to prune faster? 
    # For independent set, we want vertices with low degree?
    # Actually, for finding an independent set, we want to pick vertices that don't conflict.
    # Greedy: pick vertex with lowest degree, remove it and its neighbors.
    # But we need size 6.
    
    # Let's use a simple recursive search with memoization or just pruning.
    # Given n <= 35, and we are looking for size 6, it should be fast if we prune well.
    
    def dfs(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if len(candidates) + len(current_set) < 6:
            return None
        
        # Pruning: If the maximum independent set of the induced subgraph of candidates
        # is less than 6 - len(current_set), return None.
        # Approximation: If sum of degrees in candidates is high, it's unlikely.
        
        # Heuristic: Sort candidates by degree in the induced subgraph?
        # Or just pick the vertex with the fewest neighbors in the remaining candidates?
        # Let's try picking the vertex with the lowest degree in the whole graph first?
        # No, dynamic is better.
        
        # To speed up, we can try to find a vertex that is isolated in the remaining graph?
        # Or just iterate.
        
        # Optimization: If a vertex v is connected to all other candidates, it cannot be in the independent set.
        # We can filter such vertices out? No, they can be in the set if we don't pick their neighbors.
        
        # Let's iterate through candidates.
        for idx, v in enumerate(candidates):
            # New candidates: those in candidates[idx+1:] that are NOT neighbors of v
            new_candidates = []
            for w in candidates[idx + 1:]:
                if not adj[v, w]:
                    new_candidates.append(w)
            
            res = dfs(new_candidates, current_set + [v])
            if res:
                return res
        return None

    return dfs(list(range(n)), [])
