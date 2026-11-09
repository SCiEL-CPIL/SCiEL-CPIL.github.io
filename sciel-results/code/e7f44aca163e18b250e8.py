import random
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Leave buffer for overhead
    
    # Try to find a graph for n=35, then 34, etc.
    for n in range(35, 16, -1):
        # Multiple restarts for each n
        for restart in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Spectral construction approach:
            # Generate a random symmetric matrix, take top eigenvectors, binarize
            # This leverages algebraic structure to jump to high-n regions
            
            # Create random symmetric matrix
            A = np.random.randn(n, n)
            A = (A + A.T) / 2.0
            
            # Get eigenvalues and eigenvectors
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(A)
            except:
                continue
            
            # Use top k eigenvectors to form adjacency
            # We want to create a graph where vertices are "aligned" in eigenspace
            # Take top few eigenvectors and threshold their signs
            
            # Method 1: Use sign of projection onto top eigenvectors
            # For each pair of vertices, compute inner product of their projections
            # and threshold to create edge
            
            # Alternative: Use the top eigenvector to partition vertices
            # But we need a more nuanced approach
            
            # Let's try: create adjacency based on cosine similarity in eigenspace
            # Project vertices onto top m eigenvectors
            m = min(3, n)  # Number of eigenvectors to use
            top_eigvecs = eigenvectors[:, :m]
            
            # Compute pairwise similarities
            # For each pair (i,j), compute similarity = dot(top_eigvecs[i], top_eigvecs[j])
            # Threshold this to create edges
            
            # Actually, let's try a different approach:
            # Use the Hadamard/Paley-inspired structure
            # For n that is a power of 2 or prime power, use structured matrices
            
            if n in [16, 32]:
                # Use Hadamard matrix construction
                try:
                    H = hadamard_matrix(n)
                    # Convert to adjacency: edge if H[i,j] == -1 (or +1)
                    adj = (H < 0).astype(np.int8)
                    np.fill_diagonal(adj, 0)
                    adj = np.triu(adj, 1)
                    adj = adj + adj.T
                except:
                    continue
            elif n in [17, 31]:
                # Use Paley graph construction
                adj = paley_graph(n)
            else:
                # Use random symmetric matrix with spectral bias
                # Create random matrix, then use its top eigenvectors
                B = np.random.randn(n, n)
                B = (B + B.T) / 2.0
                try:
                    vals, vecs = np.linalg.eigh(B)
                except:
                    continue
                
                # Take top 2-3 eigenvectors
                k = min(2, n)
                V = vecs[:, :k]
                
                # Create adjacency based on sign agreement in eigenspace
                # For each pair (i,j), check if their projections have same sign pattern
                adj = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i+1, n):
                        # Compare signs of projections
                        vi = V[i]
                        vj = V[j]
                        # Edge if signs agree on at least half the coordinates
                        agreement = np.sum((vi > 0) == (vj > 0))
                        if agreement >= k // 2 + 1:
                            adj[i, j] = 1
                            adj[j, i] = 1
            
            # Local Search refinement
            for step in range(5000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Check for K4
                k4 = find_k4(adj)
                if k4:
                    # Pick random edge in K4 to remove
                    edges_in_k4 = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                                   (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                    u, v = random.choice(edges_in_k4)
                    adj[u, v] = 0
                    adj[v, u] = 0
                    continue
                
                # Check for Ind6
                ind6 = find_ind6(adj)
                if ind6:
                    # Pick random pair in Ind6 to add edge
                    pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i+1, 6)]
                    u, v = random.choice(pairs)
                    adj[u, v] = 1
                    adj[v, u] = 1
                    continue
                
                # No violations found
                # Verify exactly
                if not find_k4(adj) and not find_ind6(adj):
                    # Convert to edge list
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
    # Fallback: Return the best known small graph (Paley 17) if all else fails
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def hadamard_matrix(n):
    """Construct Hadamard matrix of order n (must be power of 2)."""
    if n & (n - 1) != 0:  # Check if power of 2
        raise ValueError("n must be power of 2")
    if n == 1:
        return np.array([[1]])
    H = hadamard_matrix(n // 2)
    return np.block([[H, H], [H, -H]])

def paley_graph(n):
    """Construct Paley graph on n vertices (n must be prime congruent to 1 mod 4)."""
    # Check if n is prime
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(np.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    
    if not is_prime(n) or n % 4 != 1:
        raise ValueError("n must be prime congruent to 1 mod 4")
    
    # Quadratic residues
    squares = {x * x % n for x in range(1, n)}
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj

def find_k4(adj):
    n = adj.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                common = np.where(adj[i] & adj[j])[0]
                if len(common) < 2:
                    continue
                for k_idx in range(len(common)):
                    for l_idx in range(k_idx+1, len(common)):
                        k = common[k_idx]
                        l = common[l_idx]
                        if adj[k, l]:
                            return [i, j, k, l]
    return None

def find_ind6(adj):
    n = adj.shape[0]
    def dfs(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if len(candidates) + len(current_set) < 6:
            return None
        
        for idx, v in enumerate(candidates):
            new_candidates = []
            for w in candidates[idx+1:]:
                if not adj[v, w]:
                    new_candidates.append(w)
            
            res = dfs(new_candidates, current_set + [v])
            if res:
                return res
        return None

    return dfs(list(range(n)), [])
