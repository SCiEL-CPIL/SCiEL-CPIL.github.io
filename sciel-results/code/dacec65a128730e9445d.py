"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin for 150s limit
    
    # Convert to float matrix for efficient matrix multiplication
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = adj > 0.5
    degrees = adj.sum(axis=1)
    
    # Precompute neighbor indices for each vertex for fast access
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_order(order):
        """Compute greedy MIS size given a priority order (array of vertex indices)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count

    def order_from_scores(s):
        """Get vertex order by descending score."""
        return np.argsort(-s)

    # Initial order: low degree first (greedy MIS heuristic)
    # We assign initial scores so that low-degree vertices come first
    # Use negative degree as initial score
    scores = -degrees.copy()
    
    current_order = order_from_scores(scores)
    current_size = greedy_size_from_order(current_order)
    best_size = current_size
    best_scores = scores.copy()
    
    # Strategy: Deterministic combinatorial "peeling" algorithm
    # This is a structural change from simulated annealing to a deterministic
    # approach that uses a peeling/layering method to identify a good ordering.
    
    # The idea is to iteratively peel off vertices with the lowest "effective degree"
    # where effective degree accounts for the remaining graph structure.
    
    # We'll use a priority queue to efficiently select the next vertex to peel.
    # This is similar to the degeneracy ordering but adapted for MIS.
    
    # First, let's try a deterministic peeling approach
    remaining = set(range(n))
    order = []
    
    # Use a min-heap based on current degree in the remaining subgraph
    # We need to track degrees dynamically
    
    # For efficiency, we'll use a lazy deletion approach with the heap
    import heapq as hq
    
    # Initialize heap with (degree, vertex)
    heap = [(degrees[v], v) for v in range(n)]
    hq.heapify(heap)
    
    # Track current degree for each vertex in the remaining graph
    curr_degree = degrees.copy()
    in_heap = np.ones(n, dtype=bool)
    
    peeled = np.zeros(n, dtype=bool)
    
    while remaining:
        # Pop the vertex with the smallest current degree
        while heap:
            d, v = hq.heappop(heap)
            if v in remaining and not peeled[v]:
                break
        else:
            break
        
        # Peel this vertex
        order.append(v)
        peeled[v] = True
        remaining.discard(v)
        
        # Update degrees of neighbors
        for u in neighbor_indices[v]:
            if u in remaining:
                curr_degree[u] -= 1
                if curr_degree[u] >= 0:
                    hq.heappush(heap, (curr_degree[u], u))
    
    # Now we have a peeling order. Let's convert it to scores.
    # The peeling order itself is a good starting point.
    # But we can also try to refine it.
    
    peel_order = np.array(order)
    peel_size = greedy_size_from_order(peel_order)
    
    if peel_size > best_size:
        best_size = peel_size
        # Convert order to scores: earlier in order = higher score
        best_scores = np.zeros(n)
        for rank, v in enumerate(peel_order):
            best_scores[v] = n - rank
    
    # Now let's try a different deterministic approach: 
    # Use the concept of "coreness" or k-core decomposition
    # Vertices in lower cores are less connected and might be better candidates
    
    # k-core decomposition: iteratively remove vertices with degree < k
    # This gives a natural ordering
    
    # Let's try multiple deterministic heuristics and pick the best
    
    # Heuristic 1: Low degree first (already have)
    # Heuristic 2: Peeling order (just computed)
    # Heuristic 3: Random restarts with local improvement
    
    # For now, let's also try a simple local search on the best order found
    # But make it deterministic: try all single swaps and keep the best
    
    # Actually, let's try a different structural approach:
    # Use the eigenvector centrality or spectral methods
    
    # Compute the adjacency matrix eigenvalues and use the smallest eigenvector
    # Vertices with large components in the smallest eigenvector are "peripheral"
    
    try:
        # Use scipy if available, otherwise fall back
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        
        A = csr_matrix(adj_bool)
        # Get the smallest eigenvalue and its eigenvector
        # For a connected graph, the smallest eigenvalue of the adjacency matrix
        # has an eigenvector that can help identify peripheral vertices
        
        # Actually, let's use a simpler approach: 
        # Compute the Fiedler vector (second smallest eigenvector of Laplacian)
        # This can help identify communities/peripheral nodes
        
        # For now, let's just use a simple spectral method
        # Use numpy's eig on the adjacency matrix (for n=200, this is fast)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(adj)
        
        # The smallest eigenvalue's eigenvector
        idx = np.argmin(eigenvalues)
        fiedler_vec = np.abs(eigenvectors[:, idx])
        
        # Higher fiedler_vec value = more peripheral = better for MIS
        spectral_scores = fiedler_vec.copy()
        
        spectral_order = order_from_scores(spectral_scores)
        spectral_size = greedy_size_from_order(spectral_order)
        
        if spectral_size > best_size:
            best_size = spectral_size
            best_scores = spectral_scores.copy()
    except Exception:
        pass
    
    # Now let's do a final local improvement pass on the best order
    # Try single swaps and keep the best
    
    best_order = order_from_scores(best_scores)
    best_order_size = greedy_size_from_order(best_order)
    
    # Try a limited number of deterministic local improvements
    # We'll try swapping each pair of adjacent elements in the order
    # and keep the best result
    
    improved = True
    max_improvement_rounds = 50
    
    for _ in range(max_improvement_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            # Try swapping i and i+1
            old_i, old_j = best_order[i], best_order[i + 1]
            best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]
            
            new_size = greedy_size_from_order(best_order)
            
            if new_size > best_order_size:
                best_order_size = new_size
                improved = True
                if best_order_size > best_size:
                    best_size = best_order_size
                    # Update best scores to reflect this order
                    best_scores = np.zeros(n)
                    for rank, v in enumerate(best_order):
                        best_scores[v] = n - rank
            else:
                # Revert
                best_order[i], best_order[i + 1] = old_i, old_j
    
    # Final check: ensure we return the scores corresponding to the best order found
    return best_scores.astype(float)

# EVOLVE-BLOCK-END
