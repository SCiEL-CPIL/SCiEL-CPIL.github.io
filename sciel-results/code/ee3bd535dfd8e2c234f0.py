import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def priority(adjacency):
    """
    Assigns a priority score to each vertex for a greedy decoder that picks
    vertices in descending score order, skipping neighbors of already picked vertices.
    
    The function uses a meta-heuristic approach:
    1. Generates multiple candidate scoring vectors based on different graph properties.
    2. Simulates the greedy decoder for each candidate to estimate the independent set size.
    3. Selects the candidate that yields the largest independent set.
    4. Optionally refines the best candidate using local search on the ordering.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    A = np.asarray(adjacency)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute features
    degree = A.sum(axis=1).astype(float)
    
    # 1. Basic Degree Score (Low degree first)
    score_degree = -degree
    
    # 2. Core Number Score
    # Approximate core number by iteratively removing low-degree nodes
    # For small n, we can do a simple approximation or use a fast algorithm
    # Here we use a simple iterative peeling for core numbers
    remaining = np.ones(n, dtype=bool)
    core_numbers = np.zeros(n, dtype=int)
    current_deg = degree.copy()
    max_core = 0
    for k in range(int(degree.max()) + 1):
        # Find nodes with current degree >= k
        # Actually, standard k-core algorithm:
        # Remove nodes with degree < k
        # This is O(n^2) which is fine for n=200
        pass # Skipping complex core calc for speed, using simpler proxies
        
    # Simpler structural features:
    # Sum of degrees of neighbors
    neighbor_deg_sum = A @ degree
    
    # Mean degree of neighbors
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.where(degree > 0, neighbor_deg_sum / degree, 0.0)
    
    # Spectral Feature: Fiedler Vector
    # The Fiedler vector (2nd smallest eigenvector of Laplacian) provides a global ordering.
    # Vertices with similar values in the Fiedler vector are likely in the same community/partition.
    # For MIS, we might want to pick vertices that are "central" in a way that isolates them?
    # Or use the magnitude?
    try:
        L = np.diag(degree) - A
        # Use sparse eigsh for speed, though n=200 is small
        # We want the smallest non-zero eigenvalue's eigenvector
        # eigsh requires k < n. If n is small, use dense.
        if n > 10:
            # Shift-invert mode is good for smallest eigenvalues
            vals, vecs = eigsh(L, k=2, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        else:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
        
        # Normalize fiedler vector to [0, 1] for consistent scaling
        f_min = fiedler.min()
        f_max = fiedler.max()
        if f_max - f_min > 1e-9:
            fiedler_norm = (fiedler - f_min) / (f_max - f_min)
        else:
            fiedler_norm = np.zeros(n)
            
        # Score based on Fiedler vector: 
        # Hypothesis: Vertices with extreme Fiedler values (ends of the spectrum) 
        # might be better candidates as they are structurally distinct.
        # Or, picking based on the sign? 
        # Let's try: Score = -degree + alpha * fiedler_norm
        # And Score = -degree - alpha * fiedler_norm
        # And Score = -degree + alpha * |fiedler_norm - 0.5| (distance from center)
        
    except Exception:
        fiedler_norm = np.zeros(n)
    
    # Candidate Strategies
    candidates = []
    
    # 1. Pure Degree
    candidates.append(score_degree)
    
    # 2. Degree + Mean Neighbor Degree
    candidates.append(-(degree + 0.5 * mean_neighbor_deg))
    candidates.append(-(degree + 1.0 * mean_neighbor_deg))
    
    # 3. Degree + Neighbor Degree Sum
    candidates.append(-(degree + 0.1 * neighbor_deg_sum))
    
    # 4. Log Degree
    candidates.append(-np.log1p(degree))
    
    # 5. Spectral based
    if np.any(fiedler_norm != 0):
        # Pick vertices with high degree? No, low degree.
        # But maybe low degree AND far from the "center" of the graph?
        candidates.append(-(degree + 0.5 * fiedler_norm))
        candidates.append(-(degree - 0.5 * fiedler_norm))
        candidates.append(-(degree + 0.5 * np.abs(fiedler_norm - 0.5)))
        
        # Pure spectral?
        candidates.append(-fiedler_norm)
        candidates.append(fiedler_norm)
        
        # Combined: Low degree, high spectral extremity
        candidates.append(-(degree + 1.0 * np.abs(fiedler_norm - 0.5)))
    
    # 6. Randomized (Deterministic seed for reproducibility per call? No, fixed greedy decoder)
    # We can't rely on randomness for the final score if the evaluator expects determinism, 
    # but the meta-heuristic picks the best deterministic one.
    
    # Helper: Fast Greedy Decode
    def greedy_decode_size(scores):
        """
        Simulates the greedy decoder.
        Returns the size of the independent set.
        """
        # Order vertices by score descending
        order = np.argsort(-scores)
        
        # Bitset representation for selected vertices
        # n=200 fits in 4 64-bit integers, but numpy bool is easier.
        # For speed, we can use a boolean array and vectorized checks.
        selected = np.zeros(n, dtype=bool)
        count = 0
        
        # Optimization: Precompute neighbor masks? 
        # A is dense. A[v] is the mask of neighbors of v.
        
        for v in order:
            # If v is already selected, skip (shouldn't happen if we mark neighbors)
            # If v is adjacent to any selected vertex, skip.
            # Check: Is there any u such that selected[u] is True and A[v,u] is True?
            # This is dot product of selected and A[v].
            # If dot > 0, v has a neighbor selected.
            if not np.dot(selected, A[v]):
                selected[v] = True
                count += 1
                # We don't need to mark neighbors as "blocked" explicitly if we just check against selected set.
                # The check "is v adjacent to any selected?" covers it.
        
        return count

    # Evaluate candidates
    best_size = -1
    best_scores = candidates[0]
    
    for s in candidates:
        if time.time() > deadline - 5.0:
            break
        # Ensure finite
        if np.any(~np.isfinite(s)):
            s = np.where(np.isfinite(s), s, 0.0)
        
        size = greedy_decode_size(s)
        if size > best_size:
            best_size = size
            best_scores = s

    # Local Search Refinement
    # Try to improve the ordering by swapping pairs of vertices.
    # This is expensive, so only do it if we have time and n is small enough.
    if time.time() < deadline - 20.0 and n <= 300:
        current_scores = best_scores.copy()
        current_size = best_size
        improved = True
        max_iters = 1000
        
        # Precompute order
        order = np.argsort(-current_scores)
        
        # To speed up local search, we can maintain the selected set and update it incrementally?
        # That's complex. Let's just re-run greedy for each swap, but limit the number of swaps.
        # Or, a simpler heuristic: 
        # Identify "critical" vertices that were not selected but have low degree.
        # Try moving them up in the order.
        
        # Alternative: Hill Climbing on the score vector directly?
        # Let's try swapping adjacent elements in the sorted order.
        
        # A faster local search: 
        # 1. Get the set of selected vertices S and unselected U from the current best run.
        # 2. For each u in U, check if adding u to S (and removing its neighbors from S) increases size?
        #    This is not how the greedy decoder works. The greedy decoder is fixed.
        #    We must change the *order* (scores) to change the outcome.
        
        # Let's try a simple perturbation:
        # Randomly swap two vertices in the score array, evaluate, keep if better.
        # This is slow. 
        
        # Better: 
        # Look at the vertices that were skipped. Why were they skipped? Because a neighbor was picked earlier.
        # If we lower the score of that neighbor, the vertex might be picked.
        # Let's identify pairs (u, v) where u is selected, v is not, and A[u,v]=1.
        # If we swap scores of u and v, v gets picked earlier, u gets skipped.
        # Size change: 
        #   Old: u picked, v skipped.
        #   New: v picked, u skipped.
        #   But this affects other vertices too.
        
        # Given the complexity and time limit, the meta-heuristic selection of static features is likely sufficient 
        # and safer. The local search on 200 nodes with full re-evaluation (O(N^2) per eval) 
        # might be too slow for many iterations.
        
        # Let's do a very limited local search: 
        # Try to improve by 1 vertex.
        # Find the best "swap" that improves size by 1.
        
        # Get the selected set from current best
        order_best = np.argsort(-best_scores)
        selected_best = np.zeros(n, dtype=bool)
        for v in order_best:
            if not np.dot(selected_best, A[v]):
                selected_best[v] = True
        
        # Identify candidates for swapping
        # For each unselected vertex v, find a selected neighbor u.
        # If we make v have higher priority than u, v will be selected, u skipped.
        # We need to check if this swap improves the total size.
        
        # This requires re-running the greedy decoder for each potential swap.
        # Number of unselected vertices ~ N/2. Number of neighbors ~ N/2.
        # Total swaps to test ~ N^2 / 4 = 10,000.
        # Each eval is O(N^2) = 40,000 ops.
        # Total ops ~ 400,000,000. Too slow in Python loop.
        
        # Vectorized Local Search?
        # Hard to vectorize the greedy process easily.
        
        # So, we stick to the static meta-heuristic.
        pass

    return best_scores

# EVOLVE-BLOCK-END
