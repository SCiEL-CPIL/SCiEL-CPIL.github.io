"""
Optimized Greedy MIS via Local Search on Vertex Scores.

This program generates priority scores for a greedy decoder.
It uses a two-phase approach:
1. Generate diverse initial score vectors based on graph structural features (Degree, Clustering, Spectral).
2. Perform local search (swaps and perturbations) on the score vectors to maximize the resulting independent set size, 
   using the greedy decoder as the objective function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 125.0  # Safety margin below 150s limit, allowing ~10s for cleanup

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists for faster greedy evaluation
    # Using lists of indices is faster than boolean slicing in a loop for N=200
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    # Also store neighbor sets for O(1) membership checks if needed
    neighbor_sets = [set(n.tolist()) for n in neighbors]

    # --- Fast Greedy Evaluator ---
    def evaluate(scores):
        """
        Simulates the greedy decoder:
        1. Sort vertices by score descending.
        2. Iterate through sorted list.
        3. If vertex is not blocked (not neighbor of any picked vertex), pick it.
        4. Mark it and its neighbors as blocked.
        Returns the count of picked vertices.
        """
        order = np.argsort(-scores, kind='stable')
        
        blocked = np.zeros(N, dtype=np.uint8)
        count = 0
        
        for v in order:
            if blocked[v] == 0:
                count += 1
                blocked[v] = 1
                if neighbors[v].size > 0:
                    blocked[neighbors[v]] = 1
        
        return count

    # --- Feature Extraction for Initial Candidates ---
    
    # 1. Degree
    D = A.sum(axis=1)
    
    # 2. Clustering Coefficient (approximated or exact)
    # For N=200, exact triangle count is feasible
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0
    D_safe = np.where(D > 0, D, 1.0)
    # Standard clustering coefficient: 2*Triangles / (deg*(deg-1))
    # Avoid division by zero
    Clus = np.zeros(N)
    mask = D >= 2
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 3. Mean Neighbor Degree
    SND = A @ D
    MeanND = SND / D_safe

    # 4. Spectral Centrality (Principal Eigenvector)
    # Vertices with high spectral centrality are central. We want low centrality for independent set.
    x = np.ones(N) / N
    for _ in range(30):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < 1e-9:
            break
        x = x_new
    Spectral = x

    # 5. Fiedler Vector (second-smallest eigenvector of Laplacian)
    # Identifies vertices minimally entangled in graph connectivity
    # High absolute Fiedler values indicate vertices on periphery/sparse regions
    L = np.diag(D) - A
    try:
        # Use scipy for efficient eigenvalue computation
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        L_sparse = csr_matrix(L)
        # Get 3 smallest eigenvalues to ensure we get the Fiedler vector
        # (eigsh requires k < N for 'SM' target)
        k_eig = min(3, N - 1)
        eigvals, eigvecs = eigsh(L_sparse, k=k_eig, which='SM')
        # Sort by eigenvalue
        idx_sort = np.argsort(eigvals)
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]
        # Fiedler vector is the second smallest (index 1)
        if len(eigvals) >= 2:
            Fiedler = np.abs(eigvecs[:, 1])
        else:
            # Fallback to principal eigenvector if only one mode found
            Fiedler = Spectral
    except Exception:
        # Fallback: use spectral as proxy
        Fiedler = Spectral

    # Normalize features to [0, 1] for consistent weighting
    def norm(x):
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    Clus_n = norm(Clus)
    MeanND_n = norm(MeanND)
    Spectral_n = norm(Spectral)
    Fiedler_n = norm(Fiedler)
    
    # "Openness": Vertices with many neighbors that are NOT connected to each other.
    # This suggests the vertex is in a sparse region.
    # OpenPairs = C(D, 2) - Tri
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1) - Tri)
    Open_n = norm(OpenPairs)

    # --- Phase 1: Initialize with Strong Heuristics ---
    
    candidates = []
    
    # H1: Min Degree (Classic)
    candidates.append(-D_n)
    
    # H2: Min Degree, High Openness
    candidates.append(-D_n + 0.5 * Open_n)
    
    # H3: Min Degree, Low Clustering
    candidates.append(-D_n - 0.5 * Clus_n)
    
    # H4: Min Degree, Low Mean Neighbor Degree (prefer peripheral)
    candidates.append(-D_n - 0.5 * MeanND_n)
    
    # H5: Low Spectral Centrality
    candidates.append(-Spectral_n)
    
    # H6: Mix of Degree and Spectral
    candidates.append(-0.7 * D_n - 0.3 * Spectral_n)
    
    # H7: High Fiedler (peripheral/sparse regions)
    candidates.append(Fiedler_n)
    
    # H8: Min Degree, High Fiedler
    candidates.append(-0.5 * D_n + 0.5 * Fiedler_n)
    
    # H9: Min Degree, High Fiedler, Low Clustering
    candidates.append(-0.4 * D_n + 0.4 * Fiedler_n - 0.2 * Clus_n)
    
    # H10: Random initialization
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))
    
    # H11: Min Degree, High Openness, Low Clustering
    candidates.append(-0.6 * D_n + 0.3 * Open_n - 0.3 * Clus_n)
    
    # H12: Min Mean Neighbor Degree (prefer vertices whose neighbors are also peripheral)
    candidates.append(-MeanND_n)
    
    # H13: Min Degree squared (stronger preference for low degree)
    D_sq = D * D
    D_sq_n = norm(D_sq)
    candidates.append(-D_sq_n)
    
    # H14: Min Degree + Min MeanND
    candidates.append(-0.5 * D_n - 0.5 * MeanND_n)
    
    best_size = -1
    best_scores = None

    for c in candidates:
        size = evaluate(c)
        if size > best_size:
            best_size = size
            best_scores = c.copy()
            
    # --- Phase 2: Local Search ---
    # Strategy:
    # 1. Try swapping scores of two random vertices.
    # 2. Try adding small Gaussian noise.
    # 3. Occasionally revert to a base heuristic + noise to escape local optima.

    # We can do many iterations in 120s. N=200.
    # Each eval is ~200 argsorts + 200 checks. Very fast.
    
    trial_count = 0
    max_trials = 100000 # Increased budget with faster evaluation
    
    # Precompute current order for reuse
    current_order = np.argsort(-best_scores, kind='stable')
    
    while trial_count < max_trials:
        if time.time() > DEADLINE:
            break
            
        # Decide move type
        r = rng.random()
        new_scores = best_scores.copy()
        
        if r < 0.35:
            # Swap move: Pick two random indices and swap their scores.
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i != j:
                new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
            else:
                continue # No change
        elif r < 0.60:
            # 2-opt neighborhood: Swap two vertices that are close in ranking
            # This is a more structured move than random swap
            order = np.argsort(-best_scores, kind='stable')
            # Pick a random position in the ranking
            pos = rng.integers(0, N - 1)
            # Swap vertices at pos and pos+1 (adjacent in ranking)
            i = order[pos]
            j = order[pos + 1]
            new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
        elif r < 0.78:
            # Targeted swap: Swap a high-ranked vertex with a low-ranked one
            # This can unlock blocked vertices
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j:
                continue
            # Prefer swapping vertices with different "status" in greedy
            new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
        elif r < 0.90:
            # Perturbation move: Add small noise to a few random vertices
            k = rng.integers(1, 4) # Perturb 1 to 3 vertices
            idxs = rng.choice(N, k, replace=False)
            noise = rng.normal(0, 0.02, k)
            new_scores[idxs] += noise
        else:
            # Restart/Escape: Blend with a random base heuristic including Fiedler
            w_d = rng.uniform(-1, 0)
            w_c = rng.uniform(-0.5, 0.5)
            w_s = rng.uniform(-0.5, 0.5)
            w_f = rng.uniform(0, 0.5)
            base = w_d * D_n + w_c * Clus_n + w_s * Spectral_n + w_f * Fiedler_n
            # Blend with current best to stay in the neighborhood
            alpha = rng.uniform(0.5, 0.85)
            new_scores = alpha * best_scores + (1 - alpha) * base
            
        size = evaluate(new_scores)
        
        if size > best_size:
            best_size = size
            best_scores = new_scores
            # Reset trial counter for success-based patience
            trial_count = 0
            # Update order for next iteration
            current_order = np.argsort(-best_scores, kind='stable')
        else:
            trial_count += 1
            
        # Early exit if stuck for too many trials
        if trial_count > 2500:
            # Diversify: jump to a new random neighborhood
            base_heuristics = [
                -D_n,
                -0.5 * D_n + 0.5 * Fiedler_n,
                Fiedler_n,
                -0.7 * D_n - 0.3 * Spectral_n,
                -0.3 * D_n + 0.3 * Fiedler_n - 0.2 * Clus_n,
                -0.6 * D_n + 0.4 * Open_n
            ]
            base = base_heuristics[rng.integers(0, len(base_heuristics))]
            alpha = rng.uniform(0.3, 0.7)
            best_scores = alpha * best_scores + (1 - alpha) * base
            best_size = evaluate(best_scores)
            current_order = np.argsort(-best_scores, kind='stable')
            trial_count = 0

    # --- Phase 3: Final Refinement (if time permits) ---
    if time.time() < DEADLINE - 5.0 and best_scores is not None:
        # Try very small swaps around the best solution
        # Focus on adjacent-in-ranking swaps for more structured exploration
        # Also try swapping vertices that are neighbors of each other in the graph
        # to potentially unblock one while blocking the other
        
        for _ in range(40000):
            if time.time() > DEADLINE - 2.0:
                break
            
            # Use 2-opt style: swap adjacent vertices in ranking
            order = np.argsort(-best_scores, kind='stable')
            pos = rng.integers(0, N - 1)
            i = order[pos]
            j = order[pos + 1]
            
            # Store old values
            si = best_scores[i]
            sj = best_scores[j]
            
            # Swap
            best_scores[i] = sj
            best_scores[j] = si
            
            size = evaluate(best_scores)
            
            if size > best_size:
                best_size = size
                # Keep swap
            else:
                # Revert swap
                best_scores[i] = si
                best_scores[j] = sj
                
            # Also try swapping a vertex with one of its neighbors
            # This can be more effective for independent set problems
            if rng.random() < 0.3:
                # Pick a random vertex and swap with a random neighbor
                v = rng.integers(0, N)
                if neighbors[v].size > 0:
                    u = neighbors[v][rng.integers(0, neighbors[v].size)]
                    sv = best_scores[v]
                    su = best_scores[u]
                    best_scores[v] = su
                    best_scores[u] = sv
                    size2 = evaluate(best_scores)
                    if size2 > best_size:
                        best_size = size2
                    else:
                        best_scores[v] = sv
                        best_scores[u] = su
                
    if best_scores is None:
        return -D.astype(float)

    return best_scores
# EVOLVE-BLOCK-END
