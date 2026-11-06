"""Maximum Cut via fast vectorized local search with spectral init, 1-flip and 2-flip moves, SA escape, and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses fast vectorized greedy local search, 2-flip moves to escape local optima,
    simulated annealing, multiple restarts, and an explicit deadline guard.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Deadline: return best result before 140s (10s safety margin from 150s kill)
    deadline = time.time() + 140.0
    start_time = time.time()

    # Precompute degree for gain calculation
    degree = adj.sum(axis=1)
    
    # Spectral initialization: compute Fiedler vector for a good starting point
    def spectral_init(seed_offset=0):
        """Generate initial labeling using the Fiedler vector (2nd smallest eigenvalue of Laplacian)."""
        try:
            # Build sparse Laplacian
            L = laplacian(adj.astype(np.float64))
            # Get the second smallest eigenvalue and its eigenvector
            # Use sigma shift for efficiency
            if n > 2:
                vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
                # The second smallest eigenvalue corresponds to the Fiedler vector
                # eigsh with sigma=0, which='LM' finds eigenvalues closest to 0
                # Sort by eigenvalue to get the second smallest
                idx = np.argsort(vals)
                fiedler_vec = vecs[:, idx[1]]
                # Try multiple threshold strategies to find best split
                best_labels = None
                best_score = -1
                
                # Strategy 1: zero threshold
                labels1 = (fiedler_vec >= 0).astype(np.int8)
                score1 = compute_cut(labels1)
                if score1 > best_score:
                    best_score = score1
                    best_labels = labels1.copy()
                
                # Strategy 2: mean threshold
                mean_val = np.mean(fiedler_vec)
                labels2 = (fiedler_vec >= mean_val).astype(np.int8)
                score2 = compute_cut(labels2)
                if score2 > best_score:
                    best_score = score2
                    best_labels = labels2.copy()
                
                # Strategy 3: median threshold
                median_val = np.median(fiedler_vec)
                labels3 = (fiedler_vec >= median_val).astype(np.int8)
                score3 = compute_cut(labels3)
                if score3 > best_score:
                    best_score = score3
                    best_labels = labels3.copy()
                
                # Strategy 4: balanced split by rank
                sorted_idx = np.argsort(fiedler_vec)
                labels4 = np.zeros(n, dtype=np.int8)
                labels4[sorted_idx[n//2:]] = 1
                score4 = compute_cut(labels4)
                if score4 > best_score:
                    best_score = score4
                    best_labels = labels4.copy()
                
                # Strategy 5: multiple quantile thresholds
                for q in [0.4, 0.6]:
                    thresh = np.quantile(fiedler_vec, q)
                    labels_q = (fiedler_vec >= thresh).astype(np.int8)
                    score_q = compute_cut(labels_q)
                    if score_q > best_score:
                        best_score = score_q
                        best_labels = labels_q.copy()
                
                return best_labels
            else:
                labels = np.zeros(n, dtype=np.int8)
        except Exception:
            # Fallback to random if spectral fails
            labels = np.random.randint(0, 2, size=n).astype(np.int8)
        return labels

    def compute_cut(labels):
        """Number of edges crossing the partition. Fast vectorized version."""
        # cut * 2 = sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # Use: (labels[None,:] != labels[:,None]) creates n x n bool matrix
        diff = (labels[None, :] != labels[:, None])
        cut2 = (adj * diff).sum()
        return cut2 // 2

    def fast_local_search(labels, max_rounds=200):
        """
        Fast greedy 1-flip local search.
        Key insight: gain[v] = 2 * (num neighbors with same label as v) - degree[v]
        We maintain same_count incrementally for speed.
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count for each vertex
        # same_count[v] = number of neighbors of v with same label as v
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_v = np.argmax(gains)
            best_gain = gains[best_v]
            
            if best_gain <= 0:
                break
            
            # Flip vertex best_v
            old_label = labels[best_v]
            labels[best_v] = 1 - old_label
            current_cut += best_gain
            
            # Update gains for neighbors of best_v
            # When we flip v, the same_count for each neighbor u changes:
            # If u and v had same label, after flip they differ -> same_count[u] decreases by 1
            # If u and v had different labels, after flip they same -> same_count[u] increases by 1
            # So: same_count[u] += (1 if labels[u] == new_label else -1) for each neighbor u
            # new_label = 1 - old_label
            # For neighbor u: if labels[u] == new_label, same_count[u] += 1; else same_count[u] -= 1
            # This is equivalent to: same_count[u] += (2 * labels[u] - 1) * (2 * new_label - 1) ... 
            # Simpler: for each neighbor u of v, same_count[u] changes by +1 if labels[u] == new_label else -1
            # = (2*labels[u] - 1) * (2*new_label - 1) ... no
            # Let's just compute: delta_same[u] = 1 if labels[u] == new_label else -1
            # = 2*labels[u] - 1 if new_label == 1, or -(2*labels[u] - 1) if new_label == 0
            # = (2*labels[u] - 1) * (2*new_label - 1)
            # Actually: if new_label == 1: delta = 1 if labels[u]==1 else -1 = 2*labels[u] - 1
            # if new_label == 0: delta = 1 if labels[u]==0 else -1 = -(2*labels[u] - 1)
            # So delta = (2*labels[u] - 1) * (2*new_label - 1)
            
            new_label = 1 - old_label
            # Get neighbors of best_v
            neighbors = np.where(adj[best_v] == 1)[0]
            if len(neighbors) > 0:
                # For each neighbor, compute delta to same_count
                # delta_same[u] = 1 if labels[u] == new_label else -1
                # = 2*labels[u] - 1 if new_label == 1, else -(2*labels[u] - 1)
                # = (2*labels[u] - 1) * (2*new_label - 1)
                sign = 2 * new_label - 1  # 1 if new_label=1, -1 if new_label=0
                delta_same = (2 * labels[neighbors] - 1) * sign
                same_count[neighbors] += delta_same
                gains[neighbors] += 2 * delta_same
            
            # Also update gain for best_v itself (it's now flipped)
            # same_count[best_v] should be recomputed or updated
            # After flip, same_count[best_v] = degree[best_v] - old_same_count[best_v]
            # Because neighbors that were same are now different, and vice versa
            same_count[best_v] = degree[best_v] - same_count[best_v]
            gains[best_v] = 2 * same_count[best_v] - degree[best_v]
        
        return labels, current_cut

    def two_flip_local_search(labels, max_rounds=50):
        """
        Vectorized 2-flip local search.
        Gain of flipping i and j together:
        gain_ij = gain_i + gain_j + 2*adj[i,j]*(2*labels[i]-1)*(2*labels[j]-1)
        where gain_i = 2*same_count[i] - degree[i]
        """
        current_cut = compute_cut(labels)
        
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree
        signs = 2 * labels.astype(np.int8) - 1  # +1 for label 1, -1 for label 0
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Compute pairwise 2-flip gains vectorized
            # gain_matrix[i,j] = gains[i] + gains[j] + 2*adj[i,j]*signs[i]*signs[j]
            gain_matrix = gains[:, None] + gains[None, :] + 2 * adj * np.outer(signs, signs)
            
            # Find best pair
            i_idx, j_idx = np.unravel_index(np.argmax(gain_matrix), gain_matrix.shape)
            best_2flip_gain = gain_matrix[i_idx, j_idx]
            
            if best_2flip_gain <= 0:
                break
            
            # Apply the flip
            old_li = labels[i_idx]
            old_lj = labels[j_idx]
            labels[i_idx] = 1 - old_li
            labels[j_idx] = 1 - old_lj
            current_cut += best_2flip_gain
            
            # Recompute same_count and gains (relatively cheap for n=100)
            same_mask = (labels[None, :] == labels[:, None])
            same_count = (adj * same_mask).sum(axis=1)
            gains = 2 * same_count - degree
            signs = 2 * labels.astype(np.int8) - 1
        
        return labels, current_cut

    def simulated_annealing(labels, initial_temp=15.0, cooling=0.9995, max_iters=20000):
        """
        Simulated Annealing with 1-flip moves.
        Uses incremental gain computation for speed.
        """
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Precompute same_count
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        
        temp = initial_temp
        no_improve_count = 0
        rng_sa = np.random.RandomState()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random vertex
            v = rng_sa.randint(n)
            
            # Compute gain of flipping v
            same_count_v = same_count[v]
            gain = 2 * same_count_v - degree[v]
            
            # Accept if gain > 0 or with probability exp(gain / temp) for gain < 0
            accepted = False
            if gain >= 0:
                accepted = True
            else:
                if rng_sa.rand() < np.exp(gain / temp):
                    accepted = True
            
            if accepted:
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gain
                
                # Update same_count for neighbors
                new_label = 1 - old_label
                neighbors = np.where(adj[v] == 1)[0]
                if len(neighbors) > 0:
                    sign = 2 * new_label - 1
                    delta_same = (2 * labels[neighbors] - 1) * sign
                    same_count[neighbors] += delta_same
                
                # Update same_count for v
                same_count[v] = degree[v] - same_count[v]
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            temp *= cooling
            
            # Adaptive cooling: cool faster if stuck
            if no_improve_count > 500:
                temp *= 0.9
                no_improve_count = 0
        
        return best_labels, best_cut

    # Multiple restarts with different initializations
    best_labels_overall = None
    best_cut_overall = -1
    rng = np.random.RandomState(42)
    
    def try_restart(labels_init, tag=""):
        nonlocal best_labels_overall, best_cut_overall
        if time.time() > deadline:
            return
        
        labels = labels_init.copy()
        # Fast local search
        labels, cut = fast_local_search(labels, max_rounds=300)
        
        # Try 2-flip escape
        if time.time() < deadline - 5:
            labels2, cut2 = two_flip_local_search(labels, max_rounds=20)
            if cut2 > cut:
                labels, cut = labels2, cut2
                # Polish after 2-flip
                labels, cut = fast_local_search(labels, max_rounds=100)
        
        if cut > best_cut_overall:
            best_cut_overall = cut
            best_labels_overall = labels.copy()
    
    # Restart 1: Spectral initialization (primary, uses global structure)
    labels_spectral = spectral_init()
    try_restart(labels_spectral, "spectral")
    
    # Restart 2: alternating
    try_restart(np.arange(n) % 2, "alt")
    
    # Restart 3: random
    try_restart(rng.randint(0, 2, size=n), "rand")
    
    # Restart 4: Spectral with sign flip (equivalent but tests robustness)
    labels_spectral2 = 1 - labels_spectral
    try_restart(labels_spectral2, "spectral_flip")
    
    # Restart 5: random with bias
    try_restart(rng.randint(0, 2, size=n), "rand2")
    
    # Iterated Local Search: perturb best solution and re-optimize
    if best_labels_overall is not None:
        ils_rounds = 12
        k_values = [3, 5, 8, 10, 15, 20]
        for ils_iter in range(ils_rounds):
            if time.time() > deadline - 10:
                break
            
            # Vary perturbation size
            k = k_values[ils_iter % len(k_values)]
            k = min(k, n - 1)
            flip_idx = rng.choice(n, size=k, replace=False)
            perturbed = best_labels_overall.copy()
            perturbed[flip_idx] = 1 - perturbed[flip_idx]
            
            # Re-optimize from perturbed state
            labels_ils = perturbed.copy()
            labels_ils, cut_ils = fast_local_search(labels_ils, max_rounds=200)
            
            # Try 2-flip escape
            if time.time() < deadline - 5:
                labels_ils2, cut_ils2 = two_flip_local_search(labels_ils, max_rounds=20)
                if cut_ils2 > cut_ils:
                    labels_ils, cut_ils = labels_ils2, cut_ils2
                    labels_ils, cut_ils = fast_local_search(labels_ils, max_rounds=100)
            
            if cut_ils > best_cut_overall:
                best_cut_overall = cut_ils
                best_labels_overall = labels_ils.copy()
    
    # If we have time, do SA from best solution
    if best_labels_overall is not None and time.time() < deadline - 10:
        labels_sa = best_labels_overall.copy()
        labels_sa, cut_sa = simulated_annealing(labels_sa, initial_temp=20.0, cooling=0.999, max_iters=30000)
        
        # Polish
        labels_sa, cut_sa = fast_local_search(labels_sa, max_rounds=200)
        
        if cut_sa > best_cut_overall:
            best_cut_overall = cut_sa
            best_labels_overall = labels_sa.copy()
    
    # Additional SA pass from spectral init if still have time
    if time.time() < deadline - 15:
        labels_sa2 = spectral_init()
        labels_sa2, cut_sa2 = simulated_annealing(labels_sa2, initial_temp=25.0, cooling=0.998, max_iters=40000)
        labels_sa2, cut_sa2 = fast_local_search(labels_sa2, max_rounds=200)
        
        if cut_sa2 > best_cut_overall:
            best_cut_overall = cut_sa2
            best_labels_overall = labels_sa2.copy()
    
    # Final high-precision polish: multiple passes of local search + 2-flip
    if best_labels_overall is not None and time.time() < deadline - 2:
        labels_final = best_labels_overall.copy()
        
        # Pass 1: standard local search
        labels_final, cut_final = fast_local_search(labels_final, max_rounds=300)
        
        # Pass 2: 2-flip escape (vectorized, so can afford more rounds)
        if time.time() < deadline - 2:
            labels_final, cut_final = two_flip_local_search(labels_final, max_rounds=50)
        
        # Pass 3: final local search polish
        if time.time() < deadline - 1:
            labels_final, cut_final = fast_local_search(labels_final, max_rounds=500)
        
        # Pass 4: one more 2-flip sweep after polish
        if time.time() < deadline - 0.5:
            labels_final, cut_final = two_flip_local_search(labels_final, max_rounds=20)
        
        if cut_final > best_cut_overall:
            best_cut_overall = cut_final
            best_labels_overall = labels_final.copy()
    
    return best_labels_overall
# EVOLVE-BLOCK-END
