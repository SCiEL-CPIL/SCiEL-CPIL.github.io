"""Maximum Cut via Spectral Initialization and Enhanced Local Search with Vectorized 2-Flips"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse.csgraph import laplacian as sp_laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

def partition(adjacency):
    """
    Assigns each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral initialization (Fiedler vector), fast vectorized 1-flip local search,
    vectorized 2-flip local search for escaping local optima, and simulated annealing.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Deadline: return best result before 140s (10s safety margin from 150s kill)
    deadline = time.time() + 140.0

    # Precompute degree for gain calculation
    degree = adj.sum(axis=1)
    
    # Precompute neighbor lists for faster 1-flip updates
    neighbors_list = [np.where(adj[i] == 1)[0] for i in range(n)]

    def compute_cut(labels):
        """Number of edges crossing the partition."""
        diff = (labels[None, :] != labels[:, None])
        cut2 = (adj * diff).sum()
        return cut2 // 2

    def fast_local_search(labels, max_rounds=500):
        """
        Fast greedy 1-flip local search with incremental gain updates.
        """
        current_cut = compute_cut(labels)
        
        # same_count[v] = number of neighbors of v with same label as v
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1).astype(np.int32)
        gains = 2 * same_count - degree.astype(np.int32)
        
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
            new_label = 1 - old_label
            neighbors = neighbors_list[best_v]
            
            if len(neighbors) > 0:
                # delta_same[u] = 1 if labels[u] == new_label else -1
                # = (2*labels[u] - 1) * (2*new_label - 1)
                sign = 2 * new_label - 1
                delta_same = (2 * labels[neighbors] - 1) * sign
                same_count[neighbors] += delta_same
                gains[neighbors] += 2 * delta_same
            
            # Update gain for best_v itself
            # After flip, same_count[best_v] = degree[best_v] - old_same_count[best_v]
            same_count[best_v] = degree[best_v] - same_count[best_v]
            gains[best_v] = 2 * same_count[best_v] - degree[best_v]
        
        return labels, current_cut

    def vectorized_two_flip_search(labels, max_rounds=50):
        """
        Vectorized 2-flip local search.
        Computes the gain of flipping any pair (i, j) and applies the best improving pair.
        """
        current_cut = compute_cut(labels)
        
        # Precompute single-flip gains
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1).astype(np.int32)
        single_gains = 2 * same_count - degree.astype(np.int32)
        
        # Interaction term for 2-flip (i, j):
        # If we flip i and j, the change in cut is:
        # gain_2(i,j) = single_gain[i] + single_gain[j] + interaction(i,j)
        # interaction(i,j) depends on whether i and j are connected and their relative labels.
        # If i and j are not connected, interaction is 0.
        # If i and j are connected:
        #   Before: edge (i,j) contributes to cut if labels[i] != labels[j].
        #   After:  labels flip, so labels[i] != labels[j] becomes (1-labels[i]) != (1-labels[j]), which is still labels[i] != labels[j].
        #   Wait, let's re-derive.
        #   Cut contribution of edge (i,j) is 1 if labels differ, 0 if same.
        #   Let L_i, L_j be labels.
        #   Before: 1 if L_i != L_j.
        #   After: L'_i = 1-L_i, L'_j = 1-L_j.
        #   L'_i != L'_j  <=>  (1-L_i) != (1-L_j)  <=>  -L_i != -L_j  <=>  L_i != L_j.
        #   So the status of the edge (i,j) does NOT change if both endpoints are flipped.
        #   However, the single_gain calculation assumes only one vertex flips.
        #   When we flip i, it affects neighbors of i. j is a neighbor of i.
        #   So the "gain" for i assumes j stays fixed. But j is also flipping.
        #   
        #   Let's use the formula:
        #   Delta_Cut(i, j) = (Sum_{u in N(i)} (L_u != 1-L_i) - (L_u != L_i)) + (Sum_{v in N(j)} (L_v != 1-L_j) - (L_v != L_j)) - (Correction for edge (i,j) counted twice or interaction)
        #   
        #   Actually, simpler:
        #   Cut = 1/2 Sum_{u,v} A_{uv} (L_u != L_v)
        #   Let f(L) be cut size.
        #   f(L') - f(L) where L' has L_i->1-L_i and L_j->1-L_j.
        #   
        #   Let's compute the gain matrix G where G[i,j] is the gain of flipping i and j.
        #   G[i,j] = single_gain[i] + single_gain[j] + 2 * A[i,j] * (1 if L_i == L_j else -1) ??
        #   
        #   Let's test with a small graph.
        #   Path 0-1-2. L = [0, 0, 0]. Cut = 0.
        #   single_gain[0]: flip 0. N(0)={1}. L1=0, L0=0. Before: same. After: diff. Cut increases by 1. Gain = 1.
        #   single_gain[1]: flip 1. N(1)={0,2}. L0=0, L2=0. Before: same, same. After: diff, diff. Cut increases by 2. Gain = 2.
        #   single_gain[2]: Gain = 1.
        #   
        #   Flip 0 and 1. L' = [1, 1, 0].
        #   Edges: (0,1): 1!=1? No. (1,2): 1!=0? Yes. Cut = 1.
        #   Gain = 1 - 0 = 1.
        #   Formula: G[0,1] = single_gain[0] + single_gain[1] + correction.
        #   1 = 1 + 2 + corr => corr = -2.
        #   L0==L1 (0==0). A[0,1]=1.
        #   
        #   Flip 0 and 2. L' = [1, 0, 1].
        #   Edges: (0,1): 1!=0? Yes. (1,2): 0!=1? Yes. Cut = 2.
        #   Gain = 2 - 0 = 2.
        #   Formula: G[0,2] = 1 + 1 + corr.
        #   2 = 2 + corr => corr = 0.
        #   A[0,2]=0.
        #   
        #   So correction is 0 if not connected.
        #   If connected and same label: -2?
        #   If connected and diff label:
        #   Path 0-1-2. L=[0,1,0]. Cut=2.
        #   single_gain[0]: Flip 0. L'=[1,1,0]. Cut=1. Gain = -1.
        #   single_gain[1]: Flip 1. L'=[0,0,0]. Cut=0. Gain = -2.
        #   Flip 0 and 1. L'=[1,0,0]. Cut=1. Gain = 1-2 = -1.
        #   G[0,1] = -1 + -2 + corr. -1 = -3 + corr => corr = 2.
        #   L0!=L1 (0!=1). A[0,1]=1.
        #   
        #   So correction = 2 * A[i,j] * (1 if L_i != L_j else -1).
        #   
        #   Let's verify.
        #   Case 1: Connected, Same. Corr = -2.
        #   Case 2: Connected, Diff. Corr = 2.
        #   Case 3: Not Connected. Corr = 0.
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Compute single gains
            same_mask = (labels[None, :] == labels[:, None])
            same_count = (adj * same_mask).sum(axis=1).astype(np.int32)
            single_gains = 2 * same_count - degree.astype(np.int32)
            
            # Compute interaction matrix
            # diff_mask[i,j] is 1 if labels[i] != labels[j]
            diff_mask = (labels[None, :] != labels[:, None])
            # interaction[i,j] = 2 * adj[i,j] * (1 if diff else -1)
            # = 2 * adj[i,j] * (2 * diff_mask[i,j] - 1)
            interaction = 2 * adj * (2 * diff_mask.astype(np.int8) - 1)
            
            # Total gain matrix for 2-flips
            # G[i,j] = single_gains[i] + single_gains[j] + interaction[i,j]
            # Note: G[i,j] = G[j,i]. We only need i < j.
            gain_matrix = single_gains[:, None] + single_gains[None, :] + interaction
            
            # Set diagonal to 0 (can't flip same node twice)
            np.fill_diagonal(gain_matrix, 0)
            
            # Find max gain
            max_idx = np.unravel_index(np.argmax(gain_matrix), gain_matrix.shape)
            max_gain = gain_matrix[max_idx]
            
            if max_gain <= 0:
                break
            
            i, j = max_idx
            if i == j:
                break
                
            # Apply flip
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            current_cut += max_gain
            
        return labels, current_cut

    def simulated_annealing(labels, initial_temp=15.0, cooling=0.999, max_iters=20000):
        """
        Simulated Annealing with 1-flip moves.
        """
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1).astype(np.int32)
        
        temp = initial_temp
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            v = np.random.randint(n)
            gain = 2 * same_count[v] - degree[v]
            
            if gain >= 0 or np.random.rand() < np.exp(gain / temp):
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gain
                
                new_label = 1 - old_label
                neighbors = neighbors_list[v]
                if len(neighbors) > 0:
                    sign = 2 * new_label - 1
                    delta_same = (2 * labels[neighbors] - 1) * sign
                    same_count[neighbors] += delta_same
                
                same_count[v] = degree[v] - same_count[v]
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            temp *= cooling
        
        return best_labels, best_cut

    # Multiple restarts
    best_labels_overall = None
    best_cut_overall = -1
    rng = np.random.RandomState(42)
    
    def try_restart(labels_init):
        nonlocal best_labels_overall, best_cut_overall
        if time.time() > deadline:
            return
        
        labels = labels_init.copy()
        
        # 1. Polish with 1-flip local search
        labels, cut = fast_local_search(labels, max_rounds=200)
        
        # 2. Escape with 2-flip local search
        if time.time() < deadline - 10:
            labels2, cut2 = vectorized_two_flip_search(labels, max_rounds=50)
            if cut2 > cut:
                labels, cut = labels2, cut2
                # Polish again
                labels, cut = fast_local_search(labels, max_rounds=100)
        
        if cut > best_cut_overall:
            best_cut_overall = cut
            best_labels_overall = labels.copy()

    # Spectral Initialization
    spectral_labels = None
    try:
        adj_sparse = csr_matrix(adj)
        L = sp_laplacian(adj_sparse, normed=False)
        
        # Get Fiedler vector
        try:
            eigvals, eigvecs = eigsh(L, k=2, which='SM')
            idx = np.argsort(eigvals)
            fiedler_vec = eigvecs[:, idx[1]]
        except Exception:
            fiedler_vec = rng.randn(n)
        
        # Threshold at median
        median_val = np.median(fiedler_vec)
        spectral_labels = (fiedler_vec > median_val).astype(np.int8)
    except Exception:
        spectral_labels = None
    
    # Restart 1: Spectral
    if spectral_labels is not None:
        try_restart(spectral_labels)
    
    # Restart 2: Random
    try_restart(rng.randint(0, 2, size=n))
    
    # Restart 3: Alternating
    try_restart(np.arange(n) % 2)
    
    # Restart 4: Random with bias
    try_restart(rng.randint(0, 2, size=n))
    
    # If time remains, run SA on best solution
    if best_labels_overall is not None and time.time() < deadline - 15:
        labels_sa = best_labels_overall.copy()
        labels_sa, cut_sa = simulated_annealing(labels_sa, initial_temp=20.0, cooling=0.998, max_iters=50000)
        labels_sa, cut_sa = fast_local_search(labels_sa, max_rounds=200)
        
        if cut_sa > best_cut_overall:
            best_cut_overall = cut_sa
            best_labels_overall = labels_sa.copy()
    
    # Final polish
    if best_labels_overall is not None and time.time() < deadline - 2:
        labels_final = best_labels_overall.copy()
        labels_final, cut_final = fast_local_search(labels_final, max_rounds=500)
        if cut_final > best_cut_overall:
            best_cut_overall = cut_final
            best_labels_overall = labels_final.copy()

    return best_labels_overall
# EVOLVE-BLOCK-END
