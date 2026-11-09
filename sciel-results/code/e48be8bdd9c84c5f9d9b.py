"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels, max_passes=20):
    """Greedy local improvement: flip vertices that increase cut size."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    row_sums = A.sum(axis=1)
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for v in range(n):
            if labels[v] == 0:
                same_set = (labels == 0)
                same_set[v] = False
            else:
                same_set = (labels == 1)
                same_set[v] = False
            
            edges_to_same = A[v, same_set].sum()
            gain = 2 * edges_to_same - row_sums[v]
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels


def _spectral_partition(adjacency):
    """Use spectral method to get initial partition via Fiedler vector."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A  # Laplacian
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler >= 0).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, n)
        return labels
    except Exception:
        return np.random.randint(0, 2, n)


def _adaptive_sa(adjacency, labels, t0, t_min, deadline, max_iter=10000):
    """
    Simulated annealing with:
      - O(n) gain computation for 1-flip and 2-flip moves
      - adaptive cooling based on acceptance ratio
      - 1-flip and 2-flip moves to escape deeper local minima
    Returns (best_labels, best_cut).
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    row_sums = A.sum(axis=1)
    
    current = labels.copy()
    current_cut = _cut_size(adjacency, current)
    best = current.copy()
    best_cut = current_cut
    
    T = t0
    accept_count = 0
    total_count = 0
    window_size = 200
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Adaptive cooling: adjust T based on recent acceptance rate
        if total_count >= window_size:
            ratio = accept_count / total_count
            if ratio < 0.1:
                T *= 0.9
            elif ratio > 0.5:
                T *= 1.05
            else:
                T *= 0.97
            T = max(T, t_min)
            accept_count = 0
            total_count = 0
        
        if T <= t_min:
            # At minimum temperature, only accept improvements (handled below)
            pass
        
        # Choose move type: 1-flip (70%) or 2-flip (30%)
        use_two_flip = (np.random.random() < 0.3)
        
        if not use_two_flip:
            # 1-flip move: O(n) gain computation
            v = np.random.randint(n)
            
            if current[v] == 0:
                same_mask = (current == 0)
            else:
                same_mask = (current == 1)
            same_mask[v] = False
            edges_to_same = A[v, same_mask].sum()
            delta = 2 * edges_to_same - row_sums[v]
            
            new_labels = current.copy()
            new_labels[v] = 1 - current[v]
        else:
            # 2-flip move: pick two distinct vertices
            v1, v2 = np.random.choice(n, size=2, replace=False)
            
            # Compute combined gain for flipping both v1 and v2
            # We need to account for the interaction between v1 and v2
            # Gain for flipping v1 alone (given current state)
            if current[v1] == 0:
                same_mask_1 = (current == 0)
            else:
                same_mask_1 = (current == 1)
            same_mask_1[v1] = False
            edges_to_same_1 = A[v1, same_mask_1].sum()
            gain_1 = 2 * edges_to_same_1 - row_sums[v1]
            
            # Gain for flipping v2 alone (given current state)
            if current[v2] == 0:
                same_mask_2 = (current == 0)
            else:
                same_mask_2 = (current == 1)
            same_mask_2[v2] = False
            edges_to_same_2 = A[v2, same_mask_2].sum()
            gain_2 = 2 * edges_to_same_2 - row_sums[v2]
            
            # Interaction term: if v1 and v2 are in the same set, flipping both
            # changes the edge between them from non-crossing to crossing (if edge exists)
            # If they are in different sets, flipping both changes it from crossing to non-crossing
            # The interaction is -2 * A[v1, v2] if they start in same set, +2 * A[v1, v2] if different
            # More precisely:
            # Current: edge (v1,v2) contributes to cut iff current[v1] != current[v2]
            # After flip: edge (v1,v2) contributes to cut iff (1-current[v1]) != (1-current[v2]) iff current[v1] == current[v2]
            # So delta_interaction = (1 if current[v1] == current[v2] else -1) * 2 * A[v1,v2]
            # Wait, let me think more carefully.
            # 
            # Let's just compute the gain directly by considering the effect on cut size.
            # The cut size changes by:
            # For each vertex u, the contribution to cut from edges incident to u changes.
            # 
            # Actually, the simplest correct approach: 
            # delta = gain_1 + gain_2 - 2 * A[v1, v2] * (1 if current[v1] == current[v2] else -1)
            # 
            # Let me verify: 
            # If v1 and v2 are in same set (say both 0):
            #   - Flipping v1: edges from v1 to set 1 become crossing, edges from v1 to set 0 (excluding v1) stop being crossing. But v2 is in set 0, so edge (v1,v2) stops being crossing... wait, it wasn't crossing before either since both in set 0.
            #   - Actually, gain_1 already accounts for all edges from v1. When we flip both, the edge (v1,v2) is handled in both gain_1 and gain_2.
            #   
            # Let me just use the direct computation for 2-flips since it's only O(n) per move anyway.
            new_labels = current.copy()
            new_labels[v1] = 1 - current[v1]
            new_labels[v2] = 1 - current[v2]
            
            # Compute delta efficiently:
            # The change in cut size when flipping v1 and v2 simultaneously
            # For any vertex u not v1 or v2:
            #   Edge (u, v1): was crossing iff current[u] != current[v1], after: crossing iff current[u] != (1-current[v1]) = (current[u] == current[v1])
            #   So edge (u,v1) changes by: +1 if current[u] == current[v1] (was non-crossing, becomes crossing), -1 if current[u] != current[v1] (was crossing, becomes non-crossing)
            #   Net change for edge (u,v1): 2*A[u,v1]*(current[u]==current[v1]) - A[u,v1] = A[u,v1]*(2*(current[u]==current[v1]) - 1)
            #   Similarly for edge (u,v2): A[u,v2]*(2*(current[u]==current[v2]) - 1)
            # For edge (v1, v2):
            #   Was crossing iff current[v1] != current[v2]
            #   After: crossing iff (1-current[v1]) != (1-current[v2]) iff current[v1] == current[v2]
            #   Change: +1 if current[v1] == current[v2] (was non-crossing, becomes crossing), -1 if current[v1] != current[v2]
            #   Net: A[v1,v2]*(2*(current[v1]==current[v2]) - 1)
            
            # Total delta = sum over all u != v1,v2 of [A[u,v1]*(2*(current[u]==current[v1]) - 1) + A[u,v2]*(2*(current[u]==current[v2]) - 1)] 
            #              + A[v1,v2]*(2*(current[v1]==current[v2]) - 1)
            
            # This is O(n) to compute
            # Let's compute it:
            # For v1: sum over u != v1,v2 of A[u,v1]*(2*(current[u]==current[v1]) - 1)
            #        = sum over u != v1 of A[u,v1]*(2*(current[u]==current[v1]) - 1) - A[v2,v1]*(2*(current[v2]==current[v1]) - 1)
            #        = [sum over all u of A[u,v1]*(2*(current[u]==current[v1]) - 1)] - A[v2,v1]*(2*(current[v2]==current[v1]) - 1)
            
            # sum over all u of A[u,v1]*(2*(current[u]==current[v1]) - 1)
            # = 2*sum_{u: current[u]==current[v1]} A[u,v1] - sum_{u: current[u]!=current[v1]} A[u,v1]
            # = 2*edges_to_same_1 - (row_sums[v1] - edges_to_same_1)  [since row_sums[v1] = sum of all A[u,v1]]
            # = 3*edges_to_same_1 - row_sums[v1]
            # Wait, that doesn't match gain_1 = 2*edges_to_same_1 - row_sums[v1]
            
            # Let me re-derive. The gain from flipping v1 alone is:
            # delta_1 = 2*edges_to_same_1 - row_sums[v1]
            # This is the change in cut size when flipping v1 in the current configuration.
            
            # When flipping both v1 and v2, the total delta is not simply delta_1 + delta_2 because of the interaction on edge (v1,v2).
            # 
            # Let's use a cleaner approach. The cut size C(labels) = sum_{i<j, labels[i]!=labels[j]} A[i,j]
            # When we flip v1 and v2, the new cut size is C(new_labels).
            # 
            # The change in C is:
            # For edges not incident to v1 or v2: no change.
            # For edge (v1, v2): change by A[v1,v2] * (indicator that (1-c1) != (1-c2) minus indicator that c1 != c2)
            #   = A[v1,v2] * (indicator(c1==c2) - indicator(c1!=c2))
            #   = A[v1,v2] * (2*indicator(c1==c2) - 1)
            # For edge (u, v1) where u != v2: change by A[u,v1] * (indicator(c_u != (1-c1)) - indicator(c_u != c1))
            #   = A[u,v1] * (indicator(c_u == c1) - indicator(c_u != c1))
            #   = A[u,v1] * (2*indicator(c_u == c1) - 1)
            # For edge (u, v2) where u != v1: change by A[u,v2] * (2*indicator(c_u == c2) - 1)
            
            # So total delta = A[v1,v2]*(2*I(c1==c2) - 1) 
            #                 + sum_{u != v1,v2} A[u,v1]*(2*I(c_u==c1) - 1)
            #                 + sum_{u != v1,v2} A[u,v2]*(2*I(c_u==c2) - 1)
            
            # Now, sum_{u != v1,v2} A[u,v1]*(2*I(c_u==c1) - 1)
            # = sum_{u != v1} A[u,v1]*(2*I(c_u==c1) - 1) - A[v2,v1]*(2*I(c2==c1) - 1)
            # = [sum_{u != v1} A[u,v1]*(2*I(c_u==c1) - 1)] - A[v1,v2]*(2*I(c1==c2) - 1)
            
            # And sum_{u != v1} A[u,v1]*(2*I(c_u==c1) - 1)
            # = 2*sum_{u != v1, c_u==c1} A[u,v1] - sum_{u != v1, c_u!=c1} A[u,v1]
            # = 2*edges_to_same_1 - (row_sums[v1] - edges_to_same_1)
            # = 3*edges_to_same_1 - row_sums[v1]
            
            # Wait, but gain_1 = 2*edges_to_same_1 - row_sums[v1]. Let me check.
            # When flipping v1 alone, the change in cut is:
            # For edge (u,v1) where u != v1: change by A[u,v1]*(I(c_u != (1-c1)) - I(c_u != c1)) = A[u,v1]*(I(c_u==c1) - I(c_u!=c1)) = A[u,v1]*(2*I(c_u==c1)-1)
            # Total: sum_{u != v1} A[u,v1]*(2*I(c_u==c1)-1) = 3*edges_to_same_1 - row_sums[v1]
            # But the standard formula says gain = 2*edges_to_same - row_sums. Let me recheck.
            # 
            # Actually, I think I'm confusing myself. Let me just compute delta for 2-flip by direct O(n) summation.
            
            c1 = current[v1]
            c2 = current[v2]
            
            # Sum over u != v1, v2 of A[u,v1]*(2*I(c_u==c1) - 1)
            # = sum over all u of A[u,v1]*(2*I(c_u==c1) - 1) - A[v1,v1]*(2*I(c1==c1)-1) - A[v2,v1]*(2*I(c2==c1)-1)
            # A[v1,v1] = 0 (no self-loops in unweighted graph, and adjacency is symmetric with 0 diagonal)
            # = sum over all u of A[u,v1]*(2*I(c_u==c1) - 1) - A[v2,v1]*(2*I(c2==c1)-1)
            
            # sum over all u of A[u,v1]*(2*I(c_u==c1) - 1)
            # Let S_same = sum_{u: c_u==c1} A[u,v1] = edges_to_same_1 (including u=v1? No, A[v1,v1]=0)
            # Let S_diff = sum_{u: c_u!=c1} A[u,v1] = row_sums[v1] - edges_to_same_1
            # So sum = 2*S_same - S_diff = 2*edges_to_same_1 - (row_sums[v1] - edges_to_same_1) = 3*edges_to_same_1 - row_sums[v1]
            
            term_v1 = 3 * edges_to_same_1 - row_sums[v1] - A[v1, v2] * (2 * (1 if c2 == c1 else 0) - 1)
            
            # Similarly for v2
            term_v2 = 3 * edges_to_same_2 - row_sums[v2] - A[v2, v1] * (2 * (1 if c1 == c2 else 0) - 1)
            
            # Edge (v1,v2) contribution
            edge_v1_v2 = A[v1, v2] * (2 * (1 if c1 == c2 else 0) - 1)
            
            delta = term_v1 + term_v2 + edge_v1_v2
            
            # Simplify: A[v1,v2] = A[v2,v1]
            # term_v1 = 3*edges_to_same_1 - row_sums[v1] - A[v1,v2]*(2*I(c1==c2)-1)
            # term_v2 = 3*edges_to_same_2 - row_sums[v2] - A[v1,v2]*(2*I(c1==c2)-1)
            # edge_v1_v2 = A[v1,v2]*(2*I(c1==c2)-1)
            # So delta = 3*edges_to_same_1 - row_sums[v1] + 3*edges_to_same_2 - row_sums[v2] - A[v1,v2]*(2*I(c1==c2)-1)
            
            # Let me just use the simpler direct computation to avoid errors:
            # delta = (gain_1 + gain_2) - 2 * A[v1,v2] * (1 if c1 == c2 else -1)
            # Let me verify this formula.
            
            # Actually, let's just compute it the safe way:
            delta = gain_1 + gain_2 - 2 * A[v1, v2] * (1 if c1 == c2 else -1)
        
        total_count += 1
        
        # Accept/reject
        if delta > 0:
            current = new_labels
            current_cut = current_cut + int(delta)
            if current_cut > best_cut:
                best_cut = current_cut
                best = current.copy()
            accept_count += 1
        else:
            if T > t_min:
                # Boltzmann acceptance
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current = new_labels
                    current_cut = current_cut + int(delta)
                    accept_count += 1
    
    return best, best_cut


def partition(adjacency):
    """Find maximum cut using spectral init + adaptive SA + local search."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    n = len(adjacency)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization + local search
    if time.time() < deadline - 20:
        labels = _spectral_partition(adjacency)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 2: Random initialization + local search
    if time.time() < deadline - 20:
        labels = np.random.randint(0, 2, n)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 3: Adaptive SA from best current state
    if best_labels is not None and time.time() < deadline - 15:
        sa_labels, sa_cut = _adaptive_sa(
            adjacency, best_labels, t0=5.0, t_min=0.01,
            deadline=deadline - 10, max_iter=20000
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Polish with local search
        if time.time() < deadline - 5:
            best_labels = _local_search(adjacency, best_labels)
            best_cut = _cut_size(adjacency, best_labels)
    
    # Phase 4: Multiple random restarts with adaptive SA
    restarts = 0
    while time.time() < deadline - 15 and restarts < 8:
        labels = np.random.randint(0, 2, n)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Run adaptive SA from this local optimum
        if time.time() < deadline - 10:
            sa_labels, sa_cut = _adaptive_sa(
                adjacency, labels, t0=3.0, t_min=0.01,
                deadline=deadline - 10, max_iter=15000
            )
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
            
            # Final polish
            if time.time() < deadline - 5:
                best_labels = _local_search(adjacency, best_labels)
                best_cut = _cut_size(adjacency, best_labels)
        
        restarts += 1
    
    # Final polish pass with more iterations for high-precision convergence
    if best_labels is not None and time.time() < deadline - 2:
        best_labels = _local_search(adjacency, best_labels, max_passes=50)
        best_cut = _cut_size(adjacency, best_labels)
    
    return best_labels
# EVOLVE-BLOCK-END
