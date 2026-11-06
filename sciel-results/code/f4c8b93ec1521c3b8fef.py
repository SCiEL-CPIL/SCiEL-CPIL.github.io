"""Simulated Annealing for Maximum Cut with vectorized operations."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses vectorized Simulated Annealing with multiple restarts.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    # Precompute degrees for efficient gain calculation
    degrees = adj.sum(axis=1)
    
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    def compute_cut(labels):
        # labels is 0/1 array
        # Cut edges are those where labels[i] != labels[j] and adj[i][j] == 1
        # Using matrix multiplication trick:
        # Let x be vector where x_i = 1 if label_i=0, -1 if label_i=1
        # Then cut_size = (M_total - x^T A x) / 4 ? No, simpler:
        # cut = sum_{i<j, Aij=1} (1 - label_i*label_j) ... wait labels are 0/1.
        # If label_i != label_j, product is 0. If same, product is 1.
        # So cut = total_edges - sum_{i<j, Aij=1} (label_i == label_j)
        # Let's use the standard trick:
        # Let s = 2*labels - 1 (so s is -1 or 1)
        # cut = (total_edges - sum(s_i * s_j * A_ij) / 2) / 2 ?
        # Actually:
        # sum_{i,j} A_ij s_i s_j = sum_{i,j} A_ij (2l_i-1)(2l_j-1)
        # = 4 sum A l_i l_j - 2 sum A l_i - 2 sum A l_j + sum A
        # This is getting complex. Let's just compute directly for simplicity in inner loop if needed,
        # but for SA we need fast evaluation.
        
        # Fast vectorized cut calculation:
        # Cut = sum_{i} sum_{j} A[i,j] * (1 if l[i]!=l[j] else 0)
        # Note: A is symmetric.
        # Let's use: cut = 0.5 * (sum_i (sum_j A[i,j]) - sum_i (sum_j A[i,j] * (2*l[i]-1) * (2*l[j]-1)) / 2 )
        # Actually, simpler:
        # s = 2*labels - 1
        # Q = s @ adj @ s
        # Q = sum_{i,j} s_i A_ij s_j
        # Q = sum_{i != j, A_ij=1} s_i s_j + sum_{i=j} 0 (since no self loops assumed, or A_ii=0)
        # If s_i s_j = 1 (same side), contributes 1. If -1 (diff side), contributes -1.
        # Let N_same = # edges within partitions. N_diff = # edges crossing.
        # Total edges M = N_same + N_diff.
        # Q = N_same - N_diff = N_same - (M - N_same) = 2*N_same - M.
        # So N_diff = (M - Q) / 2.
        
        s = 2.0 * labels - 1.0
        Q = s @ adj @ s
        M = adj.sum() / 2.0  # Total edges
        return (M - Q) / 2.0

    def local_search_sa(labels, time_limit):
        # Simulated Annealing
        # Start with current labels
        current_labels = labels.copy()
        current_cut = compute_cut(current_labels)
        
        # Initial temperature high enough to accept bad moves
        # Typical gain per move is small. Max possible gain is degree.
        # Let's set T0 = 1.0 and decay slowly.
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        start_time = time.time()
        
        while time.time() < time_limit and T > min_T:
            # Pick a random vertex
            v = np.random.randint(n)
            
            # Calculate gain of flipping v
            # If v is in group 0, it moves to 1.
            # Edges from v to group 0 become crossing (gain +1 each)
            # Edges from v to group 1 become non-crossing (loss -1 each)
            # Net gain = (# neighbors in group 1) - (# neighbors in group 0)
            
            # Vectorized gain calculation
            # neighbors = adj[v]
            # group_0_count = np.sum(neighbors * (current_labels == 0))
            # group_1_count = np.sum(neighbors * (current_labels == 1))
            # gain = group_1_count - group_0_count
            
            # Let's use the s-vector approach for consistency and speed
            s = 2.0 * current_labels - 1.0
            # Gain in Q when flipping v (s_v -> -s_v):
            # Delta Q = -2 * s_v * sum_{u != v} A[v,u] s_u
            # Since s_v is +/- 1, Delta Q = -2 * s_v * (s @ adj @ e_v) ... wait.
            # Let's just compute the change in cut size directly.
            
            # Cut size change:
            # If we flip v, edges (v,u) change status if A[v,u]=1.
            # If u is on same side as v, edge was NOT cut, now IS cut (+1).
            # If u is on diff side as v, edge WAS cut, now IS NOT cut (-1).
            
            # Count neighbors on same side vs diff side
            # same_side_mask = (current_labels == current_labels[v])
            # diff_side_mask = (current_labels != current_labels[v])
            
            # Vectorized:
            # same_count = np.dot(adj[v], (current_labels == current_labels[v]).astype(np.int8))
            # diff_count = np.dot(adj[v], (current_labels != current_labels[v]).astype(np.int8))
            # gain = diff_count - same_count  # Wait.
            # If v is 0. Neighbors 0: same. Neighbors 1: diff.
            # Flip v to 1.
            # Neighbors 0: now diff. Edge becomes cut. +1.
            # Neighbors 1: now same. Edge becomes non-cut. -1.
            # So gain = (# neighbors 0) - (# neighbors 1).
            
            # Let's generalize:
            # gain = sum_u A[v,u] * (1 if label_u == label_v else -1)
            # This is exactly:
            # gain = sum_u A[v,u] * (2*label_u - 1) * (2*label_v - 1) ? No.
            # If label_u == label_v, term is 1. If diff, term is -1.
            # Let s_u = 2*l_u - 1, s_v = 2*l_v - 1.
            # If l_u == l_v, s_u == s_v, product 1.
            # If l_u != l_v, s_u != s_v, product -1.
            # So gain = sum_u A[v,u] s_u s_v = s_v * sum_u A[v,u] s_u.
            
            s = 2.0 * current_labels - 1.0
            # sum_u A[v,u] s_u is the v-th element of (adj @ s)
            neighbor_sum = adj @ s
            gain = s[v] * neighbor_sum[v]
            
            # gain is the change in Cut Size?
            # Let's verify.
            # Q = sum s_i A_ij s_j.
            # Flip v: s_v -> -s_v.
            # New Q = Q - 2 * s_v * sum_{u != v} A[v,u] s_u.
            # Note sum_{u} A[v,u] s_u includes u=v term which is 0 (no self loop).
            # So Delta Q = -2 * s_v * neighbor_sum[v].
            # Delta Cut = -0.5 * Delta Q = s_v * neighbor_sum[v].
            # Yes, gain calculated above is Delta Cut.
            
            if gain >= 0:
                # Always accept improving or neutral moves
                current_labels[v] = 1 - current_labels[v]
                current_cut += gain
            else:
                # Accept with probability exp(gain / T)
                if np.random.random() < np.exp(gain / T):
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
            
            T *= cooling_rate
            
            # Check time
            if time.time() > time_limit:
                break
            
        return current_labels, current_cut

    # Multiple restarts
    # We have ~115 seconds.
    # Let's do 50 restarts, each with ~2 seconds budget? Or dynamic.
    # 100 nodes is small. SA converges fast.
    # Let's try 100 restarts, each with 1 second budget.
    
    n_restarts = 50
    time_per_restart = 110.0 / n_restarts
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        labels = np.random.randint(0, 2, n).astype(np.int8)
        
        # Run SA
        labels, cut = local_search_sa(labels, time.time() + time_per_restart)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
