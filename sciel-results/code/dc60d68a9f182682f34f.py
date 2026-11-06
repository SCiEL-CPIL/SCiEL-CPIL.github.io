"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def partition(adjacency):
    """
    Solve Maximum Cut using Spectral Partitioning initialization and
    Kernighan-Lin refinement with multi-start.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph.
        
    Returns:
        np.array of 0/1 labels per vertex.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        # Efficient calculation:
        # Cut size = 0.5 * sum_v (degree[v] - 2 * same_side_neighbors[v])
        # Or: 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # Using matrix multiplication:
        # Let L be labels vector (0/1).
        # Crossing edges incident to v:
        # if labels[v]=0: sum(adj[v, labels=1])
        # if labels[v]=1: sum(adj[v, labels=0])
        # Total cut = 0.5 * (sum_{v:0} adj[v,1] + sum_{v:1} adj[v,0])
        
        # Vectorized approach:
        # diff = 1 - 2*abs(labels - 0.5) ... no
        # labels_diff_matrix = 1 - 2 * np.abs(labels[:, None] - labels[None, :]) ... too slow/memory heavy for large N, but N=100 is fine.
        # Better:
        # adj_labels = adj @ labels
        # same_side = adj_labels # sum of labels of neighbors
        # diff_side = degrees - adj_labels
        # crossing_per_vertex = np.where(labels == 1, diff_side, same_side)
        # return int(np.sum(crossing_per_vertex) / 2)
        
        adj_labels = adj @ labels
        crossing_per_vertex = np.where(labels == 1, degrees - adj_labels, adj_labels)
        return int(np.sum(crossing_per_vertex) / 2)

    def kernighan_lin(labels):
        """
        Kernighan-Lin refinement.
        Returns improved labels and cut size.
        """
        labels = labels.copy()
        # Initial gain for moving vertex v to other side
        # gain[v] = (neighbors in opposite set) - (neighbors in same set)
        # If labels[v] = 0: gain = sum(adj[v, labels=1]) - sum(adj[v, labels=0])
        # If labels[v] = 1: gain = sum(adj[v, labels=0]) - sum(adj[v, labels=1))
        
        def compute_gains(labels):
            adj_labels = adj @ labels
            # sum of labels of neighbors
            # If labels[v]=0, neighbors in opposite (1) are adj_labels[v], neighbors in same (0) are degrees[v] - adj_labels[v]
            # gain = opposite - same
            # If labels[v]=0: gain = adj_labels[v] - (degrees[v] - adj_labels[v]) = 2*adj_labels[v] - degrees[v]
            # If labels[v]=1: gain = (degrees[v] - adj_labels[v]) - adj_labels[v] = degrees[v] - 2*adj_labels[v]
            gains = np.where(labels == 1, degrees - 2 * adj_labels, 2 * adj_labels - degrees)
            return gains

        gains = compute_gains(labels)
        current_cut = cut_size(labels)
        
        # We will perform swaps.
        # Standard KL:
        # 1. Compute gains.
        # 2. Find pair (u, v) with max gain[u] + gain[v] - 2*adj[u,v].
        # 3. If max > 0, swap u, v. Update gains.
        # 4. Repeat.
        
        # To avoid O(N^2) search every time which is fine for N=100, 
        # we can just do it.
        
        improved = True
        while improved:
            improved = False
            # Find best pair
            best_gain = -1
            best_u, best_v = -1, -1
            
            # Optimization: Only consider pairs where one is 0 and one is 1?
            # No, KL swaps any two vertices. If they are in same set, swapping them doesn't change the cut?
            # Wait. If u and v are in the same set, swapping their labels puts them in the other set.
            # The change in cut depends on their connections to the rest of the graph.
            # However, standard KL usually considers swapping any pair.
            # Actually, if u and v are in the same set, and we swap them, they both move to the other set.
            # This is equivalent to moving u to other set AND v to other set.
            
            # Let's iterate over all pairs. N=100 -> 10000 pairs. Very fast.
            for u in range(n):
                if u == -1: continue
                for v in range(u + 1, n):
                    # Gain of swapping u and v
                    # If we swap u and v:
                    # 1. u moves to other set.
                    # 2. v moves to other set.
                    # The gain is roughly gains[u] + gains[v] - 2*adj[u,v]
                    # Why? 
                    # gain[u] is change if u moves alone.
                    # gain[v] is change if v moves alone.
                    # If they are connected, moving u alone changes the gain of v (and vice versa).
                    # The term -2*adj[u,v] accounts for the edge between them.
                    # If adj[u,v]=1:
                    #   Before: if same side, edge not cut. After: if swapped, they are still same side? 
                    #   Wait. If u=0, v=0. Swap -> u=1, v=1. Edge still not cut.
                    #   If u=0, v=1. Swap -> u=1, v=0. Edge still cut.
                    #   So the edge between u and v does NOT change its cut status if we swap them?
                    #   Let's check.
                    #   Case 1: u=0, v=0. Edge (u,v) is internal. After swap u=1, v=1. Edge is internal. No change.
                    #   Case 2: u=0, v=1. Edge (u,v) is crossing. After swap u=1, v=0. Edge is crossing. No change.
                    #   So the edge between u and v contributes 0 to the change in cut size when swapping u and v.
                    #   
                    #   However, the *gains* calculated initially assumed the other vertex stayed put.
                    #   gain[u] = delta when u moves, v stays.
                    #   gain[v] = delta when v moves, u stays.
                    #   
                    #   If u and v are connected:
                    #   When u moves, it affects v's environment.
                    #   Specifically, if u moves from 0 to 1:
                    #     If v was 0, v now has one more neighbor in 1 (u). So v's "same side" count increases by 1, "opposite" decreases by 1?
                    #     No, v is 0. u was 0 (same). Now u is 1 (opposite).
                    #     So for v, the edge (u,v) changes from same to opposite.
                    #     This changes v's gain.
                    #     
                    #   The formula gain[u] + gain[v] - 2*adj[u,v] is the standard KL pair gain.
                    #   Let's verify.
                    #   Delta = gain[u] + gain[v] - 2 * adj[u,v]
                    
                    pair_gain = gains[u] + gains[v] - 2 * adj[u, v]
                    if pair_gain > best_gain:
                        best_gain = pair_gain
                        best_u, best_v = u, v
            
            if best_gain > 0:
                # Perform swap
                labels[best_u], labels[best_v] = labels[best_v], labels[best_u]
                current_cut += best_gain
                # Update gains
                # Instead of full recomputation, we can update, but for N=100, full recomputation is cheap.
                gains = compute_gains(labels)
                improved = True
            else:
                break
        
        return labels, current_cut

    best_labels = None
    best_cut = -1

    # Start 1: Spectral Initialization
    # Construct Laplacian L = D - A
    L = np.diag(degrees) - adj
    
    # Compute eigenvalues and eigenvectors
    # N=100, so full eigh is fine
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        # Index 0 is the smallest (0 for connected graph)
        fiedler_vec = eigvecs[:, 1]
        
        # Initialize labels based on sign of Fiedler vector
        # Threshold at 0 or mean? Mean is better to balance sizes.
        threshold = np.mean(fiedler_vec)
        labels_spectral = (fiedler_vec > threshold).astype(int)
        
        # Refine with KL
        labels_refined, cut_refined = kernighan_lin(labels_spectral)
        
        if cut_refined > best_cut:
            best_cut = cut_refined
            best_labels = labels_refined.copy()
    except:
        pass # Fallback to random if spectral fails

    # Start 2: Random Initialization
    for _ in range(5):
        labels_random = np.random.randint(0, 2, n)
        labels_refined, cut_refined = kernighan_lin(labels_random)
        
        if cut_refined > best_cut:
            best_cut = cut_refined
            best_labels = labels_refined.copy()

    # Start 3: Greedy Initialization
    labels_greedy = np.zeros(n, dtype=int)
    order = list(range(n))
    random.shuffle(order)
    for v in order:
        neighbor_labels = labels_greedy[adj[v] == 1]
        if len(neighbor_labels) == 0:
            labels_greedy[v] = random.randint(0, 1)
        else:
            same_0 = np.sum(neighbor_labels == 0)
            same_1 = np.sum(neighbor_labels == 1)
            labels_greedy[v] = 0 if same_0 > same_1 else 1
            
    labels_refined, cut_refined = kernighan_lin(labels_greedy)
    if cut_refined > best_cut:
        best_cut = cut_refined
        best_labels = labels_refined.copy()

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
